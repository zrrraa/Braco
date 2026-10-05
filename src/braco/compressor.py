"""PyTorch implementation of the Braco visual-token compressor.

The module compresses a square visual-token grid into two groups:

1. a low-frequency DCT backbone with ``cutoff**2`` tokens; and
2. learned sparse spatial residuals that preserve local detail.

The implementation is independent of LLaVA. The integration used by the
experiments is provided under ``integrations/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from entmax import sparsemax
from torch import Tensor, nn


CoordinateMode = Literal["coefficients", "idct"]


@dataclass(frozen=True)
class BracoConfig:
    """Configuration for one retained-token budget."""

    cutoff: int
    residual_tokens: int
    coordinate_mode: CoordinateMode = "coefficients"

    @property
    def output_tokens(self) -> int:
        return self.cutoff**2 + self.residual_tokens


BRACO_BUDGETS: dict[int, BracoConfig] = {
    4: BracoConfig(cutoff=1, residual_tokens=3),
    9: BracoConfig(cutoff=2, residual_tokens=5),
    16: BracoConfig(cutoff=3, residual_tokens=7),
    25: BracoConfig(cutoff=4, residual_tokens=9, coordinate_mode="idct"),
}


class BracoCompressor(nn.Module):
    """Compress a square visual-token grid with a DCT backbone and residuals.

    Args:
        dim: Visual feature dimension.
        cutoff: Side length of the retained low-frequency DCT block.
        residual_tokens: Number of learned spatial residual tokens.
        coordinate_mode: Return DCT coefficients directly, or organize the
            retained subspace with a small inverse DCT.
        residual_hidden_multiplier: Hidden width of the residual scorer,
            expressed as a multiple of ``dim``.
        polar_frequencies: Number of radial/angular frequency bands used by
            the basis-coordinate embedding.
        polar_include_raw: Include ``[r, sin(theta), cos(theta)]`` alongside
            the sinusoidal basis-coordinate features. The main-table
            configuration uses ``False``.
        polar_scale: Initial scale of the basis-coordinate embedding.
        residual_scale: Initial scale of the spatial residual branch.
        residual_prior_scale: Initial scale of the local-gradient prior in the
            residual assignment logits.
        temperature: Sparse residual-assignment temperature.
        post_concat_norm: Apply an additional LayerNorm after concatenation
            (default: False). Spatial residuals have their own normalization
            and learned scale.
        post_concat_norm_eps: Epsilon used by the post-concatenation
            LayerNorm.
        detach_transform: Detach the visual input while computing its DCT.
            This saves activation memory when the vision encoder is frozen.
    """

    def __init__(
        self,
        dim: int,
        cutoff: int,
        residual_tokens: int,
        *,
        coordinate_mode: CoordinateMode = "coefficients",
        residual_hidden_multiplier: int = 1,
        polar_frequencies: int = 8,
        polar_include_raw: bool = False,
        polar_scale: float = 0.1,
        residual_scale: float = 1.0,
        residual_prior_scale: float = 0.0,
        temperature: float = 0.8,
        post_concat_norm: bool = False,
        post_concat_norm_eps: float = 1e-5,
        detach_transform: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if residual_tokens < 0:
            raise ValueError("residual_tokens must be non-negative")
        if coordinate_mode not in {"coefficients", "idct"}:
            raise ValueError(f"unknown coordinate mode: {coordinate_mode}")

        self.dim = int(dim)
        self.cutoff = int(cutoff)
        self.residual_tokens = int(residual_tokens)
        self.coordinate_mode = coordinate_mode
        self.polar_frequencies = int(polar_frequencies)
        self.polar_include_raw = bool(polar_include_raw)
        self.post_concat_norm = bool(post_concat_norm)
        self.detach_transform = bool(detach_transform)
        self.register_buffer(
            "temperature",
            torch.tensor(float(temperature)),
            persistent=False,
        )

        polar_dim = (
            (3 if self.polar_include_raw else 0)
            + 4 * self.polar_frequencies
        )
        self.polar_projection = nn.Linear(polar_dim, self.dim, bias=False)
        self.polar_scale = nn.Parameter(torch.tensor(float(polar_scale)))

        if self.residual_tokens:
            hidden_dim = max(1, self.dim * int(residual_hidden_multiplier))
            self.residual_scorer = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.residual_tokens),
            )
            self.residual_norm = nn.LayerNorm(self.dim)
            self.residual_scale = nn.Parameter(
                torch.tensor(float(residual_scale))
            )
            self.residual_prior_scale = nn.Parameter(
                torch.tensor(float(residual_prior_scale))
            )

        if self.post_concat_norm:
            self.post_norm = nn.LayerNorm(
                self.dim, eps=float(post_concat_norm_eps)
            )

        self.register_buffer("_dct", torch.empty(0), persistent=False)
        self.register_buffer("_small_dct", torch.empty(0), persistent=False)
        self.register_buffer("_polar_features", torch.empty(0), persistent=False)

    @classmethod
    def from_budget(
        cls,
        dim: int,
        budget: int,
        **kwargs: object,
    ) -> "BracoCompressor":
        """Build one of the four configurations used in the main table."""

        try:
            config = BRACO_BUDGETS[int(budget)]
        except KeyError as error:
            supported = ", ".join(map(str, sorted(BRACO_BUDGETS)))
            raise ValueError(f"budget must be one of: {supported}") from error
        return cls(
            dim=dim,
            cutoff=config.cutoff,
            residual_tokens=config.residual_tokens,
            coordinate_mode=config.coordinate_mode,
            **kwargs,
        )

    @property
    def output_tokens(self) -> int:
        return self.cutoff**2 + self.residual_tokens

    def set_temperature(self, temperature: float) -> None:
        """Set the residual-assignment temperature used by sparsemax."""

        self.temperature.fill_(max(float(temperature), 1e-6))

    @staticmethod
    def _orthonormal_dct(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(size, device=device, dtype=dtype).view(1, -1)
        frequencies = torch.arange(size, device=device, dtype=dtype).view(-1, 1)
        basis = torch.cos(math.pi / size * (positions + 0.5) * frequencies)
        basis[0] *= 1.0 / math.sqrt(size)
        basis[1:] *= math.sqrt(2.0 / size)
        return basis

    @staticmethod
    def _as_square_grid(tokens: Tensor) -> tuple[Tensor, int]:
        if tokens.ndim != 3:
            raise ValueError(f"expected [batch, tokens, dim], got {tokens.shape}")
        length = tokens.shape[1]
        side = math.isqrt(length)
        if side * side == length:
            return tokens, side

        side = math.isqrt(length - 1)
        if length > 1 and side * side == length - 1:
            return tokens[:, 1:, :], side
        raise ValueError(
            f"token count must be N^2 or 1+N^2, got {length}"
        )

    def _get_dct(self, size: int, reference: Tensor) -> Tensor:
        if (
            self._dct.shape != (size, size)
            or self._dct.device != reference.device
            or self._dct.dtype != reference.dtype
        ):
            self._dct = self._orthonormal_dct(
                size, reference.device, reference.dtype
            )
        return self._dct

    def _get_small_dct(self, reference: Tensor) -> Tensor:
        size = self.cutoff
        if (
            self._small_dct.shape != (size, size)
            or self._small_dct.device != reference.device
            or self._small_dct.dtype != reference.dtype
        ):
            self._small_dct = self._orthonormal_dct(
                size, reference.device, reference.dtype
            )
        return self._small_dct

    def _get_polar_features(self, side: int, device: torch.device) -> Tensor:
        expected_dim = (
            (3 if self.polar_include_raw else 0)
            + 4 * self.polar_frequencies
        )
        if (
            self._polar_features.shape == (side, side, expected_dim)
            and self._polar_features.device == device
        ):
            return self._polar_features

        row = torch.arange(side, device=device, dtype=torch.float32).view(side, 1)
        col = torch.arange(side, device=device, dtype=torch.float32).view(1, side)
        rows = row.expand(side, side)
        cols = col.expand(side, side)

        radius = torch.sqrt(rows.square() + cols.square())
        radius = radius / max(math.sqrt(2.0) * (side - 1), 1.0)
        angle = torch.atan2(cols, rows)
        angle = torch.where(
            (rows == 0) & (cols == 0), torch.zeros_like(angle), angle
        )

        bands = 2.0 ** torch.arange(
            self.polar_frequencies, device=device, dtype=torch.float32
        )
        radial_phase = math.pi * radius.unsqueeze(-1) * bands
        angular_phase = angle.unsqueeze(-1) * bands
        features = []
        if self.polar_include_raw:
            features.append(
                torch.stack((radius, angle.sin(), angle.cos()), dim=-1)
            )
        features.extend(
            (
                radial_phase.sin(),
                radial_phase.cos(),
                angular_phase.sin(),
                angular_phase.cos(),
            )
        )
        self._polar_features = torch.cat(features, dim=-1)
        return self._polar_features

    @staticmethod
    def _spatial_gradient_energy(tokens: Tensor, side: int) -> Tensor:
        batch, _, dim = tokens.shape
        grid = tokens.view(batch, side, side, dim).float()
        vertical = grid[:, 1:, :, :] - grid[:, :-1, :, :]
        horizontal = grid[:, :, 1:, :] - grid[:, :, :-1, :]
        vertical_energy = vertical.square().sum(dim=-1)
        horizontal_energy = horizontal.square().sum(dim=-1)

        energy = grid.new_zeros((batch, side, side))
        energy[:, 1:, :] += vertical_energy
        energy[:, :-1, :] += vertical_energy
        energy[:, :, 1:] += horizontal_energy
        energy[:, :, :-1] += horizontal_energy
        return energy.view(batch, side * side)

    def _spatial_residuals(self, tokens: Tensor, side: int) -> Tensor | None:
        if self.residual_tokens == 0:
            return None
        logits = self.residual_scorer(tokens).transpose(1, 2).contiguous()
        energy = self._spatial_gradient_energy(tokens, side)
        energy = (energy - energy.mean(dim=-1, keepdim=True)) / (
            energy.std(dim=-1, keepdim=True) + 1e-6
        )
        logits = logits + self.residual_prior_scale.to(logits.dtype) * energy.unsqueeze(1).to(logits.dtype)
        logits = (logits / max(float(self.temperature.item()), 1e-6)).float()
        weights = sparsemax(logits, dim=-1).to(tokens.dtype)
        residuals = torch.einsum("bsl,bld->bsd", weights, tokens)
        residuals = self.residual_norm(residuals)
        return residuals * self.residual_scale.to(residuals.dtype)

    def _low_frequency_tokens(self, tokens: Tensor, side: int) -> Tensor:
        batch, _, dim = tokens.shape
        basis = self._get_dct(side, tokens)
        grid = tokens.view(batch, side, side, dim)
        grid = grid.permute(0, 3, 1, 2).reshape(batch * dim, side, side)

        def transform() -> Tensor:
            return basis @ grid @ basis.transpose(0, 1)

        if self.detach_transform:
            with torch.no_grad():
                transformed = transform()
        else:
            transformed = transform()
        transformed = transformed.view(batch, dim, side, side)

        polar = self._get_polar_features(side, tokens.device)
        polar = self.polar_projection(
            polar.to(self.polar_projection.weight.dtype)
        )
        polar = polar.permute(2, 0, 1).unsqueeze(0).to(transformed.dtype)
        transformed = transformed + self.polar_scale * polar

        cutoff = min(self.cutoff, side)
        low = transformed[:, :, :cutoff, :cutoff]
        if self.coordinate_mode == "idct":
            basis_small = self._get_small_dct(low)
            flat = low.reshape(batch * dim, cutoff, cutoff)
            flat = basis_small.transpose(0, 1) @ flat @ basis_small
            low = flat.view(batch, dim, cutoff, cutoff)
        return low.permute(0, 2, 3, 1).reshape(batch, cutoff**2, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return exactly ``cutoff**2 + residual_tokens`` visual tokens."""

        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.dim}, "
                f"got {tokens.shape[-1]}"
            )
        grid_tokens, side = self._as_square_grid(tokens)
        if self.cutoff > side:
            raise ValueError(
                f"cutoff {self.cutoff} exceeds input grid side {side}"
            )

        backbone = self._low_frequency_tokens(grid_tokens, side)
        residuals = self._spatial_residuals(grid_tokens, side)
        output = (
            backbone if residuals is None else torch.cat((backbone, residuals), dim=1)
        )
        if self.post_concat_norm:
            output = self.post_norm(output)
        if output.shape[1] != self.output_tokens:
            raise RuntimeError(
                f"expected {self.output_tokens} output tokens, "
                f"got {output.shape[1]}"
            )
        return output
