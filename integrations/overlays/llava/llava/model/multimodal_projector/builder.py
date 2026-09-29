import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, entmax_bisect, sparsemax


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class FourierFreqSelect(nn.Module):
    """Braco visual-token compressor for LLaVA.

    Retains a ``C x C`` low-frequency DCT backbone and appends
    ``spatial_keep`` learned spatial residual tokens.
    """

    def __init__(
        self,
        in_dim,
        C,
        K=0,
        *,
        dct_detach=True,
        temperature=1.0,
        temp_start=None,
        temp_end=None,
        temp_anneal_steps=0,
        temp_schedule="cosine",
        use_polar_pe=True,
        polar_num_freqs=8,
        polar_include_raw=True,
        polar_pe_scale_init=0.1,
        spatial_keep=0,
        spatial_method="learned",
        spatial_use_norm=True,
        spatial_scale_init=1.0,
        spatial_hidden_mult=1,
        spatial_selection_norm="sparsemax",
        spatial_temp_mul=1.0,
        spatial_prior="grad",
        spatial_prior_weight_init=0.0,
        post_concat_norm=False,
        post_concat_norm_eps=1e-5,
        low_idct=False,
    ):
        super().__init__()
        if int(K) != 0:
            raise ValueError(
                "Set mm_fourier_K_high=0 for the Braco token configuration."
            )
        if int(C) <= 0:
            raise ValueError("C must be positive")
        if int(spatial_keep) < 0:
            raise ValueError("spatial_keep must be non-negative")
        if str(spatial_method).lower() != "learned":
            raise ValueError(
                "Set spatial_method='learned' for Braco residual pooling."
            )
        if str(spatial_prior).lower() not in {"none", "grad"}:
            raise ValueError("spatial_prior must be 'none' or 'grad'")

        self.in_dim = int(in_dim)
        self.C = int(C)
        self.K = 0
        self.dct_detach = bool(dct_detach)
        self.low_idct = bool(low_idct)

        self.temperature = float(temperature)
        self.temp_start = (
            None if temp_start is None else float(temp_start)
        )
        self.temp_end = None if temp_end is None else float(temp_end)
        self.temp_anneal_steps = int(temp_anneal_steps)
        self.temp_schedule = str(temp_schedule).lower()
        self.register_buffer(
            "temperature_cur",
            torch.tensor(self.temperature),
            persistent=False,
        )

        self.use_polar_pe = bool(use_polar_pe)
        self.polar_num_freqs = int(polar_num_freqs)
        self.polar_include_raw = bool(polar_include_raw)
        if self.use_polar_pe:
            polar_dim = (
                (3 if self.polar_include_raw else 0)
                + 4 * self.polar_num_freqs
            )
            self.polar_proj = nn.Linear(polar_dim, self.in_dim, bias=False)
            self.polar_pe_scale = nn.Parameter(
                torch.tensor(float(polar_pe_scale_init))
            )
            self.register_buffer(
                "polar_feat", torch.empty(0), persistent=False
            )

        self.spatial_keep = int(spatial_keep)
        self.spatial_method = "learned"
        self.spatial_use_norm = bool(spatial_use_norm)
        self.spatial_hidden_mult = int(spatial_hidden_mult)
        self.spatial_selection_norm = str(spatial_selection_norm).lower()
        self.spatial_temp_mul = float(spatial_temp_mul)
        self.spatial_prior = str(spatial_prior).lower()

        if self.spatial_keep:
            hidden_dim = max(1, self.in_dim * self.spatial_hidden_mult)
            if self.spatial_hidden_mult <= 0:
                self.spatial_scorer = nn.Sequential(
                    nn.LayerNorm(self.in_dim),
                    nn.Linear(self.in_dim, self.spatial_keep),
                )
            else:
                self.spatial_scorer = nn.Sequential(
                    nn.LayerNorm(self.in_dim),
                    nn.Linear(self.in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.spatial_keep),
                )
            if self.spatial_use_norm:
                self.spatial_norm = nn.LayerNorm(self.in_dim)
            self.spatial_scale = nn.Parameter(
                torch.tensor(float(spatial_scale_init))
            )
            self.spatial_prior_weight = nn.Parameter(
                torch.tensor(float(spatial_prior_weight_init))
            )

        self.post_concat_norm = bool(post_concat_norm)
        if self.post_concat_norm:
            self.post_norm = nn.LayerNorm(
                self.in_dim, eps=float(post_concat_norm_eps)
            )

        self.register_buffer("D_cache", torch.empty(0), persistent=False)
        self.register_buffer("D_small_cache", torch.empty(0), persistent=False)

    @property
    def output_tokens(self):
        return self.C * self.C + self.spatial_keep

    @staticmethod
    def _dct_matrix(size, device, dtype):
        positions = torch.arange(size, device=device, dtype=dtype)[None, :]
        frequencies = torch.arange(size, device=device, dtype=dtype)[:, None]
        basis = torch.cos(
            math.pi / size * (positions + 0.5) * frequencies
        )
        basis[0] *= 1.0 / math.sqrt(size)
        basis[1:] *= math.sqrt(2.0 / size)
        return basis

    def _get_dct(self, size, reference, *, small=False):
        name = "D_small_cache" if small else "D_cache"
        cached = getattr(self, name)
        if (
            cached.shape != (size, size)
            or cached.device != reference.device
            or cached.dtype != reference.dtype
        ):
            cached = self._dct_matrix(
                size, reference.device, reference.dtype
            )
            setattr(self, name, cached)
        return cached

    @staticmethod
    def _infer_grid(tokens):
        if tokens.ndim != 3:
            raise ValueError(
                f"expected [batch, tokens, dim], got {tokens.shape}"
            )
        length = tokens.shape[1]
        side = math.isqrt(length)
        if side * side == length:
            return tokens, side
        side = math.isqrt(length - 1)
        if length > 1 and side * side == length - 1:
            return tokens[:, 1:, :], side
        raise ValueError(
            f"input length must be N*N or 1+N*N, got {length}"
        )

    def _get_polar_features(self, side, device):
        polar_dim = (
            (3 if self.polar_include_raw else 0)
            + 4 * self.polar_num_freqs
        )
        if (
            self.polar_feat.shape == (side, side, polar_dim)
            and self.polar_feat.device == device
        ):
            return self.polar_feat

        rows = torch.arange(
            side, device=device, dtype=torch.float32
        )[:, None].expand(side, side)
        cols = torch.arange(
            side, device=device, dtype=torch.float32
        )[None, :].expand(side, side)
        radius = torch.sqrt(rows.square() + cols.square())
        radius = radius / max(math.sqrt(2.0) * (side - 1), 1.0)
        angle = torch.atan2(cols, rows)
        angle = torch.where(
            (rows == 0) & (cols == 0),
            torch.zeros_like(angle),
            angle,
        )
        bands = 2.0 ** torch.arange(
            self.polar_num_freqs, device=device, dtype=torch.float32
        )
        radial_phase = math.pi * radius[..., None] * bands
        angular_phase = angle[..., None] * bands
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
        self.polar_feat = torch.cat(features, dim=-1)
        return self.polar_feat

    @staticmethod
    def _gradient_energy(tokens, side):
        batch, _, dim = tokens.shape
        grid = tokens.view(batch, side, side, dim).float()
        vertical = grid[:, 1:] - grid[:, :-1]
        horizontal = grid[:, :, 1:] - grid[:, :, :-1]
        vertical = vertical.square().sum(dim=-1)
        horizontal = horizontal.square().sum(dim=-1)
        energy = grid.new_zeros((batch, side, side))
        energy[:, 1:] += vertical
        energy[:, :-1] += vertical
        energy[:, :, 1:] += horizontal
        energy[:, :, :-1] += horizontal
        return energy.flatten(1)

    @staticmethod
    def _normalise(logits, norm):
        if norm == "softmax":
            return F.softmax(logits, dim=-1)
        if norm == "sparsemax":
            return sparsemax(logits, dim=-1)
        if norm == "entmax15":
            return entmax15(logits, dim=-1)
        if norm.startswith("entmax:"):
            alpha = float(norm.split(":", 1)[1])
            if alpha <= 1.0:
                raise ValueError("entmax alpha must be greater than one")
            return entmax_bisect(logits, alpha=alpha, dim=-1)
        raise ValueError(f"unknown normalization: {norm}")

    def set_step(self, step, max_steps=None):
        if self.temp_start is None or self.temp_end is None:
            self.temperature_cur.fill_(max(self.temperature, 1e-6))
            return
        steps = self.temp_anneal_steps
        if steps <= 0 and max_steps is not None:
            steps = int(max_steps)
        progress = min(max(float(step) / max(steps, 1), 0.0), 1.0)
        if self.temp_schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.temp_schedule == "exp":
            if self.temp_start <= 0 or self.temp_end <= 0:
                raise ValueError(
                    "exponential temperature endpoints must be positive"
                )
            value = self.temp_start * (
                self.temp_end / self.temp_start
            ) ** progress
            self.temperature_cur.fill_(value)
            return
        elif self.temp_schedule != "linear":
            raise ValueError(
                f"unknown temperature schedule: {self.temp_schedule}"
            )
        value = self.temp_start + progress * (
            self.temp_end - self.temp_start
        )
        self.temperature_cur.fill_(value)

    def _low_frequency_tokens(self, tokens, side):
        batch, _, dim = tokens.shape
        basis = self._get_dct(side, tokens)
        grid = tokens.view(batch, side, side, dim)
        grid = grid.permute(0, 3, 1, 2).reshape(
            batch * dim, side, side
        )

        def transform():
            return basis @ grid @ basis.transpose(0, 1)

        if self.dct_detach:
            with torch.no_grad():
                transformed = transform()
        else:
            transformed = transform()
        transformed = transformed.view(batch, dim, side, side)

        if self.use_polar_pe:
            polar = self._get_polar_features(side, tokens.device)
            polar = self.polar_proj(
                polar.to(self.polar_proj.weight.dtype)
            )
            polar = polar.permute(2, 0, 1)[None].to(transformed.dtype)
            transformed = (
                transformed
                + self.polar_pe_scale.to(transformed.dtype) * polar
            )

        low = transformed[:, :, : self.C, : self.C]
        if self.low_idct:
            small = self._get_dct(self.C, low, small=True)
            flat = low.reshape(batch * dim, self.C, self.C)
            flat = small.transpose(0, 1) @ flat @ small
            low = flat.view(batch, dim, self.C, self.C)
        return low.permute(0, 2, 3, 1).reshape(
            batch, self.C * self.C, dim
        )

    def _spatial_tokens(self, tokens, side):
        if not self.spatial_keep:
            return None
        logits = self.spatial_scorer(tokens).transpose(1, 2)
        if self.spatial_prior == "grad":
            energy = self._gradient_energy(tokens, side)
            energy = (energy - energy.mean(dim=-1, keepdim=True)) / (
                energy.std(dim=-1, keepdim=True) + 1e-6
            )
            logits = (
                logits
                + self.spatial_prior_weight.to(logits.dtype)
                * energy[:, None].to(logits.dtype)
            )
        temperature = max(
            float(self.temperature_cur.item()) * self.spatial_temp_mul,
            1e-6,
        )
        weights = self._normalise(
            (logits / temperature).float(),
            self.spatial_selection_norm,
        ).to(tokens.dtype)
        residuals = torch.einsum("bsl,bld->bsd", weights, tokens)
        if self.spatial_use_norm:
            residuals = self.spatial_norm(residuals)
        return residuals * self.spatial_scale.to(residuals.dtype)

    def forward(self, tokens):
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected feature dimension {self.in_dim}, "
                f"got {tokens.shape[-1]}"
            )
        tokens, side = self._infer_grid(tokens)
        if self.C > side:
            raise ValueError(
                f"cutoff {self.C} exceeds input grid side {side}"
            )
        low = self._low_frequency_tokens(tokens, side)
        spatial = self._spatial_tokens(tokens, side)
        output = (
            low if spatial is None else torch.cat((low, spatial), dim=1)
        )
        if self.post_concat_norm:
            output = self.post_norm(output)
        if output.shape[1] != self.output_tokens:
            raise RuntimeError(
                f"expected {self.output_tokens} output tokens, "
                f"got {output.shape[1]}"
            )
        return output


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    fourier_match = re.match(r'^fourier_mlp(\d+)x_gelu$', projector_type)
    if fourier_match:
        depth = int(fourier_match.group(1))
        compressor = FourierFreqSelect(
            in_dim=config.mm_hidden_size,
            C=getattr(config, "mm_fourier_C", 16),
            K=getattr(config, "mm_fourier_K_high", 0),
            dct_detach=getattr(config, "mm_fourier_detach_dct", True),
            temperature=getattr(config, "mm_fourier_temperature", 0.8),
            temp_start=getattr(config, "mm_fourier_temp_start", 1.5),
            temp_end=getattr(config, "mm_fourier_temp_end", 0.3),
            temp_anneal_steps=getattr(
                config, "mm_fourier_temp_steps", 2000
            ),
            temp_schedule=getattr(
                config, "mm_fourier_temp_schedule", "cosine"
            ),
            use_polar_pe=getattr(
                config, "mm_fourier_use_polar_pe", False
            ),
            polar_num_freqs=getattr(
                config, "mm_fourier_polar_num_freqs", 8
            ),
            polar_include_raw=getattr(
                config, "mm_fourier_polar_include_raw", False
            ),
            polar_pe_scale_init=getattr(
                config, "mm_fourier_polar_pe_scale_init", 0.1
            ),
            spatial_keep=getattr(
                config, "mm_fourier_spatial_keep", 0
            ),
            spatial_method=getattr(
                config, "mm_fourier_spatial_method", "learned"
            ),
            spatial_use_norm=getattr(
                config, "mm_fourier_spatial_use_norm", True
            ),
            spatial_scale_init=getattr(
                config, "mm_fourier_spatial_scale_init", 1.0
            ),
            spatial_hidden_mult=getattr(
                config, "mm_fourier_spatial_hidden_mult", 1
            ),
            spatial_selection_norm=getattr(
                config,
                "mm_fourier_spatial_selection_norm",
                "sparsemax",
            ),
            spatial_temp_mul=getattr(
                config, "mm_fourier_spatial_temp_mul", 1.0
            ),
            spatial_prior=getattr(
                config, "mm_fourier_spatial_prior", "grad"
            ),
            spatial_prior_weight_init=getattr(
                config,
                "mm_fourier_spatial_prior_weight_init",
                0.0,
            ),
            post_concat_norm=getattr(
                config, "mm_fourier_post_concat_norm", False
            ),
            post_concat_norm_eps=getattr(
                config, "mm_fourier_post_concat_norm_eps", 1e-5
            ),
            low_idct=getattr(config, "mm_fourier_low_idct", False),
        )
        modules = [
            compressor,
            nn.Linear(config.mm_hidden_size, config.hidden_size),
        ]
        for _ in range(1, depth):
            modules.extend(
                (
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            )
        return nn.Sequential(*modules)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
