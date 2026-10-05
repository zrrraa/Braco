import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from braco import BRACO_BUDGETS, BracoCompressor


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT
    / "integrations"
    / "overlays"
    / "llava"
    / "llava"
    / "model"
    / "multimodal_projector"
    / "builder.py"
)


def load_integrated_module():
    spec = importlib.util.spec_from_file_location("llava_braco_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", sorted(BRACO_BUDGETS))
def test_reference_matches_llava_integration(budget: int, dtype: torch.dtype) -> None:
    integrated_module = load_integrated_module()
    config = BRACO_BUDGETS[budget]

    torch.manual_seed(17 + budget)
    model_config = SimpleNamespace(
        mm_projector_type="fourier_mlp2x_gelu",
        mm_hidden_size=16,
        hidden_size=32,
        mm_fourier_C=config.cutoff,
        mm_fourier_K_high=0,
        mm_fourier_spatial_keep=config.residual_tokens,
        mm_fourier_use_polar_pe=True,
        mm_fourier_low_idct=config.coordinate_mode == "idct",
        mm_fourier_post_concat_norm=False,
    )
    integrated = integrated_module.build_vision_projector(
        model_config
    )[0].eval()
    reference = BracoCompressor.from_budget(
        dim=16,
        budget=budget,
        detach_transform=False,
        temperature=float(integrated.temperature_cur.item()),
    ).eval()

    assert integrated.polar_include_raw is False
    assert integrated.post_concat_norm is False
    assert reference.post_concat_norm is False
    assert integrated.polar_proj.in_features == 4 * integrated.polar_num_freqs
    assert reference.polar_include_raw is False
    assert (
        reference.polar_projection.in_features
        == integrated.polar_proj.in_features
    )

    reference.polar_projection.load_state_dict(integrated.polar_proj.state_dict())
    reference.polar_scale.data.copy_(integrated.polar_pe_scale.data)
    reference.residual_scorer.load_state_dict(
        integrated.spatial_scorer.state_dict()
    )
    reference.residual_norm.load_state_dict(integrated.spatial_norm.state_dict())
    reference.residual_scale.data.copy_(integrated.spatial_scale.data)
    reference.residual_prior_scale.data.copy_(
        integrated.spatial_prior_weight.data
    )

    integrated = integrated.to(dtype=dtype)
    reference = reference.to(dtype=dtype)
    integrated.spatial_prior_weight.data.fill_(0.2)
    reference.residual_prior_scale.data.fill_(0.2)
    tokens = torch.randn(2, 576, 16).to(dtype)
    with torch.no_grad():
        expected = integrated(tokens)
        actual = reference(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
