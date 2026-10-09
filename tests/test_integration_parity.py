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
        detach_transform=integrated.dct_detach,
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
    tokens = torch.randn(2, 576, 16).to(dtype).requires_grad_()
    expected = integrated(tokens)
    actual = reference(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad_output = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, tokens, grad_output)[0]
    actual_grad = torch.autograd.grad(actual, tokens, grad_output)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_optimizer_parameter_order_matches_training_checkpoints() -> None:
    compressor = load_integrated_module().FourierFreqSelect(
        in_dim=16, C=2, spatial_keep=5, polar_include_raw=False,
    )
    assert list(dict(compressor.named_parameters())) == [
        "polar_pe_scale", "spatial_scale", "spatial_prior_weight",
        "polar_proj.weight", "spatial_norm.weight", "spatial_norm.bias",
        "spatial_scorer.0.weight", "spatial_scorer.0.bias",
        "spatial_scorer.1.weight", "spatial_scorer.1.bias",
        "spatial_scorer.3.weight", "spatial_scorer.3.bias",
    ]


@pytest.mark.parametrize("hidden_multiplier", [0, -1])
def test_single_layer_residual_scorer_matches_integration(hidden_multiplier) -> None:
    torch.manual_seed(29)
    integrated = load_integrated_module().FourierFreqSelect(
        in_dim=16, C=2, spatial_keep=5, polar_include_raw=False,
        spatial_hidden_mult=hidden_multiplier, temperature=0.8,
    )
    torch.manual_seed(29)
    standalone = BracoCompressor.from_budget(
        dim=16, budget=9, residual_hidden_multiplier=hidden_multiplier,
    )
    tokens = torch.randn(2, 576, 16)
    torch.testing.assert_close(standalone(tokens), integrated(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("previous_public_names", [False, True])
def test_optional_concat_norm_checkpoint_loading(previous_public_names) -> None:
    cls = load_integrated_module().FourierFreqSelect
    original = cls(in_dim=16, C=2, spatial_keep=5, post_concat_norm=True)
    restored = cls(in_dim=16, C=2, spatial_keep=5, post_concat_norm=True)
    weights = original.state_dict()
    assert "post_concat_ln.weight" in weights
    if previous_public_names:
        weights = {k.replace("post_concat_ln.", "post_norm."): v
                   for k, v in weights.items()}
    restored.load_state_dict(weights, strict=True)
    tokens = torch.randn(2, 576, 16)
    torch.testing.assert_close(restored(tokens), original(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("anneal_steps", [0, -1])
def test_disabled_annealing_keeps_fixed_temperature(anneal_steps) -> None:
    compressor = load_integrated_module().FourierFreqSelect(
        in_dim=8, C=2, temperature=0.8, temp_start=2.0,
        temp_end=1.2, temp_anneal_steps=anneal_steps,
    )
    compressor.set_step(50, max_steps=100)
    assert compressor.temperature_cur.item() == pytest.approx(0.8)


@pytest.mark.parametrize(
    "schedule, expected", [
        ("cosine", [2.0, 1.6, 1.2]),
        ("linear", [2.0, 1.6, 1.2]),
        ("exp", [2.0, 1.28, 1.208]),
    ],
)
def test_temperature_schedule_reference_values(schedule, expected) -> None:
    compressor = load_integrated_module().FourierFreqSelect(
        in_dim=8, C=2, temp_start=2.0, temp_end=1.2,
        temp_anneal_steps=100, temp_schedule=schedule,
    )
    for step, value in zip([0, 50, 100], expected):
        compressor.set_step(step, max_steps=1000)
        assert compressor.temperature_cur.item() == pytest.approx(value)


@pytest.mark.parametrize("idct", [False, True])
def test_backbone_only_output_is_contiguous(idct) -> None:
    compressor = load_integrated_module().FourierFreqSelect(
        in_dim=16, C=4, spatial_keep=0, low_idct=idct,
    )
    tokens = torch.randn(2, 577, 16)
    output = compressor(tokens)
    assert output.shape == (2, 16, 16)
    assert output.is_contiguous()
    assert "spatial_norm.weight" in compressor.state_dict()
