import pytest
import torch

from braco import BRACO_BUDGETS, BracoCompressor


def test_main_table_polar_features_exclude_raw_coordinates() -> None:
    compressor = BracoCompressor.from_budget(dim=16, budget=9)
    assert compressor.polar_include_raw is False
    assert compressor.polar_projection.in_features == 32
    assert compressor.post_concat_norm is False


def test_raw_polar_coordinates_remain_optional() -> None:
    compressor = BracoCompressor.from_budget(
        dim=16,
        budget=9,
        polar_include_raw=True,
    )
    assert compressor.polar_projection.in_features == 35


def test_main_table_normalizes_only_residual_tokens() -> None:
    compressor = BracoCompressor.from_budget(
        dim=16,
        budget=9,
        detach_transform=False,
    ).eval()
    tokens = torch.randn(2, 576, 16)

    with torch.no_grad():
        output = compressor(tokens)

    assert not hasattr(compressor, "post_norm")
    assert output[:, :4].mean(dim=-1).abs().max() > 1e-4
    torch.testing.assert_close(
        output[:, 4:].mean(dim=-1),
        torch.zeros_like(output[:, 4:, 0]),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("budget", sorted(BRACO_BUDGETS))
def test_main_table_budget_shapes_and_backward(budget: int) -> None:
    compressor = BracoCompressor.from_budget(
        dim=32,
        budget=budget,
        detach_transform=False,
    )
    tokens = torch.randn(2, 576, 32, requires_grad=True)
    output = compressor(tokens)

    assert output.shape == (2, budget, 32)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_eval_forward_is_deterministic() -> None:
    torch.manual_seed(0)
    compressor = BracoCompressor.from_budget(
        dim=16,
        budget=9,
        detach_transform=False,
    ).eval()
    tokens = torch.randn(1, 576, 16)

    with torch.no_grad():
        first = compressor(tokens)
        second = compressor(tokens)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_optional_cls_token_is_removed() -> None:
    compressor = BracoCompressor.from_budget(dim=8, budget=4).eval()
    tokens = torch.randn(1, 577, 8)

    with torch.no_grad():
        output = compressor(tokens)

    assert output.shape == (1, 4, 8)


def test_non_square_token_count_is_rejected() -> None:
    compressor = BracoCompressor.from_budget(dim=8, budget=4)
    with pytest.raises(ValueError, match=r"N\^2"):
        compressor(torch.randn(1, 102, 8))
