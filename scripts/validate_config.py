#!/usr/bin/env python3
"""Validate the Braco training and evaluation configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


class ProtocolError(ValueError):
    """Raised when the experiment matrix is internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate(config: dict[str, Any]) -> None:
    require(config["native_visual_tokens"] == 576, "native token count must be 576")
    require(config["grid"] == [24, 24], "input grid must be 24 x 24")
    budgets = config["budgets"]
    require(bool(budgets) and all(type(k) is int and 1 <= k <= 576 for k in budgets),
            "budgets must be integers from 1 to 576")
    require(len(budgets) == len(set(budgets)), "budgets must be unique")

    training = config["training"]
    require(
        training["stage1"]["global_batch_size"] == 256,
        "stage-1 global batch size must be 256",
    )
    require(
        training["stage2"]["global_batch_size"] == 128,
        "stage-2 global batch size must be 128",
    )

    methods = config["methods"]
    require(set(methods) == {"braco"}, "configuration must describe Braco")
    braco = methods["braco"]
    require(braco["boundary"] == "pre_projector", "Braco operates before the projector")
    require(braco.get("post_concat_norm") is False, "Braco normalizes residuals before concatenation")
    require(braco["budgets"] == budgets, "Braco budgets must match the budget list")
    require(set(braco["split"]) == set(budgets), "missing Braco budget configuration")
    for budget, split in braco["split"].items():
        c, s = split["C"], split["S"]
        require(type(c) is int and 1 <= c <= min(config["grid"]),
                f"Braco K={budget}: C must be an integer from 1 to 24")
        require(type(s) is int and s >= 0, f"Braco K={budget}: S must be a non-negative integer")
        retained = c ** 2 + s
        require(retained == budget, f"Braco K={budget} resolves to {retained} tokens")

    benchmarks = config["evaluation"]["benchmarks"]
    require(len(benchmarks) == len(set(benchmarks)), "benchmark keys must be unique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate(config)
    print(f"Validated protocol: {args.config}")


if __name__ == "__main__":
    main()
