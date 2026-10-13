# Training and evaluation guide

The launch scripts read model and dataset paths from `MODEL_ROOT` and `DATA_ROOT`.

## Model files

```text
MODEL_ROOT/
├── vicuna-7b-v1.5/
└── clip-vit-large-patch14-336/
```

Use the public model releases referenced by LLaVA-v1.5. Model licenses and access terms apply.

## Training data

```text
DATA_ROOT/
├── LLaVA-Pretrain/
│   ├── blip_laion_cc_sbu_558k.json
│   └── images/
└── LLaVA-Instruct/
    ├── llava_v1_5_mix665k.json
    └── images/
```

The default setup uses LLaVA-558K for multimodal alignment in stage 1 and LLaVA-665K for instruction tuning in stage 2.

## Evaluation data

```text
DATA_ROOT/eval/
├── gqa/
├── mmbench/
├── mme/
├── pope/
├── scienceqa/
├── textvqa/
└── mmvet/
```

The paths expected inside each benchmark directory are listed directly in `scripts/eval_one.sh`. Benchmark annotations and images follow their official distribution terms.

## Training settings

The scripts provide the following defaults for an eight-GPU run:

| Setting | Stage 1: alignment | Stage 2: instruction tuning |
|---|---:|---:|
| Dataset | LLaVA-558K | LLaVA-665K |
| Epochs | 1 | 1 |
| Global batch size | 256 | 128 |
| Learning rate | 1e-3 | 2e-5 |
| Scheduler | cosine | cosine |
| Warm-up | 3% | 3% |
| Frozen modules | vision encoder, LLM | vision encoder |
| Checkpoint interval | 500 steps | 500 steps |
| Retained checkpoints | 2 | 2 |

In this setup, the Braco compressor and multimodal projector are optimized
in both stages, and the language model is also optimized in stage 2.
Each configuration has its own trained checkpoint.

The default splits `(C, S) = (1, 3), (2, 5), (3, 7), (4, 9)` give 4, 9, 16,
and 25 tokens. `C` is the side length of the retained DCT block, and `S` is
the number of spatial residual tokens, giving `K = C² + S` tokens.
For example, the 9-token configuration is:

```bash
bash scripts/train_braco.sh --c 2 --s 5 --seed 0
```

This is equivalent to `bash scripts/train_braco.sh 9 0`. The `--seed` option
defaults to `0`.

The default basis is DCT; the compressor can be adapted to other bases.
Use `--pe` to enable or disable basis-coordinate embeddings and `--idct`
to choose the coordinate organization:

```bash
bash scripts/train_braco.sh 25 0 --pe true --idct true
```

All token budgets default to `--pe true --idct false`.
Both options accept `true` or `false` for every budget.
Run directories include C/S and PE/IDCT suffixes when they differ from the
defaults. The IDCT example above saves to
`$REPRO_ROOT/outputs/braco/k25/seed0-pe-true-idct-true/`.
The resolved settings are saved in `metadata/resolved_protocol.yaml` and the checkpoint.

`GRAD_ACCUM_STEPS` controls gradient accumulation in both stages (default: `1`).
With four GPUs, use `2` to retain global batch sizes of 256 and 128:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GRAD_ACCUM_STEPS=2 \
  bash scripts/train_braco.sh 9 0
```

By default, spatial residual tokens pass through LayerNorm and a learned scale before
concatenation with the DCT backbone and projection into the LLM.
For sparsemax, one schedule to try is cosine annealing from 2.0 to 1.2 over
2180 pre-training steps, then 1.2 to 0.8 over 4677 instruction-tuning steps,
with an inference temperature of 0.8. The training defaults can be adjusted
in `scripts/train_braco.sh`; `configs/braco.yaml` lists the default budget
splits and coordinate settings.

## Evaluation workflow

Run `bash scripts/evaluate.sh BUDGET SEED` on one GPU. The manifest covers
GQA, MMBench-EN/CN, MME, POPE, ScienceQA, TextVQA, and MMVet.
Answers and available local scores are saved beneath the run's `evaluation/`
directory. The launcher creates a `llava-braco` symlink to the stage-2 checkpoint
for LLaVA's model-family detection.

Pass the same options used for training:

```bash
bash scripts/evaluate.sh --c 2 --s 5 --seed 0
bash scripts/evaluate.sh 25 0 --pe true --idct true
```

Inference reads C, S, PE, and IDCT from the checkpoint configuration.
For standalone use, construct `BracoCompressor(dim=1024, cutoff=2, residual_tokens=5)`
for the 9-token configuration.

Use the following metrics when reading the evaluator output:

- **ScienceQA:** image-subset accuracy (`IMG-Accuracy`).
- **MME All:** the sum of the Perception and Cognition total scores.
- **POPE:** F1 expressed as a percentage; the evaluator reports the random,
  popular, and adversarial subsets separately.

For MMBench, the launcher creates the official upload artifacts. For MMVet,
it creates the answer JSON for the official judge-based evaluator. Complete
these benchmarks with their respective official evaluation services or tools.

## LLaVA integration

The preparation script checks out [LLaVA](https://github.com/haotian-liu/LLaVA)
at commit `c121f0432da27facab705978f83c4ada465e46fd` and applies the three files
under `integrations/overlays/llava/`: the compressor/projector, multimodal
configuration, and training entrypoint. Use this prepared checkout for training
and inference. LLaVA is licensed under Apache-2.0; model weights and datasets
follow their respective providers' terms.
