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

Stage 1 uses LLaVA-558K for multimodal alignment. Stage 2 uses LLaVA-665K for instruction tuning.

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

The Braco compressor and multimodal projector are optimized in both stages;
the language model is also optimized in stage 2. The settings above use
eight training GPUs. Each token budget has its own trained checkpoint.

`GRAD_ACCUM_STEPS` controls gradient accumulation in both stages (default: `1`).
With four GPUs, use `2` to retain global batch sizes of 256 and 128:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GRAD_ACCUM_STEPS=2 \
  bash scripts/train_braco.sh 9 0
```

Spatial residual tokens pass through LayerNorm and a learned scale before
concatenation with the DCT backbone and projection into the LLM.
The sparsemax temperature follows a cosine schedule: 2.0 to 1.2 over 2180
steps in pre-training, then 1.2 to 0.8 over 4677 steps in instruction-tuning.
Inference uses temperature 0.8. The budget splits and coordinate choices
are defined in `configs/braco.yaml`.

## Evaluation workflow

Run `bash scripts/evaluate.sh BUDGET SEED` on one GPU. The manifest covers
GQA, MMBench-EN/CN, MME, POPE, ScienceQA, TextVQA, and MMVet.
Answers and available local scores are saved beneath the run's `evaluation/`
directory. The launcher creates a `llava-braco` symlink to the stage-2 checkpoint
for LLaVA's model-family detection.

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
