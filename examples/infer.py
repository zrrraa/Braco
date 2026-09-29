#!/usr/bin/env python3
"""Run single-image question answering with a trained Braco LLaVA checkpoint."""

import argparse
import json
import os
from pathlib import Path
import sys


def checkpoint_budget(model_path: Path) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("mm_projector_type") != "fourier_mlp2x_gelu":
        raise ValueError("Use the full stage-2 checkpoint produced by train_braco.sh.")
    cutoff = int(config.get("mm_fourier_C", 0))
    residuals = int(config.get("mm_fourier_spatial_keep", 0))
    if cutoff < 1 or residuals < 1 or not config.get("mm_fourier_use_polar_pe"):
        raise ValueError("The checkpoint must contain the Braco backbone and spatial branch.")
    return cutoff * cutoff + residuals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Full stage-2 checkpoint directory.")
    parser.add_argument("--image", type=Path, required=True, help="Local image file.")
    parser.add_argument("--question", default="Describe this image in detail.")
    parser.add_argument("--llava-root", type=Path,
                        help="Prepared LLaVA checkout; defaults to $REPRO_ROOT/upstreams/llava.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for greedy decoding; positive values enable sampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optionally save the answer as JSON.")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0:
        parser.error("max-new-tokens must be positive and temperature must be non-negative")
    if not args.question.strip():
        parser.error("question must not be empty")
    if args.llava_root is None:
        repro_root = os.environ.get("REPRO_ROOT")
        if not repro_root:
            parser.error("provide --llava-root or set REPRO_ROOT")
        args.llava_root = Path(repro_root) / "upstreams" / "llava"
    return args


def main():
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    llava_root = args.llava_root.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    builder = llava_root / "llava/model/multimodal_projector/builder.py"
    if not builder.is_file() or "class FourierFreqSelect" not in builder.read_text(encoding="utf-8"):
        raise ValueError("Prepare the Braco LLaVA checkout with scripts/prepare_upstreams.sh first.")
    budget = checkpoint_budget(model_path)
    weight_files = ("model.safetensors", "model.safetensors.index.json",
                    "pytorch_model.bin", "pytorch_model.bin.index.json")
    if not any((model_path / name).is_file() for name in weight_files):
        raise ValueError("Use a full stage-2 model checkpoint containing the language-model weights.")
    sys.path.insert(0, str(llava_root))

    import torch
    from PIL import Image
    from transformers import set_seed
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    if not torch.cuda.is_available():
        raise RuntimeError("This example runs the LLaVA-v1.5-7B checkpoint on a CUDA GPU.")
    set_seed(args.seed)
    # Explicit model name selects LLaVA even when the directory is named stage2.
    tokenizer, model, processor, context_len = load_pretrained_model(
        str(model_path), None, "llava-braco-v1.5-7b",
        device_map="cuda:0", device="cuda",
    )
    model.eval()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], image_token + "\n" + args.question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    if input_ids.shape[1] - 1 + budget + args.max_new_tokens > context_len:
        raise ValueError("Question and answer exceed the model context; shorten the question or max-new-tokens.")
    pixels = process_images([image], processor, model.config)
    if isinstance(pixels, list):
        pixels = [p.to(model.device, dtype=torch.float16) for p in pixels]
    else:
        pixels = pixels.to(model.device, dtype=torch.float16)
    generation = {"do_sample": args.temperature > 0,
                  "max_new_tokens": args.max_new_tokens, "use_cache": True}
    if args.temperature > 0:
        generation["temperature"] = args.temperature
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=pixels,
                                    image_sizes=[image.size], **generation)
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(answer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "model_path": str(model_path), "visual_tokens": budget,
            "image": str(image_path), "question": args.question, "answer": answer,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
