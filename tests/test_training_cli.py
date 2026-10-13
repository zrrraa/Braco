"""Exercise launcher options without loading models or starting GPU jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Training launchers target Linux")


@pytest.fixture
def launchers(tmp_path):
    package = tmp_path / "package"
    for name in ["scripts/common.sh", "scripts/train_braco.sh", "scripts/evaluate.sh",
                 "configs/braco.yaml", "configs/benchmarks.txt"]:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    work, models, data, bin_dir = [tmp_path / n for n in ["work", "models", "data", "bin"]]
    for folder in [work / "upstreams/llava", models / "vicuna-7b-v1.5",
                   models / "clip-vit-large-patch14-336", bin_dir]:
        folder.mkdir(parents=True)
    for name in ["LLaVA-Pretrain/blip_laion_cc_sbu_558k.json", "LLaVA-Instruct/llava_v1_5_mix665k.json"]:
        file = data / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("[]")
    (bin_dir / "python").symlink_to(sys.executable)
    for command in ["nvidia-smi", "git"]:
        file = bin_dir / command
        file.write_text("#!/bin/sh\nprintf 'test metadata\\n'\n")
        file.chmod(0o755)
    stub = bin_dir / "deepspeed"
    stub.write_text("""#!/usr/bin/env python
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with (Path(os.environ['REPRO_ROOT'])/'launches.jsonl').open('a') as f:
    f.write(json.dumps(args)+'\\n')
out = Path(args[args.index('--output_dir')+1])
out.mkdir(parents=True, exist_ok=True)
if out.name == 'stage1': (out/'mm_projector.bin').touch()
""")
    stub.chmod(0o755)
    (package / "scripts/eval_one.sh").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$MODEL_PATH" >> "$RESULT_ROOT/models.txt"\n'
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "REPRO_ROOT": str(work), "MODEL_ROOT": str(models), "DATA_ROOT": str(data)}
    return package, work, env


def run(launchers, script, *args):
    package, _, env = launchers
    return subprocess.run(["bash", str(package / "scripts" / script), *map(str, args)],
                          env=env, text=True, capture_output=True)


@pytest.mark.parametrize("budget", [4, 9, 16, 25])
def test_default_paths_and_budget_settings(launchers, budget):
    package, work, env = launchers
    command = 'source "$1"; shift; parse_braco_run "$@"; printf "%s %s %s %s %s" "$C" "$S" "$USE_PE" "$LOW_IDCT" "$RUN_ROOT"'
    result = subprocess.run(["bash", "-c", command, "bash", str(package / "scripts/common.sh"), str(budget), "0"],
                            env=env, text=True, capture_output=True, check=True)
    c, s, pe, idct, output = result.stdout.split()
    assert int(c)**2 + int(s) == budget
    assert pe == "True" and idct == "False"
    assert output == str(work / f"outputs/braco/k{budget}/seed0")


@pytest.mark.parametrize("budget", [4, 9, 16, 25])
@pytest.mark.parametrize("options,pe,idct", [
    ([], True, False),
    (["--pe", "false", "--idct", "false"], False, False),
    (["--pe=True", "--idct=True"], True, True),
    (["--pe", "false", "--idct", "true"], False, True),
])
def test_two_stages_metadata_and_evaluation_select_same_run(launchers, budget, options, pe, idct):
    _, work, _ = launchers
    result = run(launchers, "train_braco.sh", budget, 7, *options)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in (work / "launches.jsonl").read_text().splitlines()]
    assert len(commands) == 2
    for args in commands:
        assert args[args.index("--mm_fourier_use_polar_pe")+1] == str(pe)
        assert args[args.index("--mm_fourier_low_idct")+1] == str(idct)
    out = Path(commands[1][commands[1].index("--output_dir")+1]).parent
    suffix = f"-pe-{str(pe).lower()}-idct-{str(idct).lower()}" if not pe or idct else ""
    assert out.name == f"seed7{suffix}"
    assert commands[1][commands[1].index("--pretrain_mm_mlp_adapter")+1] == str(out / "stage1/mm_projector.bin")
    config = yaml.safe_load((out / "metadata/resolved_protocol.yaml").read_text())
    assert config["seed"] == 7 and config["selected_budget"] == budget
    assert config["methods"]["braco"]["use_polar_pe"] is pe
    assert config["methods"]["braco"]["split"][budget]["low_idct"] is idct
    result = run(launchers, "evaluate.sh", budget, 7, *options)
    assert result.returncode == 0, result.stderr
    evaluated = (out / "evaluation/models.txt").read_text().splitlines()
    assert len(evaluated) == 8 and set(evaluated) == {str(out / "llava-braco")}
    assert (out / "llava-braco").resolve() == out / "stage2"


@pytest.mark.parametrize("args", [[], [9], [9, 0, "--pe"], [9, 0, "--idct=maybe"],
                                  [9, 0, "--pe", "0"], [9, 0, "--unknown"], [9, -1], [17, 0]])
def test_invalid_options_do_not_launch_training(launchers, args):
    _, work, _ = launchers
    result = run(launchers, "train_braco.sh", *args)
    assert result.returncode == 2 and not (work / "launches.jsonl").exists()


@pytest.mark.parametrize("script", ["train_braco.sh", "evaluate.sh"])
def test_help_without_model_or_data_paths(script):
    env = {k:v for k,v in os.environ.items() if k not in ["REPRO_ROOT", "MODEL_ROOT", "DATA_ROOT"]}
    result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--help"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0 and "--pe" in result.stdout and "--idct" in result.stdout
