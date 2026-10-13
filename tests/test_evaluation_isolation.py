"""Concurrent evaluation runs must leave the shared dataset unchanged."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="Evaluation launchers target Linux")


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


WAIT_FOR_BOTH_RUNS = """
import os, time
sync = Path(os.environ['SYNC_ROOT'])
(sync / os.environ['RUN_LABEL']).touch()
deadline = time.monotonic() + 10
while len(list(sync.iterdir())) < 2:
    if time.monotonic() > deadline:
        raise RuntimeError('Concurrent evaluator did not arrive')
    time.sleep(0.01)
"""


@pytest.mark.parametrize("benchmark", ["gqa", "mme"])
def test_parallel_runs_keep_predictions_and_scores_separate(tmp_path, benchmark):
    data, code, bin_dir, sync = [tmp_path / name for name in ("data", "code", "bin", "sync")]
    code.mkdir()
    sync.mkdir()
    # Replace only model generation. The converter/scorer fixtures retain the
    # official tools' path conventions, including their shared-directory defaults.
    wrapper = bin_dir / "python"
    write(wrapper, f"#!{sys.executable}\n" + textwrap.dedent("""
        import json, os, sys
        from pathlib import Path
        args = sys.argv[1:]
        if args[:2] == ['-m', 'llava.eval.model_vqa_loader']:
            output = Path(args[args.index('--answers-file') + 1])
            output.write_text(json.dumps({'question_id': 'q1', 'text': os.environ['RUN_LABEL']}) + '\\n')
        else:
            os.execv(sys.executable, [sys.executable, *args])
    """))
    wrapper.chmod(0o755)

    write(code / "scripts/convert_gqa_for_eval.py", """
        import argparse, json
        from pathlib import Path
        p = argparse.ArgumentParser()
        p.add_argument('--src'); p.add_argument('--dst')
        a = p.parse_args()
        rows = [json.loads(line) for line in Path(a.src).read_text().splitlines()]
        Path(a.dst).write_text(json.dumps({r['question_id']: r['text'] for r in rows}))
    """)
    write(data / "eval/gqa/data/eval/eval.py", textwrap.dedent("""
        import argparse, json
        from pathlib import Path
        p = argparse.ArgumentParser()
        p.add_argument('--tier')
        p.add_argument('--predictions', default='{tier}_predictions.json')
        a = p.parse_args()
    """) + WAIT_FOR_BOTH_RUNS + "\nprint(json.loads(Path(a.predictions.format(tier=a.tier)).read_text())['q1'])\n")
    write(data / "eval/gqa/data/testdev_balanced_predictions.json", '{"q1":"existing"}')

    write(data / "eval/mme/MME_Benchmark_release_version/gt.txt", "reference")
    write(data / "eval/mme/answers/braco_k9_seed0.jsonl", "existing predictions")
    write(data / "eval/mme/eval_tool/answers/braco_k9_seed0/result.txt", "existing score")
    write(data / "eval/mme/convert_answer_to_mme.py", """
        import argparse, json
        from pathlib import Path
        p = argparse.ArgumentParser(); p.add_argument('--experiment')
        a = p.parse_args()
        assert Path('MME_Benchmark_release_version/gt.txt').read_text() == 'reference'
        rows = [json.loads(line) for line in Path('answers', a.experiment + '.jsonl').read_text().splitlines()]
        out = Path('eval_tool/answers') / a.experiment
        out.mkdir(parents=True, exist_ok=True)
        (out / 'result.txt').write_text(rows[0]['text'])
    """)
    write(data / "eval/mme/eval_tool/calculation.py", textwrap.dedent("""
        import argparse
        from pathlib import Path
        p = argparse.ArgumentParser(); p.add_argument('--results_dir')
        a = p.parse_args()
    """) + WAIT_FOR_BOTH_RUNS + "\nprint((Path(a.results_dir) / 'result.txt').read_text())\n")
    before = snapshot(data)

    processes = []
    try:
        for label in ("idct-on", "idct-off"):
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                   "MODEL_PATH": label, "RESULT_ROOT": str(tmp_path / label),
                   "DATA_ROOT": str(data), "METHOD": "braco", "BUDGET": "9", "SEED": "0",
                   "RUN_LABEL": label, "SYNC_ROOT": str(sync)}
            processes.append(subprocess.Popen(
                ["bash", str(ROOT / "scripts/eval_one.sh"), benchmark],
                cwd=code, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    for label in ("idct-on", "idct-off"):
        output = tmp_path / label / benchmark
        assert (output / "score.txt").read_text().strip() == label
        if benchmark == "mme":
            assert (output / "official_answers/result.txt").read_text() == label
            assert not list(output.glob(".scoring.*"))
    assert snapshot(data) == before
