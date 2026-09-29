"""Validate checkpoint selection and CLI usage without loading model weights."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "infer.py"
spec = importlib.util.spec_from_file_location("braco_inference_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class InferenceExampleTests(unittest.TestCase):
    def test_budget_is_read_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for cutoff, residuals, expected in [(1, 3, 4), (2, 5, 9), (3, 7, 16), (4, 9, 25)]:
                (path / "config.json").write_text(json.dumps({
                    "mm_projector_type": "fourier_mlp2x_gelu",
                    "mm_fourier_C": cutoff, "mm_fourier_spatial_keep": residuals,
                    "mm_fourier_use_polar_pe": True,
                }), encoding="utf-8")
                self.assertEqual(example.checkpoint_budget(path), expected)

    def test_vanilla_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                '{"mm_projector_type": "mlp2x_gelu"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage-2"):
                example.checkpoint_budget(path)

    def test_help_runs_without_model_dependencies(self):
        result = subprocess.run([sys.executable, str(EXAMPLE), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-path", result.stdout)
        self.assertIn("--llava-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
