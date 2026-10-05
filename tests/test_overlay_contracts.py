import ast
from pathlib import Path

import yaml

from scripts.validate_config import validate


ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ROOT / "integrations" / "overlays"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_overlay_python_file_compiles() -> None:
    for path in OVERLAYS.rglob("*.py"):
        compile(_text(path), str(path), "exec")


def test_llava_train_imports_only_released_compressor() -> None:
    builder_path = (
        OVERLAYS
        / "llava"
        / "llava"
        / "model"
        / "multimodal_projector"
        / "builder.py"
    )
    train_path = OVERLAYS / "llava" / "llava" / "train" / "train.py"
    builder_tree = ast.parse(_text(builder_path))
    train_tree = ast.parse(_text(train_path))

    defined = {
        node.name
        for node in builder_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    imported = set()
    for node in ast.walk(train_tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            == "llava.model.multimodal_projector.builder"
        ):
            imported.update(alias.name for alias in node.names)

    assert imported == {"FourierFreqSelect"}
    assert imported <= defined


def test_llava_overlay_excludes_research_only_artifacts() -> None:
    contents = "\n".join(
        _text(path) for path in (OVERLAYS / "llava").rglob("*.py")
    )
    for forbidden in (
        "FourierBudgetRotate",
        "FourierAutoSearch",
        "FourierQuestionRouter",
        "matplotlib",
        "contentReference",
    ):
        assert forbidden not in contents


def test_conversion_launchers_have_required_setup() -> None:
    evaluate = _text(ROOT / "scripts" / "evaluate.sh")
    one = _text(ROOT / "scripts" / "eval_one.sh")

    assert 'MODEL_PATH="${RUN_ROOT}/llava-braco"' in evaluate
    assert 'mkdir -p "${RESULT_ROOT}/${BENCHMARK}/upload"' in one
    assert (
        '--dst "${EVAL}/gqa/data/testdev_balanced_predictions.json"'
        in one
    )
    assert '--experiment "${MME_EXPERIMENT}"' in one
    assert '--results_dir "answers/${MME_EXPERIMENT}"' in one


def test_environment_matches_pinned_llava_runtime() -> None:
    environment = _text(ROOT / "environment.yml")

    assert "transformers==4.37.2" in environment
    assert "tokenizers==0.15.1" in environment


def test_braco_protocol_is_valid() -> None:
    config = yaml.safe_load((ROOT / "configs/braco.yaml").read_text(encoding="utf-8"))
    validate(config)


def test_temperature_initialized_before_first_and_resumed_steps() -> None:
    from types import SimpleNamespace

    tree = ast.parse(_text(OVERLAYS / "llava/llava/train/train.py"))
    callback_node = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "FourierAnnealCallback")

    class Compressor:
        def set_step(self, step, max_steps):
            self.last_step = (step, max_steps)

    namespace = {"transformers": SimpleNamespace(TrainerCallback=object),
                 "FourierFreqSelect": Compressor}
    exec(compile(ast.Module(body=[callback_node], type_ignores=[]),
                 "callback", "exec"), namespace)
    callback = namespace["FourierAnnealCallback"]()
    compressor = Compressor()
    model = SimpleNamespace(modules=lambda: [compressor])
    for step in (0, 500):
        state = SimpleNamespace(global_step=step, max_steps=2181)
        callback.on_train_begin(None, state, None, model=model)
        assert compressor.last_step == (step, 2181)
        state.global_step += 1
        callback.on_step_end(None, state, None, model=model)
        assert compressor.last_step == (step + 1, 2181)
