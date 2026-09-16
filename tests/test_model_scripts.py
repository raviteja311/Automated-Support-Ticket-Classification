import importlib.util
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_scripts_import_and_define_main():
    train_module = _load_module("train_script", "models/train.py")
    evaluate_module = _load_module("evaluate_script", "models/evaluate.py")

    assert callable(train_module.main)
    assert callable(evaluate_module.main)
