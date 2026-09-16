from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def _resolve_project_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


class DataConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    raw_path: str
    processed_dir: str
    n_samples: int
    test_size: float
    random_state: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    max_features: int
    ngram_max: int
    C: float
    max_iter: int
    model_path: str


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    metrics_path: str


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    data: DataConfig
    model: ModelConfig
    evaluate: EvaluateConfig


def load_config(path: str | Path = PARAMS_PATH) -> Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(**raw)
    cfg.data.raw_path = _resolve_project_path(cfg.data.raw_path)
    cfg.data.processed_dir = _resolve_project_path(cfg.data.processed_dir)
    cfg.model.model_path = _resolve_project_path(cfg.model.model_path)
    cfg.evaluate.metrics_path = _resolve_project_path(cfg.evaluate.metrics_path)
    return cfg
