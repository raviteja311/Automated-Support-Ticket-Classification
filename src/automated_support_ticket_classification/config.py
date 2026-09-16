from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

PARAMS_PATH = Path("params.yaml")


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


def load_config(path: Path = PARAMS_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
