from pathlib import Path
import yaml
from pydantic import BaseModel

PARAMS_PATH = Path("params.yaml")

class DataConfig(BaseModel):
    raw_path: str
    processed_dir: str
    n_samples: int
    test_size: float
    random_state: int

class ModelConfig(BaseModel):
    max_features: int
    ngram_max: int
    C: float
    max_iter: int
    model_path: str

class EvaluateConfig(BaseModel):
    metrics_path: str

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    evaluate: EvaluateConfig

def load_config(path: Path = PARAMS_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)