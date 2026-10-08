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
    # "synthetic" uses the seeded template generator; "banking77" uses a real
    # support corpus. Defaulted so existing params.yaml files still load, and
    # declared as a DVC param so switching corpus reruns the pipeline.
    source: str = "synthetic"
    cache_dir: str = "data/external"
    # "official" reuses banking77's published train/test files, so results are
    # comparable with the literature; "random" is a stratified re-split by
    # test_size. Synthetic data has no official split and needs "random".
    split: str = "random"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    max_features: int
    ngram_max: int
    C: float
    max_iter: int
    model_path: str
    # "logreg" or "linearsvc_calibrated"; see models/train.py. Defaulted so
    # older params.yaml files still load as the original LogReg model.
    type: str = "logreg"
    # CalibratedClassifierCV method for linearsvc_calibrated: sigmoid or isotonic.
    calibration: str = "sigmoid"


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    metrics_path: str
    # Written by the baselines stage. Defaulted so older params.yaml files load.
    baselines_path: str = "metrics/baselines.json"


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # The prediction log the drift monitor reads.
    predictions_path: str = "data/predictions.jsonl"
    # Mask emails and long digit runs (card, account numbers) before writing.
    redact: bool = True
    # Rotate once the live file would pass this size; 0 disables rotation.
    max_bytes: int = 5_000_000
    # Rotated files kept beside the live one. Older ones are deleted.
    backup_count: int = 3


class ServeConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Predictions whose top probability is below this are answered with
    # label "needs_review" instead of a queue. 0 disables the route. Choose it
    # from reports/coverage_accuracy.csv, written by the evaluate stage.
    review_threshold: float = 0.0


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    data: DataConfig
    model: ModelConfig
    evaluate: EvaluateConfig
    monitoring: MonitoringConfig = MonitoringConfig()
    serve: ServeConfig = ServeConfig()


def load_config(path: str | Path = PARAMS_PATH) -> Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(**raw)
    cfg.data.raw_path = _resolve_project_path(cfg.data.raw_path)
    cfg.data.processed_dir = _resolve_project_path(cfg.data.processed_dir)
    cfg.data.cache_dir = _resolve_project_path(cfg.data.cache_dir)
    cfg.model.model_path = _resolve_project_path(cfg.model.model_path)
    cfg.evaluate.metrics_path = _resolve_project_path(cfg.evaluate.metrics_path)
    cfg.evaluate.baselines_path = _resolve_project_path(cfg.evaluate.baselines_path)
    cfg.monitoring.predictions_path = _resolve_project_path(cfg.monitoring.predictions_path)
    return cfg
