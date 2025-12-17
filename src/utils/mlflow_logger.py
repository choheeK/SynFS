# src/utils/mlflow_logger.py

import mlflow
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig


class MLflowLogger:
    """
    MLflow logger that is Hydra-safe and reusable.

    Responsibilities:
    - set tracking URI
    - set experiment
    - start / end run
    - log Hydra config
    - log params & metrics
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(cfg.get("mlflow", {}).get("enabled", False))
        self.run = None

        if not self.enabled:
            return

        # --------------------------------------------------
        # 1. Set tracking URI (Hydra-safe)
        # --------------------------------------------------
        project_root = get_original_cwd()
        mlruns_dir = Path(project_root) / "mlruns"
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")

        # --------------------------------------------------
        # 2. Set experiment
        # --------------------------------------------------
        exp_name = cfg.mlflow.get("experiment_name", "default")
        mlflow.set_experiment(exp_name)

    # --------------------------------------------------
    # Run lifecycle
    # --------------------------------------------------
    def start_run(self):
        if not self.enabled:
            return None

        self.run = mlflow.start_run()

        # Log config once at run start
        self._log_hydra_config()
        self._log_flat_config()

        return self.run

    def end_run(self):
        if self.enabled and self.run is not None:
            mlflow.end_run()
            self.run = None

    # --------------------------------------------------
    # Logging helpers
    # --------------------------------------------------
    def log_metrics(self, metrics: dict, step: int):
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int):
        if self.enabled:
            mlflow.log_metric(key, value, step=step)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _log_flat_config(self):
        """
        Log flattened config as MLflow parameters
        """
        flat_cfg = OmegaConf.to_container(self.cfg, resolve=True)

        def _log_dict(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _log_dict(v, key)
                else:
                    try:
                        mlflow.log_param(key, v)
                    except Exception:
                        # MLflow params must be simple types
                        mlflow.log_param(key, str(v))

        _log_dict(flat_cfg)

    def _log_hydra_config(self):
        """
        Log full resolved Hydra config as an artifact
        """
        resolved_yaml = OmegaConf.to_yaml(self.cfg, resolve=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config_resolved.yaml"
            path.write_text(resolved_yaml)
            mlflow.log_artifact(str(path), artifact_path="config")

        # Hydra metadata (nice to have)
        try:
            hydra_cfg = HydraConfig.get()
            mlflow.set_tag("hydra.run_dir", hydra_cfg.runtime.output_dir)
            mlflow.set_tag("hydra.job_name", hydra_cfg.job.name)
        except Exception:
            pass
