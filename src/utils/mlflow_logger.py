import mlflow
from omegaconf import OmegaConf

class MLflowLogger:
    def __init__(self, cfg):
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        self.run = mlflow.start_run(run_name=cfg.mlflow.run_name)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    def log_metrics(self, metrics: dict, epoch: int, prefix="train"):
        mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=epoch)

    def save_model(self, model, path):
        mlflow.pytorch.log_model(model, path)

    def end(self):
        mlflow.end_run()
