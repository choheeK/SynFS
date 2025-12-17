import hydra
from omegaconf import DictConfig

from src.data.datamodule import build_dataloaders
from src.trainer.trainer import SynFSTrainer
from src.utils.mlflow_logger import MLflowLogger
from src.utils.seed import fix_seed
from src.models.synfs_model import SynFSModel
from hydra.utils import get_original_cwd




@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    fix_seed(cfg.seed)

    logger = MLflowLogger(cfg)

    with logger.start_run():
        train_loader, val_loader = build_dataloaders(cfg)
        model = SynFSModel(cfg.model)
        trainer = SynFSTrainer(cfg, model)
        best_val_auroc = trainer.train(train_loader, val_loader)
        
    return best_val_auroc


if __name__ == "__main__":
    main()
