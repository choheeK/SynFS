import hydra
from omegaconf import DictConfig

from src.data.datamodule import build_dataloaders
from src.trainer.trainer import SynFSTrainer
from src.utils.seed import fix_seed
from src.models.synfs_model import SynFSModel


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    fix_seed(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    model = SynFSModel(cfg.model)
    trainer = SynFSTrainer(cfg, model)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
