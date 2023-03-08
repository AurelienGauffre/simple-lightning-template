import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules.imagedatamodule import Imagenette160Datamodule
from models.imageclassifier import ImageClassifier

# Parsing CLI and config files
parser = argparse.ArgumentParser(description='Lightning')
parser.add_argument('--config', type=str, default='config.yaml', help='config file')
args = parser.parse_args()
params = OmegaConf.load(Path(Path(__file__).parent.resolve() / 'configs' / args.config))
params.root_dir = str(Path(__file__).parent.resolve())

if __name__ == "__main__":
    seed_everything(params.seed, workers=True)
    # Wandb Logger
    wandb_logger = WandbLogger(project=params.wandb.project,
                               name=params.wandb.name,
                               save_dir=str(Path(params.root_dir) / 'saved')
                               )

    # Lightning Datamodules (handles dataset, dataloader, augmentations or collate functions)
    datamodule = Imagenette160Datamodule(params)

    # Lighting Model (architecture, optimizer, scheduler)
    model = ImageClassifier(params)

    # Lighting Model checkpointing strategy (save best model according to val accuracy)
    model_checkpoint_callback = ModelCheckpoint(dirpath=str(Path(params.root_dir) / 'saved'),
                                                filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
                                                monitor="val_acc",
                                                mode="max",
                                                save_on_train_epoch_end=True,
                                                )
    # model_checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last" #consistent name for last epoch

    trainer = Trainer(
        max_epochs=params.epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=20),
            model_checkpoint_callback],
        logger=wandb_logger,
        precision=params.precision
    )

    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule) #no test dataloader
