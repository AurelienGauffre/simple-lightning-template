import argparse
import torch
import sys
from omegaconf import OmegaConf
from pathlib import Path

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from datamodules.imagedatamodule import Imagenette160Datamodule
from models.imageclassifier import ImageClassifier
from dic import dic_datamodules, dic_models

# Parsing config files
# parser = argparse.ArgumentParser(description='Lightning')
# parser.add_argument('--config', type=str, default='config.yaml', help='config file')
# args = parser.parse_args()
params = OmegaConf.load(Path(Path(__file__).parent.resolve() / 'configs' / sys.argv[1]))
params.root_dir = str(Path(__file__).parent.resolve())

# Logger
wandb_logger = WandbLogger(project=params.wandb.project,
                           name=params.wandb.name,
                           save_dir=str(Path(params.root_dir) / 'saved'))

# Lightning Datamodules (dataset, dataloader, augmentation)
datamodule = dic_datamodules[params.datamodule.name](params)
# Lighting Model (architecture, optimizer, scheduler)
model = dic_models[params.model.name](params)

trainer = Trainer(
    max_epochs=params.epochs,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=20)],
    logger=wandb_logger
)

trainer.fit(model=model, datamodule=datamodule)
# trainer.test(model=model, datamodule=datamodule) #no test dataloader
