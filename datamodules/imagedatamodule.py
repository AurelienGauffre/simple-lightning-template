import torch
import lightly

from torch import nn
import torchvision
import PIL
import os
import pathlib
import pytorch_lightning as pl

from pathlib import Path


class ImageDataModule(pl.LightningDataModule):
    """Basic image datamodule to load images from a folder for image classification.
    The dataset folder must contain at least two folders, 'train' and 'val', each of which being in Pytroch ImageFolder
    format (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).
    """

    def __init__(self, params):
        super().__init__()
        self.batch_size = params.batch_size
        self.nb_workers = params.nb_workers
        self.data_dir_train = Path(Path(params.dataset_path) / 'train')
        self.data_dir_val = Path(Path(params.dataset_path) / 'val')

        self.train_transforms = None  # todo RaiseImplement error ?
        self.val_transforms = None
        self.num_classes = None

    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(str(self.data_dir_train), transform=self.train_transforms)

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.nb_workers,
        )

    def val_dataloader(self):
        val_dataset = torchvision.datasets.ImageFolder(str(self.data_dir_val), transform=self.val_transforms)

        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.nb_workers,
        )


class Imagenette160Datamodule(ImageDataModule):
    """ Download dataset at : https://github.com/fastai/imagenette"""

    def __init__(self, params):
        super().__init__(params)
        self.num_classes = params.num_classes
        tr_normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            tr_normalize
        ])

        self.val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            tr_normalize
        ])
