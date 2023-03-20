# ⚡ Simple-Lightning-Template

This is a basic and simple template for PyTorch projects that utilizes PyTorch Lightning and Weight and Biases (Wandb).
The goal of this project is to streamline the process of creating PyTorch projects from scratch or adapting existing
ones using these helpful libraries.

# How to Use this Project

1. Wrap your all your data processing step (which includes the creation of datasets, augmentations, and dataloaders)
   into a [Lightning datamodule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) (example
   provided in the 'datamodules' folder).
2. Wrap your Torch model into
   a [Lightning model](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) (example
   provided in the 'model' folder), more specifically from a classical Pytorch project :
    * Place the training/validation/test into the Lightning module using the appropriate hook and log the metrics with
      the logger
    * Use the configure_optimizers() hook to move Optimizers and schedulers to the Lightning model
3. Create a new config file or modify the existing 'config.yaml' to include all the hyperparameters you need inside your
   model and datamodule and to specify the right path to your data or model checkpointing

# Note

The configuration file is currently managed using the Omegaconf package, which simply allows to transform the YAML 
config file into a python object similar to argparse. Using Hydra, which also uses Omegaconf, would
currently introduce too much complexity since it still create some frictions with other libraries and functionalities 
such as DDP training or W&B sweeps.

# Requirements

* Python >= 3.5
* Pytorch
* PyTorch-Lightning
* Wandb
* Omegaconf

# Features

* Easy hyperparameters management with YAML config files with Omegaconf
* Modular code using Pytorch Lightning models and datamodules
* Native support Multi GPU, mix precision training with Pytorch Lightning
* Clean logging of the metrics with W&B

# Todo :

- [ ] Wandb sweeps
- [ ] Advanced scheduling
- [ ] Clean architecture loading from arch files (add path or create package ?)
- [ ] Automatic num_classes computation
- [ ] Dependency management with virtual env
- [ ] Finetuning example
- [ ] Multiple datasets
- [ ] Add basic/intermediate examples in this repo of Lighting data modules and Lighting models
  (e.g. contrastive learning with fine-tuning, semi-supervised learning or other task like object detection,
  segmentation or NLP tasks) ex

