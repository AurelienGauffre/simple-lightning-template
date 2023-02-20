# ⚡ Lightning-Wandb-template
This is a basic and simple template for PyTorch projects that utilizes PyTorch Lightning and Weight and Biases (Wandb).
The goal of this project is to streamline the process of creating PyTorch projects from scratch or adapting existing 
ones using these helpful libraries.

# How to Use this Project
1. Wrap your all your data processing step (which includes the creation of datasets, augmentations, and dataloaders) 
into a [Lightning datamodule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) (example provided in the 'datamodules' folder). 
2. Wrap your Torch model into a [Lightning model](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) (example provided in the 'model' folder), more specifically :
   * Place the training/validation/test into the Lightning module using the appropriate hook and log the metrics with the logger
   * Use the configure_optimizers() hook to move Optimizers and schedulers to the Lightning model
3. Create a new config file or modify the existing 'config.yaml' to include all the hyperparameters you need inside your model and datamodule and to specify right path to your data or model checkpointing
# Note
The configuration file is currently managed using the Omegaconf package. I find that Hydra, which also uses Omegaconf, adds a real layer of complexity
which might not bt not necessary, especially since it raises issues with DDP training, W&B sweeps and more.

# Requirements
* Python >= 3.5
* Pytorch
* PyTorch-Lightning
* Wandb
* Omegaconf

# Features
* Easy hyper parameters management with yaml config file with Omegaconf
* Very modular code using Pytorch Lightning models and datamodules
* Native support Multi GPU, mix precision training with to Pytorch Lightning
* Clean logging of the metrics with W&B


# Todo :
- [ ] Models clean checkpointing
- [ ] Wandb sweeps
- [ ] Advanced scheduling
- [ ] Multi GPU training
- [ ] Clean architecture loading from arch files (add path or create package ?)
- [ ] Mix precision
- [ ] Add basic/intermediate examples in this repo of Lighting data modules and Lighting models 
(e.g. contrastive learning with fine-tuning, semi-supervised learning or other task like object detection, segmentation or NLP tasks) ex

