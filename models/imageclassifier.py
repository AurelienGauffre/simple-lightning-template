import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torchmetrics.functional import accuracy
from torchvision.models import resnet18, resnet34, resnet50
from archs.resnet import ResNet18, ResNet34, ResNet50, ResNet101
import torch
import time

dic_arch = {'resnet18': resnet18(),
            'resnet34': resnet34(),
            'resnet50': resnet50()}

class ImageClassifier(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.start_time = None
        self.arch = dic_arch[params.arch]
        self.lr = params.lr
        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.num_classes = params.num_classes

    def forward(self, x):
        out = self.arch(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)

        if stage:
            self.log_dict({f"{stage}_loss": loss, f"{stage}_acc": acc}, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        end_time = time.time()
        elapsed_time = int(end_time - self.start_time)/60
        self.log_dict({"total time": elapsed_time,"epoch avg time": elapsed_time/self.epochs}, on_epoch=True, on_step=False)
     
