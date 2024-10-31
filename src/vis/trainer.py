import sys
sys.path.append('.')

import re
import os
import wandb
from time import time
from io import StringIO
from copy import deepcopy
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torchxrayvision import models
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from vision_FT.dataset import getDataLoadersFor5FoldCV


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def printAndLog(msg: str, logger: StringIO = None):
    """
    Utility function to print and log the message.

    Args:
        msg (str): message to print and log, if logger is provided.
        logger (StringIO, optional): Logger file. Defaults to None.
    """
    print(msg)
    if logger:
        msg = msg + "\n" if not msg.endswith("\n") else msg
        logger.write(msg)

    return


def customBCE(x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

    x_full = torch.stack((x, 1-x), dim=-1) # (B, 11) || (B, 11) => (B, 11, 2)
    y_full = torch.stack((y, 1-y), dim=-1)
    w_full = torch.stack((weight, weight), dim=-1) if weight is not None else None
    loss = nn.functional.binary_cross_entropy(x_full, y_full, weight=w_full)

    return loss


class Trainer:

    def __init__(self, run_name: str, device: Union[int, str], 
                 num_classes: int, with_tracking: bool = True, 
                 freeze_backbone: bool = True, ckpt_path: Optional[str] = None):

        # device stuff
        self.device = device
        print(f"Using device: {self.device}")
        self.ddp = isinstance(self.device, int)
        print(f"Using DDP? {'yes' if self.ddp else 'no'}")
        self.device_check = not self.ddp or (self.ddp and self.device == 0)

        # data stuff
        self.data_path = "/home/f20212582/git/radio-lm/data/train_data_small.csv"
        self.data_loaders = getDataLoadersFor5FoldCV(self.data_path, ddp=self.ddp)
        assert len(self.data_loaders) == 6, "Data loaders should be of length 5 + 1."

        # resnet backbone
        self.freeze_backbone = freeze_backbone
        self.backbone = models.ResNet(weights="resnet50-res512-all")
        self.backbone = self.backbone.to(self.device)
        pattern = re.compile(r'model.layer4.2.*3')
        for param in self.backbone.parameters():
            param.requires_grad = False
        if freeze_backbone:
            self.backbone.eval()
        else: # self.ddp
            for name, param in self.backbone.named_parameters():
                if pattern.match(name) is not None:
                    param.requires_grad = True
            if self.ddp:
                self.backbone = DDP(self.backbone, device_ids=[self.device])
        # store for resetting params
        self.init_state_dict = {
            "backbone": deepcopy(self.backbone.state_dict())
        }

        # model
        self.model = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        ).to(self.device)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device])
        # store for resetting params
        self.init_state_dict["model"] = deepcopy(self.model.state_dict())

        # loss and optimizer
        self.loss_fn = customBCE # nn.CrossEntropyLoss()
        if freeze_backbone:
            params = self.model.parameters()
        else:
            params = list(self.model.parameters()) + list(self.backbone.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
        self.fold = 0
        self.epoch = 0

        # logging
        if not os.path.exists("vision_FT/experiments"):
            os.makedirs("vision_FT/experiments")
        self.expt_path = "vision_FT/experiments/" + run_name
        if not os.path.exists(self.expt_path):
            os.makedirs(self.expt_path)
        self.logger = open(self.expt_path + "/run.log", "w")
        self.with_tracking = with_tracking
        self.ovr_metrics = {}

        # load checkpoint if provided
        if ckpt_path:
            self.load_ckpt(ckpt_path)

        printAndLog(f"Dataset Size: {len(self.data_loaders[-1]) * 8}")
        num_param1 = sum([p.numel() for p in self.model.parameters()])
        num_param2 = sum([p.numel() for p in self.backbone.parameters() if p.requires_grad])
        printAndLog(f"Trainable Parameters: {(num_param1 + num_param2)/1e6:.2f}M = {num_param1/1e3:.3f}K + {num_param2/1e6:.3f}M")
        printAndLog(f"RUN_NAME: {run_name}")
        printAndLog(f"NUM_CLASSES: {num_classes}")
        printAndLog(f"WITH_TRACKING: {with_tracking}")
        printAndLog(f"FREEZE_BACKBONE: {freeze_backbone}")

    def save_ckpt(self, fold: int, epoch: int = 0):

        model_state = self.model.state_dict() if not self.ddp else self.model.module.state_dict()
        backbone_state = self.backbone.state_dict()

        state = {
            "model": model_state,
            "backbone": backbone_state,
            "optimizer": self.optimizer.state_dict(),
            "best_metrics": self.best_metrics,
            "fold": fold,
            "epoch": epoch,
        }

        if self.device_check:
            printAndLog(f"Saving checkpoint for fold {fold} at epoch {epoch}...", self.logger)
            save_path = self.expt_path + f"/{fold}_best.pt"
            torch.save(state, save_path)

    def load_ckpt(self, path: str):

        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.backbone.load_state_dict(state["backbone"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_metrics = state["best_metrics"]
        self.fold = state["fold"]
        self.epoch = state["epoch"]

    def step(self, img, label, weights) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img = img.to(self.device)
        label = label.to(self.device)
        weights = weights.to(self.device)

        features = self.backbone.features(img)
        out = self.model(features)
        logits = torch.sigmoid(out)

        loss = self.loss_fn(logits, label, weight=weights)
        preds = out.reshape(-1) > 0.5
        labels = label.reshape(-1)
        accuracy = (preds == labels).float().mean()

        return loss, accuracy, preds

    def train_epoch(self, loader: DataLoader):

        total_loss = []
        total_accuracy = []
        start = time()
        for imgs, labels, weights in loader:
            loss, accuracy, _ = self.step(imgs, labels, weights)
            if self.ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                accuracy /= dist.get_world_size()

            total_loss += [loss.item()]
            total_accuracy += [accuracy.item()]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_metrics = {
            "loss": sum(total_loss) / len(total_loss),
            "accuracy": sum(total_accuracy) / len(total_accuracy),
            "time": time() - start,
        }
        return train_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):

        val_loss = []
        predictions, targets = [], []
        start = time()
        for imgs, labels, weights in loader:
            labels = labels.to(self.device)
            loss, _, pred = self.step(imgs, labels, weights)

            if self.ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                ovr_pred = [torch.zeros_like(pred, device=self.device) for _ in range(dist.get_world_size())]
                dist.all_gather(ovr_pred, pred)
                pred = torch.cat(ovr_pred, dim=0).reshape(-1).cpu().detach().tolist()

                ovr_target = [torch.zeros_like(labels, device=self.device) for _ in range(dist.get_world_size())]
                dist.all_gather(ovr_target, labels)
                target = torch.cat(ovr_target, dim=0).reshape(-1).cpu().detach().tolist()
            else:
                pred = pred.reshape(-1).cpu().detach().tolist()
                target = labels.reshape(-1).cpu().detach().tolist()

            val_loss += [loss.item()]
            predictions.extend(pred)
            targets.extend(target)

        val_loss = sum(val_loss) / len(val_loss)
        accuracy = (torch.tensor(predictions) == torch.tensor(targets)).float().mean().item()
        val_metrics = {
            "loss": val_loss,
            "accuracy": accuracy,
            "time": time() - start,
        }

        return val_metrics

    def train(self, num_epochs: int = 20):

        for fold_num, (train_loader, val_loader) in enumerate(self.data_loaders[self.fold:-1]):

            self._on_fold_begin(fold_num)

            start = time()
            for epoch in range(self.epoch, num_epochs):

                # train
                self._on_train_epoch_start()
                train_metrics = self.train_epoch(train_loader)
                self._on_train_epoch_end(train_metrics, epoch)

                # validate
                self._on_val_epoch_start()
                val_metrics = self.evaluate(val_loader)
                self._on_val_epoch_end(val_metrics, fold_num, epoch)

            end = time()
            self._on_fold_end(fold_num, end-start)

        printAndLog("Training complete", self.logger)
        printAndLog("Training on whole dataset to save overall model...", self.logger)
        self._on_fold_begin(6)
        for epoch in range(num_epochs):
            self._on_train_epoch_start()
            train_metrics = self.train_epoch(self.data_loaders[-1])
            self._on_train_epoch_end(train_metrics, epoch)

        self.save_ckpt(6, epoch=-1)
        avg_accuracy = sum([self.ovr_metrics[i]["accuracy"] for i in range(5)]) / 5
        printAndLog("-"*88, self.logger)
        printAndLog(f"Average accuracy over 5 folds: {avg_accuracy:.2f}", self.logger)
        printAndLog("-"*88, self.logger)
        self.logger.flush()
        self.logger.close()
        return

    def _on_fold_begin(self, fold_num: int):

        # initialize best_metrics if running from scratch
        if self.fold == 0:
            self.best_metrics = {
                "accuracy": 0.0,
            }

        # reset model parameters after each fold, it's a linear layer, so use uniform -root(1/k) to root(1/k)
        if fold_num != 0:
            model_dict = self.init_state_dict["model"]
            self.model.load_state_dict(model_dict)
            if not self.freeze_backbone:
                backbone_dict = self.init_state_dict["backbone"]
                self.backbone.load_state_dict(backbone_dict)

        # Initialize a new wandb run for each fold with a distinct name and group them under one group
        if self.device_check:
            printAndLog(f"Starting fold {fold_num}...", self.logger)
            if self.with_tracking:
                wandb.init(
                    project="vision_FT", 
                    entity="bits-goa",
                    group=self.expt_path, # Group all fold runs under one experiment
                    name=f"fold_{fold_num}", # Unique name for each fold
                )

    def _on_fold_end(self, fold_num: int, time_taken: float):

        self.ovr_metrics[fold_num] = self.best_metrics
        if self.device_check:
            printAndLog(f"Fold {self.fold} complete, took {time_taken/60:.2f}m", self.logger)
            printAndLog("-"*88, self.logger)
            if self.with_tracking:
                wandb.join()

    def _on_train_epoch_start(self):
        self.model.train()
        # mayebmore?

    def _on_train_epoch_end(self, train_metrics: dict, epoch: int):

        if self.device_check:
            printAndLog("-"*88)
            printAndLog(f"Epoch: {epoch}, Train Loss: {train_metrics['loss']:.2f}, Train Accuracy: {train_metrics['accuracy']:.2f}, Time/epoch: {train_metrics['time']:.3f}s",
                        self.logger)
            if self.with_tracking:
                wandb.log({
                    "Train Loss": train_metrics["loss"], 
                    "Train Accuracy": train_metrics["accuracy"]},
                    step=epoch)

    def _on_val_epoch_start(self):
        self.model.eval()
        # maybemore?

    def _on_val_epoch_end(self, val_metrics: dict, fold: int, epoch: int):

        if val_metrics["accuracy"] > self.best_metrics["accuracy"] and epoch > 10: 
            # note the epoch check, the starting model is quite random 
            # and gets high accuracy out of pure luck, so we wait for 
            # a few epochs
            self.best_metrics["accuracy"] = val_metrics["accuracy"]

        if self.device_check:
            if epoch % 25 == 0:
                self.save_ckpt(fold, epoch)
            printAndLog(f"Epoch: {epoch}, Val Loss: {val_metrics['loss']:.2f}, Val Accuracy: {val_metrics['accuracy']:.2f}, Time/epoch: {val_metrics['time']:.3f}s", 
                        self.logger)
            printAndLog(f"Best Val Accuracy: {self.best_metrics['accuracy']:.2f}", self.logger)
            if self.with_tracking:
                wandb.log({
                    "Val Loss": val_metrics["loss"], 
                    "Val Accuracy": val_metrics["accuracy"]},
                    step=epoch)


def DDP_launch(rank: int, world_size: int, run_name: str, num_classes: int, 
               with_tracking: bool = True, freeze_backbone: bool = True,
               ckpt_path: Optional[str] = None):

    setup(rank, world_size)
    trainer = Trainer(run_name, rank, num_classes, 
                      with_tracking, freeze_backbone, ckpt_path)
    trainer.train(250)
    cleanup()
    return


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    run_name = "better"
    num_classes = 6
    with_tracking = True
    ckpt_path = None
    freeze_backbone = False
    if world_size > 1:
        mp.spawn(DDP_launch, args=(world_size, run_name, num_classes, 
                                   with_tracking, freeze_backbone, ckpt_path), 
                 nprocs=world_size)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(run_name, device, num_classes, with_tracking, 
                          freeze_backbone, ckpt_path)
        trainer.train(250)
