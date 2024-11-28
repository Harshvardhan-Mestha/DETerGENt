import sys
sys.path.append(".")
import os
import wandb
import pandas as pd
from time import time
from io import StringIO
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.vis.model import ResNetAndHead
from src.vis.dataset import getDataLoadersFor5FoldCV


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def print_and_log(msg: str, logger: StringIO = None):
    """
    Utility function to print and log the message.

    Args
    ----
        msg (str): message to print and log, if logger is provided.
        logger (StringIO, optional): Logger file. Defaults to None.

    Returns
    -------
        None
    """
    print(msg)
    if logger:
        msg = msg + "\n" if not msg.endswith("\n") else msg
        logger.write(msg)


def customBCE(x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

    # x_full = torch.stack((x, 1-x), dim=-1) # (B, 11) || (B, 11) => (B, 11, 2)
    # y_full = torch.stack((y, 1-y), dim=-1)
    # w_full = torch.stack((weight, weight), dim=-1) if weight is not None else None
    # loss = nn.functional.binary_cross_entropy(x_full, y_full, weight=w_full)
    loss = nn.functional.binary_cross_entropy(x, y, weight=weight)

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
        self.data_path = "data/xray_data.csv"
        self.data_loaders = getDataLoadersFor5FoldCV(self.data_path, ddp=self.ddp)
        assert len(self.data_loaders) == 6, "Data loaders should be of length 5 + 1."

        # model stuff
        self.model = ResNetAndHead(num_classes, freeze_backbone).to(self.device)
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device])

        # loss and optimizer
        self.loss_fn = customBCE # nn.CrossEntropyLoss()
        if freeze_backbone:
            params = self.model.head.parameters()
        else:
            params = list(self.model.parameters()) + list(self.model.backbone.parameters())
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

        print_and_log(f"Dataset Size: {len(self.data_loaders[-1]) * 8}")
        num_param1 = sum([p.numel() for p in self.model.head.parameters()])
        num_param2 = sum([p.numel() for p in self.model.backbone.parameters() if p.requires_grad])
        print_and_log(f"Trainable Parameters: {(num_param1 + num_param2)/1e6:.2f}M = {num_param1/1e3:.3f}K + {num_param2/1e6:.3f}M")
        print_and_log(f"RUN_NAME: {run_name}")
        print_and_log(f"NUM_CLASSES: {num_classes}")
        print_and_log(f"WITH_TRACKING: {with_tracking}")
        print_and_log(f"FREEZE_BACKBONE: {freeze_backbone}")

    def save_ckpt(self, fold: int, epoch: int = 0):
        """
        Saves a checkpoint of the model, optimizer, and best metrics.

        Args
        ----
            fold (int)
                Current fold number.
            epoch (int, optional)
                Current epoch number. Defaults to 0.

        Returns
        -------
            None
        """

        model_state = self.model.state_dict() if not self.ddp else self.model.module.state_dict()
        backbone_state = self.model.backbone.state_dict()

        state = {
            "model": model_state,
            "backbone": backbone_state,
            "optimizer": self.optimizer.state_dict(),
            "best_metrics": self.best_metrics,
            "fold": fold,
            "epoch": epoch,
        }

        if self.device_check:
            print_and_log(f"Saving checkpoint for fold {fold} at epoch {epoch}...", self.logger)
            save_path = self.expt_path + f"/{fold}_best.pt"
            torch.save(state, save_path)

    def load_ckpt(self, path: str):
        """
        Loads a checkpoint of the model, optimizer, and best metrics.

        Args
        ----
            path (str)
                Path to the checkpoint file.

        Returns
        -------
            None
        """

        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.backbone.load_state_dict(state["backbone"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best_metrics = state["best_metrics"]
        self.fold = state["fold"]
        self.epoch = state["epoch"]

    def step(self, img, label, weights) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a step in the training loop.

        Args
        ----
            img (torch.Tensor)
                Image tensor.
            label (torch.Tensor)
                Label tensor.
            weights (torch.Tensor)
                Weight tensor. (Optional)

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                Loss, accuracy, and predictions.
        """

        img = img.to(self.device)
        label = label.to(self.device)
        weights = weights.to(self.device)

        out = self.model(img)
        logits = torch.sigmoid(out)

        loss = self.loss_fn(logits, label, weight=weights)
        preds = out.reshape(-1) > 0.5
        labels = label.reshape(-1)
        accuracy = (preds == labels).float().mean()

        return loss, accuracy, preds

    def train_epoch(self, loader: DataLoader) -> dict:
        """
        Train the model for one epoch.

        Args
        ----
            loader (DataLoader)
                DataLoader for training data.

        Returns
        -------
            dict
                Training metrics (loss, accuracy, time).
        """

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
    def evaluate(self, loader: DataLoader) -> dict:
        """
        Evaluate the model on the validation set.

        Args
        ----
            loader (DataLoader)
                DataLoader for validation data.

        Returns
        -------
            dict
                Validation metrics (loss, accuracy, time).
        """

        val_loss = []
        predictions, targets = [], []
        start = time()
        for imgs, labels, weights, _ in loader:
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
            "time": time() - start
        }

        return val_metrics

    def train(self, num_epochs: int = 20):
        """
        Trains the model in a 5-fold cross-validation manner.

        Args
        ----
            num_epochs (int, optional)
                Number of epochs to train per fold. Defaults to 20.

        Returns
        -------
            None
        """

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
            self._on_fold_end(fold_num, end-start, val_loader)

        print_and_log("Training complete", self.logger)
        print_and_log("Training on whole dataset to save overall model...", self.logger)
        self._on_fold_begin(6)
        for epoch in range(num_epochs):
            self._on_train_epoch_start()
            train_metrics = self.train_epoch(self.data_loaders[-1])
            self._on_train_epoch_end(train_metrics, epoch)

        self.save_ckpt(6, epoch=-1)
        avg_accuracy = sum([self.ovr_metrics[i]["accuracy"] for i in range(5)]) / 5
        print_and_log("-"*88, self.logger)
        print_and_log(f"Average accuracy over 5 folds: {avg_accuracy:.2f}", self.logger)
        print_and_log("-"*88, self.logger)
        self.logger.flush()
        self.logger.close()

    def _on_fold_begin(self, fold_num: int):
        """
        Stuff to do at the beginning of each fold.
        Currently, it initializes the best_metrics if running from scratch,
        resets model parameters after each fold, 
        and initializes a new wandb run.

        Args
        ----
            fold_num (int): Current fold number.

        Returns
        -------
            None
        """

        # initialize best_metrics if running from scratch
        if self.fold == 0:
            self.best_metrics = {
                "accuracy": 0.0,
            }

        # reset model parameters after each fold
        if fold_num != 0:
            self.model._reset_head()
            self.model._reset_backbone()

        # Initialize a new wandb run for each fold with a distinct name and group them under one group
        if self.device_check:
            print_and_log(f"Starting fold {fold_num}...", self.logger)
            if self.with_tracking:
                wandb.init(
                    project="vision_FT", 
                    entity="bits-goa",
                    group=self.expt_path, # Group all fold runs under one experiment
                    name=f"fold_{fold_num}", # Unique name for each fold
                )

    def _on_fold_end(self, fold_num: int, time_taken: float, loader: DataLoader):
        """
        Stuff to do at the end of each fold.
        Currently, it saves the best metrics for the fold,
        prints the time taken,
        and closes the wandb run.

        Args
        ----
            fold_num (int)
                Current fold number.
            time_taken (float)
                Time taken to complete the fold.

        Returns
        -------
            None
        """

        self.ovr_metrics[fold_num] = self.best_metrics
        if self.device_check:
            print_and_log(f"Fold {self.fold} complete, took {time_taken/60:.2f}m", self.logger)
            print_and_log("-"*88, self.logger)
            if self.with_tracking:
                wandb.join()

        # load best model for predictions
        print_and_log("Loading best model for predictions...", self.logger)
        self.load_ckpt(self.expt_path + f"/{fold_num}_best.pt")

        # save predictions
        all_predictions = {
            "predictions": [],
            "case": [],
            "label": [],
        }
        for imgs, labels, weights, case_num in loader:
            labels = labels.tolist()
            cases = case_num.tolist()
            cases = [f"case_{x}" for x in cases]
            _, _, pred = self.step(imgs, labels, weights)
            if self.ddp:
                ovr_pred = [torch.zeros_like(pred, device=self.device) for _ in range(dist.get_world_size())]
                dist.all_gather(ovr_pred, pred)
                pred = torch.cat(ovr_pred, dim=0).reshape(-1).cpu().detach().tolist()
            else:
                pred = pred.reshape(-1).cpu().detach().tolist()

            all_predictions["predictions"].extend(pred)
            all_predictions["case"].extend(cases)
            all_predictions["label"].extend(labels)

        print_and_log(f"Saving predictions for fold {fold_num}...", self.logger)
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_csv(f"{fold_num}_predictions.csv", index=False)

    def _on_train_epoch_start(self):
        """
        Stuff to do on the start of each training epoch.
        Not much here, just setting the model to train mode.
        """
        self.model.train()
        # mayebmore?

    def _on_train_epoch_end(self, train_metrics: dict, epoch: int):
        """
        Stuff to do at the end of each training epoch.
        Currently, it prints the training metrics,
        logs the metrics to wandb if enabled.

        Args
        ----
            train_metrics (dict)
                Training metrics (loss, accuracy, time) for the epoch.
            epoch (int)
                Current epoch number.

        Returns
        -------
            None
        """

        if self.device_check:
            print_and_log("-"*88)
            print_and_log(f"Epoch: {epoch}, Train Loss: {train_metrics['loss']:.2f}, Train Accuracy: {train_metrics['accuracy']:.2f}, Time/epoch: {train_metrics['time']:.3f}s",
                        self.logger)
            if self.with_tracking:
                wandb.log({
                    "Train Loss": train_metrics["loss"], 
                    "Train Accuracy": train_metrics["accuracy"]},
                    step=epoch)

    def _on_val_epoch_start(self):
        """
        Stuff to do at the start of each validation epoch.
        Not much here, just setting the model to eval mode.
        """
        self.model.eval()
        # maybemore?

    def _on_val_epoch_end(self, val_metrics: dict, fold: int, epoch: int):
        """
        Stuff to do at the end of each validation epoch.
        Currently, it prints the validation metrics,
        logs the metrics to wandb if enabled,
        saves the checkpoint if the model is better than the previous best OR
        after every 25 epochs.

        Args
        ----
            val_metrics (dict)
                Validation metrics (loss, accuracy, time) for the epoch.
            fold (int)
                Current fold number.
            epoch (int)
                Current epoch number.

        Returns
        -------
            None
        """

        if val_metrics["accuracy"] > self.best_metrics["accuracy"] and epoch > 10: 
            # note the epoch check, the starting model is quite random 
            # and gets high accuracy out of pure luck, so we wait for 
            # a few epochs
            self.best_metrics["accuracy"] = val_metrics["accuracy"]

        if self.device_check:
            if epoch % 25 == 0:
                self.save_ckpt(fold, epoch)
            print_and_log(f"Epoch: {epoch}, Val Loss: {val_metrics['loss']:.2f}, Val Accuracy: {val_metrics['accuracy']:.2f}, Time/epoch: {val_metrics['time']:.3f}s", 
                        self.logger)
            print_and_log(f"Best Val Accuracy: {self.best_metrics['accuracy']:.2f}", self.logger)
            if self.with_tracking:
                wandb.log({
                    "Val Loss": val_metrics["loss"], 
                    "Val Accuracy": val_metrics["accuracy"]},
                    step=epoch)


def DDP_launch(rank: int, world_size: int, run_name: str, num_classes: int, 
               with_tracking: bool = True, freeze_backbone: bool = True,
               ckpt_path: Optional[str] = None):
    """
    Wrapper to launch DDP training.

    Args
    ----
        rank (int)
            Device number.
        world_size (int)
            Number of devices.
        run_name (str)
            Name of the run set by the user.
        num_classes (int)
            Number of classes in the dataset.
        with_tracking (bool, optional)
            Enables wandb tracking. Defaults to True.
        freeze_backbone (bool, optional)
            Freezes ViT backbone. Defaults to True.
        ckpt_path (Optional[str], optional)
            Checkpoint to load if resuming. Defaults to None.

    Returns
    -------
        None
    """

    setup(rank, world_size)
    trainer = Trainer(run_name, rank, num_classes, 
                      with_tracking, freeze_backbone, ckpt_path)
    trainer.train(10)
    cleanup()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    run_name = "base"
    num_classes = 11
    with_tracking = False
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
        trainer.train(10)
