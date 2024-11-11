"""
This contains a basic ResNet + Head model.
"""

import re
from copy import deepcopy

import torch
import torch.nn as nn
from torchxrayvision import models

class ResNetAndHead(nn.Module):

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # create backbone
        self._reset_backbone()

        # create head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes)
        )
        self.reset_model_state = deepcopy(self.state_dict())

    def _reset_backbone(self):
        """
        Resets the ResNet backbone to its initial state.
        Assigns it to self.backbone.
        """
        # load a pretrained ResNet model
        backbone = models.ResNet(weights="resnet50-res512-all")

        # unfreeze weights selectively
        pattern = re.compile(r'model.layer4.2.*3')
        for param in backbone.parameters():
            param.requires_grad = False
        if self.freeze_backbone:
            backbone.eval()
        else:
            for name, param in backbone.named_parameters():
                if pattern.match(name) is not None:
                    param.requires_grad = True
        self.backbone = backbone

    def _reset_head(self):
        """
        Resets the head to its initial state.
        Loads the initial state dict into self.head.
        """
        original_state = deepcopy(self.reset_model_state)
        self.head.load_state_dict(original_state)

    def forward(self, x: torch.Tensor):
        """
        Simple forward pass through the model.

        Args:
            x (torch.Tensor): input tensor of shape (B, C, H, W) / Images
        """

        x = self.backbone(x)
        x = self.head(x)

        return x
