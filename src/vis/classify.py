"""
Provides a simple way to interact with the vision model.    
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.vis.model import ResNetAndHead


def classify_loader(loader: DataLoader, ckpt_path: str, num_classes: int) -> None:
    """
    Classifies the images in the loader using the model saved at ckpt_path.
    Prints the predictions.
    """
    # load the model
    model = ResNetAndHead(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    ovr = []
    # iterate over the loader
    for images, labels, _, cases in loader:
        predictions = model(images)
        predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        ovr.extend((predictions, labels, cases))