import os
import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from torch.nn import functional as F
from typing import Union, List, Tuple
from torchvision import transforms as T
from sklearn.model_selection import KFold
from torchxrayvision.datasets import normalize
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, DataLoader


label2int = {
    "Atelectasis": torch.tensor(0),
    "Cardiomegaly": torch.tensor(1),
    "Calcifications": torch.tensor(2), 
    "COPD": torch.tensor(3), 
    "Lung Nodules": torch.tensor(4), 
    "Mesothelioma": torch.tensor(5),
    "Cardiomegaly": torch.tensor(6), 
    "Plueral Effusion": torch.tensor(7),
    "Pneumonia": torch.tensor(8), 
    "Pneumothorax": torch.tensor(9), 
    "Tuberculosis": torch.tensor(10),
    np.nan: np.nan
}

label2int_small = {
    "Atelectasis": torch.tensor(0),
    "Cardiomegaly": torch.tensor(1),
    "Plueral Effusion": torch.tensor(2),
    "Pneumothorax": torch.tensor(3),
    "Lung Nodules": torch.tensor(4),
    "Pneumonia": torch.tensor(5),
    np.nan: np.nan
}


class XRayDataset(Dataset):

    def __init__(self, data: Union[str, pd.DataFrame], training: bool = True):
        """
        Initialize the dataset.

        Args:
            data (Union[str, pd.DataFrame]): path to CSV file / DataFrame of the CSV, attributes : [image_path, label1, label2, label3, case]
            training (bool, optional): wether to use in training mode. Defaults to True.
        """

        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            assert isinstance(data, pd.DataFrame), "data should be either path to CSV file or pandas DataFrame"
            self.data = data

        # drop erroneous rows (45, 147, 158)
        mask = self.data["case"].isin(["case_46", "case_148", "case_159"])
        self.data = self.data[~mask]

        # Split the 'labels' column into separate columns
        split_labels = data['label'].str.split(',', expand=True)

        for i in range(3):  # Add missing columns if they don't exist
          if i >= split_labels.shape[1]:
            split_labels[i] = ''

        # Rename the new columns (optional)
        split_labels.columns = ["label1","label2","label3"]

        # Concatenate the original DataFrame with the new columns
        self.data = pd.concat([data, split_labels], axis=1)

        self.data['label1'] = self.data['label1'].map(label2int)
        self.data['label2'] = self.data['label2'].map(label2int)
        self.data['label3'] = self.data['label3'].map(label2int)
        self.num_classes = len(label2int) 
        self.training = training

        transform_list = [
            T.Resize((512, 512)),
        ]
        if training:
            transform_list = [
                *transform_list,
                T.RandomRotation(15),
                T.RandomHorizontalFlip(),
            ]
        self.transforms = T.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.data.iloc[idx]["img_dir"]
        image_path = f"./data/" + x + "/" + os.listdir(f"./data/" + x)[0]
        image = image_path
        image = normalize(imread(image), 255)
        if image.ndim > 2:
            image = image.mean(2)
        image = torch.from_numpy(image)[None, ...] # (1, 512, 512)
        image = self.transforms(image)
        
        label = F.one_hot(self.data.iloc[idx]["label1"], self.num_classes).to(torch.float32)
        label += F.one_hot(self.data.iloc[idx]["label2"], self.num_classes).to(torch.float32) if not np.isnan(self.data.iloc[idx]["label2"]) else 0
        label += F.one_hot(self.data.iloc[idx]["label3"], self.num_classes).to(torch.float32) if not np.isnan(self.data.iloc[idx]["label3"]) else 0
        weights = torch.ones_like(label)
        if self.training:
            return image, label, weights
        else:
            return image, label, weights, torch.tensor(self.data.iloc[idx]["case"])


def getDataLoadersFor5FoldCV(data: str, batch_size: int = 8, ddp: bool = True) -> List:
    """
    Get the dataloaders for 5 fold cross validation.

    Args:
        data (str): path to CSV file, attributes : [image_path, label1, label2, label3]
        batch_size (int, optional): batch size. Defaults to 32.
        ddp (bool, optional): whether to use DistributedDataParallel. Defaults to True.

    Returns:
        list: list of dataloaders for 5 fold cross validation
    """
    data = pd.read_csv(data)
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    # y = data["label1"].values
    kf = KFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    dataloaders = []
    for i in range(5):
        train_data = data[data.kfold != i].reset_index(drop=True)
        valid_data = data[data.kfold == i].reset_index(drop=True)
        train_dataset = XRayDataset(train_data)
        valid_dataset = XRayDataset(valid_data, training=False)
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False) if ddp else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True if train_sampler is None else False, 
                                  drop_last=True, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                                  shuffle=False, drop_last=True, sampler=valid_sampler)
        dataloaders.append((train_loader, valid_loader))

    # full data loader
    full_dataset = XRayDataset(data)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    dataloaders.append(full_loader)

    return dataloaders


if __name__ == "__main__":
    dataloaders = getDataLoadersFor5FoldCV("/home/f20212582/git/radio-lm/data/train_data.csv")
    for train_loader, valid_loader in dataloaders[:-1]:
        print(len(train_loader), len(valid_loader))
        for i, (x, y) in enumerate(train_loader):
            print(x.shape, y.shape)
            if i == 0:
                break
    print(len(dataloaders[-1]))