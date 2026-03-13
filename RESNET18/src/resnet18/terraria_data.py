# resnet18/terraria_data.py

import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from dataprep import CLASSES  # your existing CLASSES list

class TerrariaBiomeDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, crop_size=(216,384), crops_per_image=9):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image

    def __len__(self):
        return len(self.data) * self.crops_per_image

    def __getitem__(self, idx):

        img_index = idx // self.crops_per_image
        row = self.data.iloc[img_index]
    
        img_path = os.path.join(self.root_dir, row['filename'])
        img = cv2.imread(img_path)
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        label = CLASSES.index(row['biome'])
    
        h, w, _ = img.shape
        ch, cw = self.crop_size
    
        center_y = h // 2
        center_x = w // 2
    
        jitter_y = h // 6
        jitter_x = w // 6
    
        cy = center_y + torch.randint(-jitter_y, jitter_y + 1, (1,)).item()
        cx = center_x + torch.randint(-jitter_x, jitter_x + 1, (1,)).item()
    
        top = max(0, min(h - ch, cy - ch // 2))
        left = max(0, min(w - cw, cx - cw // 2))
    
        img = img[top:top+ch, left:left+cw]
    
        if self.transform:
            img = self.transform(img)
    
        return img, label


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader