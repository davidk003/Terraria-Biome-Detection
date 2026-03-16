# src/resnet_model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from dataprep import CLASSES

NUM_CLASSES = len(CLASSES)

def get_resnet18(weights=ResNet18_Weights.DEFAULT, freeze_backbone=False):

    # load pretrained ResNet-18
    model = resnet18(weights=weights)

    # replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:  # keep classifier trainable
                param.requires_grad = False

    return model