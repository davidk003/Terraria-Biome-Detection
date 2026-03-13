# dataprep.py

import torchvision.transforms as T

# class order must match inference code

CLASSES = [
    'Corruption', 'Crimson', 'Desert', 'Dungeon', 'Forest',
    'Hallow', 'Hell', 'Jungle', 'Mushroom', 'Ocean',
    'Snow', 'Space', 'Underground'
]

# normalization values used in inference script
MEAN = [0.1473, 0.1647, 0.2079]
STD = [0.1967, 0.2150, 0.2937]

# final inference resolution
INPUT_HEIGHT = 216
INPUT_WIDTH = 384


def get_train_transforms():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((240,420)),
        T.RandomCrop((INPUT_HEIGHT,INPUT_WIDTH)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])


def get_val_transforms():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])