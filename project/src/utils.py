import torch 
import torch.nn as nn 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import shutil

# Create a constants.py file to make it cleaner
# --------Constants---------
# Device 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
LOAD_MODEL = False
SAVE_MODEL = True

# model checkpoint file name 
CHECKPOINT_FILE = "checkpoint.pth.tar"

# Anchor boxes scaled between 0 and 1
ANCHORS = [                                                               
    (0.14105, 0.10625), (0.11015, 0.16249), (0.09875, 0.14781),
    (0.08998, 0.13511), (0.08254, 0.12333), (0.07731, 0.19987),
    (0.07835, 0.09580), (0.06312, 0.11607), (0.06459, 0.21892)]

NUM_ANCHORS = len(ANCHORS)

# Batch size for training 
BATCH_SIZE = 32

# Learning rate for training 
LEARNING_RATE = 1e-4

# Number of epochs for training 
NB_EPOCHS = 400

# Image sizes
ORIG_IMG_WIDTH = 6000
ORIG_IMG_HEIGHT = 4000
RESIZED_IMG_WIDTH = 576
RESIZED_IMG_HEIGHT = 384
GRID_WIDTH = 576 // 32
GRID_HEIGHT = 384 // 32

# Scaling the anchors 
SCALED_ANCHORS = torch.tensor(ANCHORS) * torch.tensor([GRID_WIDTH, GRID_HEIGHT])
SCALED_ANCHORS = SCALED_ANCHORS.to(DEVICE)

# Class labels 
CLASS_NAMES = [ 
    "Jelly_White","Jelly_Milk","Jelly_Black","Amandina","Creme_brulee",
    "Triangolo","Tentation_noir","Comtesse","Noblesse","Noir_authentique",
    "Passion_au_lait","Arabia","Stracciatella"
]

CLASS_NAMES_CSV = [ 
    "Jelly White","Jelly Milk","Jelly Black","Amandina","Crème brulée",
    "Triangolo","Tentation noir","Comtesse","Noblesse","Noir authentique",
    "Passion au lait","Arabia","Stracciatella"
]

# Setting number of classes 
NUM_CLASSES = len(CLASS_NAMES)

test_dir = "../dataset_project_iapr2025/test"
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".JPG")])

# -----------Functions-------------

def reset_val_dir():
    """
    Clear content of validation images from previous runs.
    
    Args:
        None : it's always the same folder  
    
    Returns:
        None
    """

    val_image_dir = "val_images"
    if os.path.exists(val_image_dir):
        shutil.rmtree(val_image_dir)
    os.makedirs(val_image_dir)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class ChocolateCounterNetDeep(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 192x288
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)  # 96x144
        )

        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2)   # 48x72
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)  # 24x36
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)  # 12x18
        self.layer4 = self._make_layer(512, 1024, num_blocks=2, stride=2) # 6x9

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return self.fc(x)
    

class ChocolateCountDataset(Dataset):
    def __init__(self, image_filenames, image_dir, label_dir, transform=None, augment=False, n_augmentations=3):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.n_augmentations = n_augmentations
        self.image_filenames = image_filenames

        self.base_transform = transforms.Compose([
            transforms.Resize((384, 576)),
            transforms.ToTensor()
        ])

        self.augment_transform = transform or transforms.Compose([
            transforms.Resize((384, 576)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

        if self.augment:
            self.samples = []
            for fname in self.image_filenames:
                img_path = os.path.join(self.image_dir, fname)
                label_path = os.path.join(self.label_dir, os.path.splitext(fname)[0] + ".txt")

                label_data = np.loadtxt(label_path, ndmin=2)
                classes = label_data[:, 0].astype(int)
                class_counts = np.bincount(classes, minlength=NUM_CLASSES)

                image = Image.open(img_path).convert("RGB")
                base_image = self.base_transform(image)
                self.samples.append((base_image, torch.tensor(class_counts, dtype=torch.float32)))

                for _ in range(n_augmentations):
                    augmented_image = self.augment_transform(image)
                    self.samples.append((augmented_image, torch.tensor(class_counts, dtype=torch.float32)))
        else:
            self.samples = self.image_filenames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.augment:
            return self.samples[idx]
        else:
            img_name = self.samples[idx]
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

            image = Image.open(img_path).convert("RGB")
            image = self.base_transform(image)

            label_data = np.loadtxt(label_path, ndmin=2)
            classes = label_data[:, 0].astype(int)
            class_counts = np.bincount(classes, minlength=NUM_CLASSES)

            return image, torch.tensor(class_counts, dtype=torch.float32)
