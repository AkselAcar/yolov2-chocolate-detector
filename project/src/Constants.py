import torch
import os

# Device 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
LOAD_MODEL = True
SAVE_MODEL = True

# model checkpoint file name to load
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "Model_checkpoints")
CHECKPOINT_FILE_TO_LOAD = os.path.join(MODEL_CHECKPOINTS_DIR, "checkpoint_2.pth.tar")

# Anchor boxes scaled between 0 and 1
ANCHORS = [
	(0.06327, 0.11604), (0.07833, 0.09567), (0.08240, 0.12298),
	(0.08851, 0.13328), (0.09456, 0.14135), (0.06968, 0.21130),
	(0.14110, 0.10559), (0.10143, 0.15234), (0.11200, 0.16326)]

NUM_ANCHORS = len(ANCHORS)

# Batch size for training 
BATCH_SIZE = 5

# Learning rate for training 
LEARNING_RATE = 1e-3

# Number of epochs for training 
NB_EPOCHS = 1

# Image sizes
ORIG_IMG_WIDTH = 6000
ORIG_IMG_HEIGHT = 4000
RESIZED_IMG_WIDTH = 576
RESIZED_IMG_HEIGHT = 384
STRIDE_TOT = 16
GRID_WIDTH = 576 // STRIDE_TOT
GRID_HEIGHT = 384 // STRIDE_TOT

# Scaling the anchors 
SCALED_ANCHORS = torch.tensor(ANCHORS) * torch.tensor([GRID_WIDTH, GRID_HEIGHT])
SCALED_ANCHORS = SCALED_ANCHORS.to(DEVICE)

# Class labels 
CLASS_NAMES = [ 
	"Jelly_White","Jelly_Milk","Jelly_Black","Amandina","Creme_brulee",
	"Triangolo","Tentation_noir","Comtesse","Noblesse","Noir_authentique",
	"Passion_au_lait","Arabia","Stracciatella"
]

# Setting number of classes 
NUM_CLASSES = len(CLASS_NAMES)