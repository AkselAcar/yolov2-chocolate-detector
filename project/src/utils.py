import torch 
import torch.nn as nn 
import torch.optim as optim 

from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2 

import os 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

from tqdm import tqdm

import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.ops import nms as torch_nms

import csv
from collections import defaultdict

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
LEARNING_RATE = 1e-3

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

def iou(box1, box2, is_pred=True):
    """
    Function to calculate Intersection over Union (IoU).

    Inputs :    
        box1 : tensor of shape ([...,X,2]) (is_pred=False) or ([...,4]) (is_pred=True)
        box2 : tensor of shape([...,Y,2]) (is_pred=False) or ([...,4]) (is_pred=True)
        is_pred : bool

    Output :
        iou_score : tensor of shape ([...,Y,X]) (is_pred=False) or ([...,1]) (is_pred=True)
    """
    
    if is_pred: 
        # IoU score for prediction and label
        
        # Box coordinates of prediction 
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth 
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle 
        x1 = torch.max(b1_x1, b2_x1) 
        y1 = torch.max(b1_y1, b2_y1) 
        x2 = torch.min(b1_x2, b2_x2) 
        y2 = torch.min(b1_y2, b2_y2) 
        # Make sure the intersection is at least 0 
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 

        # Calculate the union area 
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
        union = box1_area + box2_area - intersection 

        # Calculate the IoU score 
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score 
        return iou_score 
    
    else: # this case is used in the dataset class to calculate IoU of anchor boxes with ground truth boxes 
          # (so don't need x, y coordinates), we just want to know what is the best shape of anchor for the ground truth of one cell

        # Calculate intersection area 
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 

        # Calculate union area 
        box1_area = box1[..., 0] * box1[..., 1] 
        box2_area = box2[..., 0] * box2[..., 1] 
        union_area = box1_area + box2_area - intersection_area 

        # Calculate IoU score 
        iou_score = intersection_area / union_area 

        # Return IoU score 
        return iou_score
    
def convert_cell_to_bboxes(prediction, scaled_anchors=SCALED_ANCHORS, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT, num_classes=NUM_CLASSES, conf_threshold = 0.6):
    
    """
    Function to convert a prediction (output) of the model to a 
    Input :
        prediction : tensor of shape ([num_anchors, grid_height, grid_width, 5 + num_classes]) with 5 for (score, x, y, w, h) in different scales (raw output)
        scaled_anchors : tensor of shape ([num_anchors, 2])
        other parameters : int

    Output : 
        boxes : tensor of shape (num_anchors x grid_height x grid_width, 6) with only the boxes whose confidence level is above conf_threshold,
                with 6 for (x, y, w, h, confidence, class_idx),
                with x, y, w, h in pixels with respect to the resized image (576x384).
    """

    A = prediction.size(0)
    device = prediction.device
    
    # Extract objectness score (confidence about having an object in the cell), shape = (num_anchors, grid_height, grid_width)
    objectness = torch.sigmoid(prediction[..., 0])  # the sigmoid is in the loss (not at the end of the model) so we need it here	
    
    # Extract box coordinates
    pred_xy = torch.sigmoid(prediction[..., 1:3]) # coordinates of the center in each grid cell, shape = (num_anchors, grid_height, grid_width, 2)

    scaled_anchors = scaled_anchors.to(device) # to ensure everything is on GPU
    pred_wh = torch.exp(prediction[..., 3:5]) * scaled_anchors.view(A, 1, 1, 2) # Same conversion as in the loss to find w and h with respect to the grid size
    
    # Generate grid indices [0, 1, ..., grid_size-1]. i corresponds to the rows and j to the columns
    grid_j, grid_i = torch.meshgrid(
        torch.arange(grid_width, device=device), 
        torch.arange(grid_height, device=device),
        indexing="xy"  # Align with image coordinates (i=row, j=column)
    )
    # Add anchor and channel dimensions: (1, S, S, 1) → (A, S, S, 1)
    cell_x = grid_j.unsqueeze(0).unsqueeze(-1).expand(A, -1, -1, -1) # shape = (num_anchors, grid_height, grid_width, 1), in each cell there is an integer between 0 and grid_width-1
    cell_y = grid_i.unsqueeze(0).unsqueeze(-1).expand(A, -1, -1, -1) # shape = (num_anchors, grid_height, grid_width, 1), in each cell there is an integer between 0 and grid_height-1
    
    # Compute x and y in absolute pixels (on a resized_img_size)
    x = (pred_xy[..., 0:1] + cell_x) * (resized_img_width / grid_width) # shape = (num_anchors, grid_height, grid_width, 1), each cell contains the center of the box of the cell in pixels between 0 and resized_img_width-1
    y = (pred_xy[..., 1:2] + cell_y) * (resized_img_height / grid_height) # shape = (num_anchors, grid_height, grid_width, 1), each cell contains the center of the box of the cell in pixels between 0 and resized_img_height-1
    
    # Convert w and h in pixels (on a resized_img_size)
    w = pred_wh[..., 0:1] * (resized_img_width / grid_width) # shape = (num_anchors, grid_height, grid_width, 1), each cell contains the width of the box of the cell in pixels between 0 and resized_img_width-1
    h = pred_wh[..., 1:2] * (resized_img_height / grid_height) # shape = (num_anchors, grid_height, grid_width, 1), each cell contains the height of the box of the cell in pixels between 0 and resized_img_width-1
    
    # Get class scores from the one hot encoding vector (ideally, because it is not in reality)
    class_logits = prediction[..., 5 : 5 + num_classes]  # shape = (num_anchors, grid_height, grid_width, num_classes)
    class_probs = torch.softmax(class_logits, dim=-1) # the output of the model is unnormalized, the cross-entropy in the loss does the softmax so we need it here also
    class_conf, class_idx = torch.max(class_probs, dim=-1, keepdim=True)  # shapes = (num_anchors, grid_height, grid_width, 1), class_conf is an integer in [0,1], class_idx is an integer between [0, num_classes-1]

    # Combine all information and reshape
    confidence = objectness.unsqueeze(-1) * class_conf # shape = (num_anchors, grid_height, grid_width, 1), each cell contains a proba between 0 and 1 of having an object OF THE RIGHT CLASS (objectness was just a proba of having an object)
    boxes = torch.cat([x, y, w, h, confidence, class_idx.float()], dim=-1)
    boxes = boxes.reshape(-1, 6) # shape = (num_anchors x grid_height x grid_width, 6)
    print("confidence max/min = ", confidence.max(), confidence.min())
    
    # Filter out boxes with low confidence
    mask = boxes[:, 4] >= conf_threshold
    boxes = boxes[mask]
    return boxes

def plot_image(image, boxes, class_names=CLASS_NAMES, save_dir=None, filename="image.png"):
    """
    Function to plot/save an image with its predicted bounding boxes (already above a confidence threshold) and class labels.
    
    Inputs:
        image: Image as numpy array (H, W, C) in [0, 255]
        boxes: List of bounding boxes [class, confidence, x, y, width, height] with x,y,w,h between 0 and 1, class between 0 and num_classes-1
        class_names: list of the class_names
        save_dir: Directory to save the image
        filename: Filename to save the image
    """
    
    # Create figure and axes 
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    # Convert image to uint8 if needed
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Display image
    ax.imshow(image)

    # Get color map
    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_names))]

    if boxes is not None and len(boxes) > 0:
        # Process each box
        for box in boxes:
            class_pred = int(box[0])
            confidence = box[1]

            # Convert normalized coordinates to pixel coordinates
            h, w = image.shape[:2]
            x_center = box[2] * w
            y_center = box[3] * h
            width = box[4] * w
            height = box[5] * h
            
            # Calculate rectangle boundaries with clamping
            x1 = max(0, min(x_center - width/2, w - 1))
            y1 = max(0, min(y_center - height/2, h - 1))
            x2 = max(0, min(x_center + width/2, w - 1))
            y2 = max(0, min(y_center + height/2, h - 1))
            
            # Skip invalid boxes
            #if x2 <= x1 or y2 <= y1:
            #	continue

            # Create rectangle with visible parameters
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,  # Increased line width
                edgecolor=colors[class_pred],
                facecolor="none",
                alpha=0.9  # Add transparency if needed
            )
            ax.add_patch(rect)

            # Add text label with background
            text = f"{class_names[class_pred]}: {confidence:.2f}"
            text_x = x1
            text_y = max(y1 - 3, 10)  # Keep text within image bounds
            
            ax.text(
                text_x,
                text_y,
                text,
                color="white",
                fontsize=12,
                bbox={
                    "facecolor": colors[class_pred],
                    "alpha": 0.8,
                    "pad": 2,
                    "edgecolor": "none"
                }
            )
    else:
        ax.text(10, 20, "No detection", color='yellow', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.5))
    # Save or display
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, filename),
            bbox_inches="tight",
            pad_inches=0,
            dpi=100
        )
        plt.close()
    else:
        plt.show()
    plt.close()
    
def save_val_images(model, loader, device, scaled_anchors, save_dir, epoch, class_names = CLASS_NAMES, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT):
    """
    Function to save to apply the model on images of a loader, and plot/save them by calling the function plot_image
    """
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            outputs = model(x) # shape = (batch_size, num_anchors, grid_height, grid_width, 5 + num_classes)

            for img_idx in range(x.size(0)): # iterates over the current batch
                pred = outputs[img_idx]

                # Convert predictions to tensor of boxes [num_anchors*grid_height*grid_width, 6] and filter all the low confidences
                boxes = convert_cell_to_bboxes(pred, scaled_anchors, grid_width, grid_height)
                final_boxes = []

                if boxes.size(0) > 0:
                # Split into components
                    coords = boxes[:, :4]      # [x_center, y_center, width, height] in absolute pixels with respect to the resized image
                    scores = boxes[:, 4]       # confidences
                    labels = boxes[:, 5].int() # class indices
            
                    # Convert from center to corner coordinates [x1, y1, x2, y2]
                    boxes_corners = torch.zeros_like(coords)
                    boxes_corners[:, 0] = coords[:, 0] - coords[:, 2]/2
                    boxes_corners[:, 1] = coords[:, 1] - coords[:, 3]/2
                    boxes_corners[:, 2] = coords[:, 0] + coords[:, 2]/2
                    boxes_corners[:, 3] = coords[:, 1] + coords[:, 3]/2
            
                    # torch_nms removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box. It expects the format [x1,y1,x2,y2]
                    keep = torch_nms(boxes_corners, scores, iou_threshold=0.4)
                    boxes_nms = boxes_corners[keep]
                    scores_nms = scores[keep]
                    labels_nms = labels[keep]
            
                    # Convert to [class, conf, x_center, y_center, width, height] (and normalized to [0,1]) format before ploting it
                    for box, score, label in zip(boxes_nms, scores_nms, labels_nms):
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        x_center = box[0] + width / 2
                        y_center = box[1] + height / 2
                        final_boxes.append([
                        label.item(), 
                        score.item(),
                        x_center.item() / resized_img_width,  # Normalize between 0 and 1
                        y_center.item() / resized_img_height,
                        width.item() / resized_img_width,
                        height.item() / resized_img_height
                        ])
                
                img = x[img_idx].permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                # plot_image expects boxes in the format [class, confidence, x, y, width, height] with conf,x,y,w,h between 0 and 1, and class between 0 and 12
                plot_image(img, final_boxes, class_names=CLASS_NAMES, save_dir=save_dir, filename=f"epoch_{epoch}_img_{batch_idx + img_idx}.png")

######################################## SAVE CHECKPOINTS ########################################
# Function to save checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
    print("==> Saving checkpoint") 
    checkpoint = { 
        "state_dict": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
    } 
    torch.save(checkpoint, filename)

######################################## LOAD CHECKPOINTS ########################################
# Function to load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
    print("==> Loading checkpoint") 
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE) 
    model.load_state_dict(checkpoint["state_dict"]) 
    optimizer.load_state_dict(checkpoint["optimizer"]) 

    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 

############################################################ DATASET AND DATA TRANSFORM ############################################################

# Data transform (to update)
class YoloCompatibleTransform:
    def __init__(self, resized_img_width=576, resized_img_height=384, # WAS AT 640X640 BEFORE !!!!!!!!
                 train=True, max_rotation=30, max_translation=0.1,
                 brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1,
                 blur_prob=0.5, blur_radius_max=0.5, noise_std=0.05):

        self.resized_img_width = resized_img_width
        self.resized_img_height = resized_img_height
        self.train = train
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_prob = blur_prob
        self.blur_radius_max = blur_radius_max
        self.noise_std = noise_std

        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, bboxes):
        image = Image.fromarray(image)

        # Resize image
        orig_width, orig_height = image.size
        scale = min(self.resized_img_width / orig_width, self.resized_img_height / orig_height) # = 0.096
        new_w, new_h = int(orig_width * scale), int(orig_height * scale)
        image = image.resize((self.resized_img_width, self.resized_img_height))
        # Calculate center position for pasting the resized image
        paste_x = (self.resized_img_width - new_w) // 2
        paste_y = (self.resized_img_height - new_h) // 2

        angle = 0
        dx, dy = 0,0
        if self.train:
            # Random rotation
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            image = F.rotate(image, angle)

            # Rotate bboxes (approximate using affine math)
            angle_rad = -np.radians(angle)
            cx, cy = self.resized_img_width / 2, self.resized_img_height / 2

            # Random translation
            max_dx = self.max_translation * self.resized_img_width
            max_dy = self.max_translation * self.resized_img_height
            dx = np.random.uniform(-max_dx, max_dx)
            dy = np.random.uniform(-max_dy, max_dy)

            image = F.affine(image, angle=0, translate=(dx, dy), scale=1.0, shear=0)

            # Color jitter
            image = self.color_jitter(image)

            # Gaussian blur
            if np.random.random() < self.blur_prob:
                radius = np.random.uniform(0, self.blur_radius_max)
                image = image.filter(ImageFilter.GaussianBlur(radius))

            # Add noise
            image_np = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.noise_std, image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0.0, 1.0)
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            # Convert to tensor for non-augmented data
            image = self.to_tensor(image)
            
        transformed_bboxes = []
        
        # Calculate transformation matrix for bounding boxes
        # For simplicity, we'll apply transformations in reverse order:
        # 1. Scale and center (always applied)
        # 2. Flip (if applicable)
        # 3. Rotate and translate (if applicable)
        
        for box in bboxes:
            x_center, y_center, w, h, cls = box
            
            # Original image coordinates
            x1 = (x_center - w/2) * orig_width
            y1 = (y_center - h/2) * orig_height
            x2 = (x_center + w/2) * orig_width
            y2 = (y_center + h/2) * orig_height
            
            # Step 1: Scale coordinates to resized image
            x1 = x1 * scale
            y1 = y1 * scale
            x2 = x2 * scale
            y2 = y2 * scale
            
            # Step 2: Add padding to center in the image
            x1 += paste_x
            y1 += paste_y
            x2 += paste_x
            y2 += paste_y
            
            # Step 4: Apply rotation and translation
            if angle != 0 or (dx, dy) != (0, 0):
                # Convert box to 4 corner points
                points = [
                    (x1, y1),  # top-left
                    (x2, y1),  # top-right
                    (x2, y2),  # bottom-right
                    (x1, y2),  # bottom-left
                ]
                
                # Apply transformation to each point
                transformed_points = []
                for px, py in points:
                    # Center for rotation
                    cx, cy = self.resized_img_width / 2, self.resized_img_height / 2
                    
                    # Translate point to origin
                    px -= cx
                    py -= cy
                    
                    # Apply rotation (convert angle to radians)
                    angle_rad = -angle * np.pi / 180.0  # Negative for clockwise rotation
                    px_rot = px * np.cos(angle_rad) - py * np.sin(angle_rad)
                    py_rot = px * np.sin(angle_rad) + py * np.cos(angle_rad)
                    
                    # Translate back and apply translation
                    px = px_rot + cx + dx
                    py = py_rot + cy + dy
                    
                    transformed_points.append((px, py))
                
                # Get axis-aligned bounding box (AABB) of transformed points
                # This ensures the box remains aligned with horizontal/vertical axes
                x_coords = [p[0] for p in transformed_points]
                y_coords = [p[1] for p in transformed_points]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
            
            # Ensure coordinates are within image bounds and calculate intersection with image
            x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
            
            # Clip coordinates to image boundaries
            x1 = np.clip(x1, 0, self.resized_img_width)
            y1 = np.clip(y1, 0, self.resized_img_height)
            x2 = np.clip(x2, 0, self.resized_img_width)
            y2 = np.clip(y2, 0, self.resized_img_height)
            
            # Calculate how much of the box is still inside the image
            original_area = (x2_orig - x1_orig) * (y2_orig - y1_orig) # area of the box potentially outside
            if original_area <= 0:  # Avoid division by zero
                continue
                
            intersection_area = (x2 - x1) * (y2 - y1)
            visibility_ratio = intersection_area / original_area
            
            # If less than 25% of the box is visible, we skip the box
            if visibility_ratio < 0.25:
                continue
            
            # Convert back to YOLO format (x_center, y_center, width, height) normalized
            w = (x2 - x1) / self.resized_img_width
            h = (y2 - y1) / self.resized_img_height
            x_center = (x1 + x2) / 2 / self.resized_img_width
            y_center = (y1 + y2) / 2 / self.resized_img_height
            
            # Add transformed box if it's still visible
            if visibility_ratio >= 0.4:  # Minimum size threshold
                transformed_bboxes.append([x_center, y_center, w, h, cls])
        
        # Convert to tensor and normalize
        # image_tensor = self.to_tensor(image)
        
        return {"image": image, "bboxes": transformed_bboxes}
    
# Create a dataset class to load the images and labels from the folder 
class YoloDetectionAugmentedDataset(torch.utils.data.Dataset):
	def __init__(self, image_filenames, image_dir, label_dir, anchors,
				 transform, num_augmentations=2,
				 resized_img_width=576, resized_img_height=384,
				 grid_width=18, grid_height=12,
				 num_classes=13, ignore_iou_thresh=0.8):
		"""
		Create a fixed dataset with N augmentations per image.

		Args:
			image_filenames: list of image file names
			image_dir: path to directory with images
			label_dir: path to directory with YOLO-format labels (.txt)
			anchors: list of anchor box sizes
			transform: YoloCompatibleTransform (with train=True)
			num_augmentations: number of augmented copies per image
		"""
		self.anchors = torch.tensor(anchors)
		self.num_anchors = len(anchors)
		self.num_classes = num_classes
		self.grid_width = grid_width
		self.grid_height = grid_height
		self.ignore_iou_thresh = ignore_iou_thresh

		self.original_transform = YoloCompatibleTransform(
			resized_img_width=resized_img_width,
			resized_img_height=resized_img_height,
			train=False
		)
		self.augmenting_transform = transform
		self.data = []

		print("Preprocessing images with fixed augmentations...")
		for filename in tqdm(image_filenames):
			label_filename = os.path.splitext(filename)[0] + ".txt"
			label_path = os.path.join(label_dir, label_filename)
			bboxes = np.roll(np.loadtxt(label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

			img_path = os.path.join(image_dir, filename)
			image = np.array(Image.open(img_path).convert("RGB"))

			transforms_all = [self.original_transform] + [self.augmenting_transform] * num_augmentations
			for tf in transforms_all:
				transformed = tf(image, bboxes)
				image_tensor = transformed["image"]
				transformed_bboxes = transformed["bboxes"]
				target_tensor = self.encode_targets(transformed_bboxes)
				self.data.append((image_tensor, target_tensor))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	def encode_targets(self, bboxes):
		"""
		Convert bounding boxes into YOLO-style target tensor.
		Format: (num_anchors, grid_height, grid_width, 6)
		"""
		target = torch.zeros((self.num_anchors, self.grid_height, self.grid_width, 6))
		for box in bboxes:
			iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
			anchor_indices = iou_anchors.argsort(descending=True)

			x, y, w, h, cls = box
			has_anchor = False
			for anchor_idx in anchor_indices:
				i = min(int(self.grid_height * y), self.grid_height - 1)
				j = min(int(self.grid_width * x), self.grid_width - 1)

				if not has_anchor:
					target[anchor_idx, i, j, 0] = 1
					x_cell = self.grid_width * x - j
					y_cell = self.grid_height * y - i
					w_cell = w * self.grid_width
					h_cell = h * self.grid_height
					target[anchor_idx, i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
					target[anchor_idx, i, j, 5] = int(cls)
					has_anchor = True
				elif iou_anchors[anchor_idx] > self.ignore_iou_thresh:
					target[anchor_idx, i, j, 0] = -1
		return target
    
######################################## MODEL ############################################################

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_ratio=0.25, expansion_ratio=1.0, use_residual=True):
        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_ratio)
        expanded_channels = int(in_channels * expansion_ratio)
        
        self.conv1 = CNNBlock(in_channels, bottleneck_channels, kernel_size=1)
        self.conv2 = CNNBlock(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv3 = CNNBlock(bottleneck_channels, expanded_channels, kernel_size=1)
        
        self.proj = nn.Identity() if in_channels == expanded_channels else \
                   CNNBlock(in_channels, expanded_channels, kernel_size=1)
                   
        self.dropout = nn.Dropout(0.1)
        self.use_residual = use_residual
        
    def forward(self, x):
        residual = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        
        if self.use_residual:
            x = x + residual
            
        return x
        
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, (num_classes + 5) * num_anchors, kernel_size=1),
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        B, _, H, W = x.shape
        output = self.pred(x)
        output = output.view(B, self.num_anchors, self.num_classes + 5, H, W)
        output = output.permute(0, 1, 3, 4, 2)
        return output
        
class EfficientYOLO_12x18Grid(nn.Module):
    """
    Input: (B, 3, 384, 576)
    Output: (B, NUM_ANCHORS, 12, 18, 5 + num_classes)
    """
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(0.2)
        
        self.backbone = nn.Sequential(
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            BottleneckBlock(64, bottleneck_ratio=0.25),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            BottleneckBlock(128, bottleneck_ratio=0.25),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            BottleneckBlock(256, bottleneck_ratio=0.25),
            CNNBlock(256, 384, kernel_size=3, stride=2, padding=1),
            BottleneckBlock(384, bottleneck_ratio=0.25),
            CNNBlock(384, 512, kernel_size=3, stride=2, padding=1),
            BottleneckBlock(512, bottleneck_ratio=0.25),
        )

        self.spatial_path = nn.Sequential(
            CNNBlock(512, 256, kernel_size=3, padding=1),
            CNNBlock(256, 256, kernel_size=3, padding=1),
        )

        self.context_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CNNBlock(512, 256, kernel_size=1),
        )

        self.head = nn.Sequential(
            CNNBlock(512, 384, kernel_size=1),
            CNNBlock(384, 512, kernel_size=3, padding=1),
            CNNBlock(512, 256, kernel_size=1),
        )

        self.scale_pred = nn.Sequential(
            CNNBlock(256, 256, kernel_size=3, padding=1),
            ScalePrediction(256, num_classes, NUM_ANCHORS)  # Must return shape (B, A, 12, 18, 5+C)
        )

    def forward(self, x):
        features = self.backbone(x)
        spatial_features = self.spatial_path(features)
        context_features = self.context_path(features)
        context_features = context_features.expand(-1, -1, spatial_features.size(2), spatial_features.size(3))
        fused_features = torch.cat([spatial_features, context_features], dim=1)
        detection_features = self.head(fused_features)
        dropped = self.drop(detection_features)
        output = self.scale_pred(dropped)
        return output

def init_yolo_weights(model):
    """Initialize weights for YOLO model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming for Conv layers followed by BatchNorm
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            # BatchNorm: gamma=1, beta=0
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    # Special initialization for final prediction layer
    for m in model.modules():
        if isinstance(m, ScalePrediction):
            last_conv = m.pred[-1]  # Final Conv2d in ScalePrediction
            if isinstance(last_conv, nn.Conv2d):
                # Xavier for weight
                nn.init.xavier_normal_(last_conv.weight)
                # Initialize bias for objectness to low confidence
                if last_conv.bias is not None:
                    num_anchors = NUM_ANCHORS
                    num_classes = model.num_classes
                    bias = last_conv.bias.view(num_anchors, 5 + num_classes)
                    bias.data[:, 0] = -torch.log(torch.tensor(99.0))  # sigmoid(0) ≈ 0.01
                    bias.data[:, 5:] = 0  # Class logits initialized to 0
                    last_conv.bias.data = bias.view(-1)
    return model


########################################## LOSS FUNCTION ############################################################

# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.mse = nn.MSELoss()                    # for continuous values (x, y, w, h)
        self.bce = nn.BCEWithLogitsLoss()          # applies sigmoid on the prediction and then cross-entropy => for objectness (with logits),
                                                   # enables to remove false positives because BCE penalizes much more a big prediction for a low ground truth than the other way.
        self.cross_entropy = nn.CrossEntropyLoss() # penalizes if the logit of the correct class is too low. It makes the prediction converge to a one hot encoding vector AFTER BEING PASSED IN A SOFTMAX.
        self.sigmoid = nn.Sigmoid()                # forces data between 0 and 1
        
        self.lambda_box = 1.0 # weight for box coordinates loss
        self.lambda_object = 2.0 # weight for object loss
        self.lambda_no_object = 2.0 # weight for no object loss
        self.lambda_class = 1.0 # weight for class loss
    
    def forward(self, pred, target, scaled_anchors=SCALED_ANCHORS):
        """
        Inputs : 
            pred : output of the model, tensor of shape (batch_size, num_anchors, grid_height, grid_width, 5 + num_classes)), raw logits, unnormalized
            target : target from __getitem__ of the dataloader, tensor of shape (batch_size, num_anchors, grid_height, grid_width, 6) 
                     with 6 for (objectness, x, y, w, h, class_label), (w,y,w,h) between 0 and 1, (x,y) relatives to the cell, (w,h) relatives to the grid size, class label between 0 and num_classes-1
            scaled_anchors : tensor of shape (num_anchors, 2), with the anchors normalized relatives to the GRID SIZE (so x between 0 and 18, y between 0 and 12)
        Output :
            total loss of YOLO network (scalar)
        """
        
        # Identifying which cells in target have objects (and what is the best anchor, the one to detect)
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Reshaping anchors to match predictions. pred.size(1) = num_anchors ==> scaled_anchors.shape becomes (., 9, ., ., 2) which is compatible with pred[..., 3:5].shape to multiply them below
        scaled_anchors = scaled_anchors.reshape(1, pred.size(1), 1, 1, 2).to(pred.device) 

        ###################################################### Calculating the BOX LOSS #####################################################        
        # Predicted box coordinates 
        pred_copy_for_box_loss = pred.clone()
        pred_copy_for_box_loss[..., 1:3] = self.sigmoid(pred_copy_for_box_loss[..., 1:3])
        # Target box coordinates
        target_copy_for_box_loss = target.clone()
        target_copy_for_box_loss[..., 3:5] = torch.log(1e-6 + target_copy_for_box_loss[..., 3:5] / scaled_anchors)
        # Calculating box coordinate loss 
        box_loss = self.mse(pred_copy_for_box_loss[..., 1:5][obj], target_copy_for_box_loss[..., 1:5][obj])
        # Goal : (x,y) post-sigmoid are gonna be close to (x,y) of the target,
        # and (w,h) post-processed (so after exponential and multiplication with the scaled_anchor) are gonna be close to the targeted scaled_anchor (the closest to the real target, that has been defined in the dataset __getitem__)

        ################################################### Calculating the NO OBJECT LOSS ###################################################
        # Applies, for the cells with no object, the binary cross entropy (bce) between the predicted score (raw logit) and the label (0)
        # Makes the model learn to put a small value in the score where there is no object (so sigmoid(score) is close to zero)
        no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

        ##################################################### Calculating the OBJECT LOSS ####################################################
        
        # Clone not to modify the original with sigmoid and exponential because that could cause problems
        pred_copy_for_object_loss = pred.clone()
        pred_xy_01_for_object_loss = self.sigmoid(pred_copy_for_object_loss[..., 1:3])
        pred_wh_ratio_for_object_loss = torch.exp(pred_copy_for_object_loss[..., 3:5]) * scaled_anchors
        # the model predicts w (and h) such that when we do exp(w) (or h) * the predicted scaled_anchors (thanks to the mask "obj"), we get the width (and height) of the box to plot, relative to the grid size (hopefully close to the (w,h) of the targeted scaled_anchor)
        box_preds_for_object_loss = torch.cat([pred_xy_01_for_object_loss, pred_wh_ratio_for_object_loss],dim=-1)

        # Calculating intersection over union of the prediction processed and the target 
        ious = iou(box_preds_for_object_loss[obj], target[..., 1:5][obj]).detach()
        pred_score_for_object_loss = self.sigmoid(pred_copy_for_object_loss[..., 0:1]) # pred[..., 0:1] = score (logit) of the object
        
        object_loss = self.mse(pred_score_for_object_loss[obj], ious * target[...,0:1][obj])
        # Goal : where there is an object, the predicted score after sigmoid has to be close to the IoU of the predicted box (post processed) and the target (the best anchor). At the end it should be one.

        ##################################################### Calculating the CLASS LOSS ####################################################  
        # Claculating class loss 
        class_loss = self.cross_entropy((pred[..., 5:][obj]), 
                                         target[..., 5][obj].long()) 

        # print("box_loss = ", box_loss)
        # print("object_loss =", object_loss)
        # print("no object_loss =", no_object_loss)
        # print("class_loss =", class_loss)
    
        # Total loss 
        return ( 
            self.lambda_box * box_loss 
            + self.lambda_object * object_loss 
            + self.lambda_no_object * no_object_loss 
            + self.lambda_class * class_loss 
        )
    
########################################## CSV SUBMISSION ############################################################

def generate_submission_csv_with_nms(
    model, 
    loader, 
    output_csv_path="submission.csv",
    conf_threshold=0.6,
    iou_threshold=0.4,
    resized_img_width=RESIZED_IMG_WIDTH,
    resized_img_height=RESIZED_IMG_HEIGHT
):
    """Generate Kaggle submission CSV with proper NMS handling"""
    
    model.eval()
    predictions = []

    for img_tensor, fname in tqdm(loader, desc="Generating submission"):
        img_tensor = img_tensor.to(DEVICE)
        
        with torch.no_grad():
            pred = model(img_tensor)

        # Process each image in the batch
        for i in range(img_tensor.size(0)):
            # Squeeze to get single image prediction
            single_pred = pred[i]

            # Convert predictions to bounding boxes
            bboxes = convert_cell_to_bboxes(
                single_pred, 
                scaled_anchors=SCALED_ANCHORS,
                grid_width=GRID_WIDTH,
                grid_height=GRID_HEIGHT,
                resized_img_width=resized_img_width,
                resized_img_height=resized_img_height,
                num_classes=NUM_CLASSES,
                conf_threshold=conf_threshold
            )

            # Process boxes with class-aware NMS
            final_counts = defaultdict(int)
            if len(bboxes) > 0:
                # Ensure bboxes are on CPU for processing
                bboxes = bboxes.cpu()

                # Convert to [x1, y1, x2, y2] format for NMS
                boxes_corners = torch.zeros((len(bboxes), 4))
                boxes_corners[:, 0] = bboxes[:, 0] - bboxes[:, 2]/2  # x1
                boxes_corners[:, 1] = bboxes[:, 1] - bboxes[:, 3]/2  # y1
                boxes_corners[:, 2] = bboxes[:, 0] + bboxes[:, 2]/2  # x2
                boxes_corners[:, 3] = bboxes[:, 1] + bboxes[:, 3]/2  # y2
                
                scores = bboxes[:, 4]
                class_ids = bboxes[:, 5].int()

                # Class-aware NMS
                for cls_id in torch.unique(class_ids):
                    cls_mask = class_ids == cls_id
                    cls_boxes = boxes_corners[cls_mask]
                    cls_scores = scores[cls_mask]
                    
                    if len(cls_boxes) == 0:
                        continue
                    
                    # Apply NMS for this class
                    keep = torch_nms(cls_boxes, cls_scores, iou_threshold)
                    final_counts[cls_id.item()] = len(keep)

            # Format row
            curr_fname = fname[i]
            image_id = os.path.splitext(curr_fname)[0].replace("L", "").replace("IMG_", "")
            row = [image_id] + [final_counts[i] for i in range(NUM_CLASSES)]
            predictions.append(row)


    # Write to CSV
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + CLASS_NAMES_CSV)
        writer.writerows(predictions)

    print(f"Submission file saved at: {output_csv_path}")

class Dataset(torch.utils.data.Dataset): 
    def __init__( 
        self, df, image_dir, label_dir, anchors, 
        resized_img_width=576, resized_img_height=384,
        grid_width=18, grid_height=12,
        num_classes=13, transform=None, ignore_iou_thresh=0.8
    ): 
        self.label_list = df
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        self.resized_img_width = resized_img_width
        self.resized_img_height = resized_img_height
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.anchors = torch.tensor(anchors)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes 
        self.ignore_iou_thresh = ignore_iou_thresh
        
        # Store transforms
        self.transform = transform
        self.original_transform = None
        if self.transform is not None:
            # Create a non-augmenting transform using the same parameters
            self.original_transform = YoloCompatibleTransform(
                resized_img_width=self.transform.resized_img_width,
                resized_img_height=self.transform.resized_img_height,
                train=False,
                max_rotation=self.transform.max_rotation,
                max_translation=self.transform.max_translation,
                brightness=self.transform.brightness,
                contrast=self.transform.contrast,
                saturation=self.transform.saturation,
                hue=self.transform.hue,
                blur_prob=self.transform.blur_prob,
                blur_radius_max=self.transform.blur_radius_max,
                noise_std=self.transform.noise_std
            )

    def __len__(self): 
        return 2 * len(self.label_list) 

    def __getitem__(self, idx):
        # Getting the label path of the idx
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) #self.label_list.iloc[idx, 1] gets the value of the first column (label_name) in the row "idx"
        # We are applying roll to move class label to the last column => 5 columns: x, y, width, height, class_label 
        bboxes = np.roll(np.loadtxt(fname=label_path, 
                        delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.loadtxt() loads the file label_path (file .txt) containing the YOLO annotations of the corresponding image.
                                                                     # ndmin=2 to ensure that even if there is only one box/row, we obtain a 2D table.
                                                                     # .tolist() makes a list from numpy table.
        # Getting the image path of the idx
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
        image = np.array(Image.open(img_path).convert("RGB")) # open the image and convert to numpy array
        
        # Augmentations : always resize from 6000x4000 to 576x384 (for our case) and apply randomly flips, translations, rotations
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] # image is a tensor of shape (3, 416, 416)
            bboxes = augs["bboxes"] # bboxes is a list of bounding boxes in the format [x, y, width, height, class_label] with x,y,w,h normalized between 0 and 1
        
        
        # Be carefull : first dimension is vertical cells = Y-axis and second dimension is horizontal cells = X-axis
        target = torch.zeros((self.num_anchors, self.grid_height, self.grid_width, 6))
        # targets is a tensor of shape (num_anchors, grid_height, grid_width, 6) and 6 for [probabilities, x, y, width, height, class_label]

        # Identify the cell and the best anchor box of the ground truth for each bounding box
        for box in bboxes: 
            # Calculate iou of each bounding box WITH ANCHOR BOXES
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False) # iou_anchors is a tensor of shape (9,)
                                                      
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # anchor_indices is a tensor of shape (9,)
            x, y, width, height, class_label = box
            # x,y,width,height,class_label are all normalized between 0 and 1 (except class_label between 0 and num_classes-1)

            has_anchor = False
            # Assigning the bounding box to the best matching anchor box
            for anchor_idx in anchor_indices:
                # Identifying the cell to which the bounding box belongs
                # Be carefull : i corresponds to y (rows) and j corresponds to x (columns)
                i = min(int(self.grid_height * y), self.grid_height - 1)
                j = min(int(self.grid_width * x), self.grid_width - 1)
                
                # Check if the anchor box is already assigned
                # anchor_taken = target[anchor_idx, i, j, 0] NOT SURE WE CAN REMOVE ???
                
                if not has_anchor: # basically it is the first iteration # and not anchor taken ????
                    # Set the probability to 1 
                    target[anchor_idx, i, j, 0] = 1

                    # Calculating the center of the bounding box relative to the cell (between 0 and 1)
                    x_cell, y_cell = self.grid_width * x - j, self.grid_height * y - i

                    # Calculating the width and height of the bounding box relative to the grid 
                    width_cell, height_cell = (width * self.grid_width, height * self.grid_height)

                    # Identify the box coordinates 
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # Assigning the box coordinates to the target 
                    target[anchor_idx, i, j, 1:5] = box_coordinates 

                    # Assigning the class label to the target 
                    target[anchor_idx, i, j, 5] = int(class_label) 

                    # Set the anchor box as assigned
                    has_anchor = True

                # If the anchor box is already assigned, check if the IoU is greater than the threshold 
                elif iou_anchors[anchor_idx] > self.ignore_iou_thresh: # and not anchor_taken
                    # Setting this to -1 will enable in the loss not to penalize if the model has found a good anchor but not the best MAYBE REMOVE IT IF NO MODEL CONVERGES ??????
                    target[anchor_idx, i, j, 0] = -1
        
        # print(target.shape)
        
        # Return the image and the target
        return image, target

class YOLOTestDataset(Dataset):
    def __init__(self, img_dir, filenames, transform=None):
        self.img_dir = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        # image = Image.open(img_path).convert("RGB")
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image_dict = self.transform(image, [])
            image = image_dict["image"]
        else:
            image = T.ToTensor()(image)

        #dummy_boxes = torch.zeros((GRID_HEIGHT, GRID_WIDTH, 2 * 5 + NUM_CLASSES))
        return image, fname # dummy_boxes, fname


########################################### TRAINING AND EVALUATION ############################################################

def training_loop(loader, model, optimizer, loss_fn, scaled_anchors=SCALED_ANCHORS, device=DEVICE):
    """
    Function to train the model on one epoch (of the training loader)
    """
    
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True)
    # Initializing a list to store the losses 
    losses = []
    model.train()
    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device)
        y = y.to(device)
        # Getting the model predictions 
        outputs = model(x) # tensor (batch_size, 9, grid_size, grid_size, num_classes + 5)
        # Calculating the loss at each scale 
        loss = loss_fn(outputs, y, scaled_anchors) # scaled_anchors is a tensor of shape (9, 2) (num_anchors, 2)

        # Add the loss to the list
        losses.append(loss.item()) 

        # Reset gradients 
        optimizer.zero_grad() 

        # Backpropagate the loss 
        loss.backward()

        # Optimization step 
        optimizer.step()

        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)

def evaluate(loader, model, loss_fn, scaled_anchors=SCALED_ANCHORS, device=DEVICE):
    """
    Function to evaluate the model on one epoch (of the validation loader)
    """

    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss = loss_fn(outputs, y, scaled_anchors)
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses) if losses else 0
    return mean_loss


############################################ VISUALIZATION OF TRANSFORMS ############################################################

def save_gt_images(loader, class_names=CLASS_NAMES, save_dir="gt_visualizations", grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT):
    """
    Saves images with ground truth bounding boxes from a DataLoader
    
    Args:
        loader: PyTorch DataLoader (train or val)
        class_names: List of class names
        save_dir: Directory to save visualized images
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving ground truth images to {save_dir}/")
    # Color map for different classes
    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_names))]
    
    for batch_idx, (images, targets) in enumerate(tqdm(loader)):
        # Convert images from tensor to numpy and denormalize
        images = torch.stack(images, dim=0)  # Stack the list of tensors into a batch tensor
        images = images.permute(0, 2, 3, 1).cpu().numpy() # shape (batch_size, channels, height, width) to (batch_size, height, width, channels)
        images = (images * 255).astype(np.uint8)
        
        # Process each image in the batch
        for img_idx in range(images.shape[0]):
            img = images[img_idx]
            print(np.array(img).shape)
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            target = targets[img_idx]
            boxes = target["boxes"]
            labels = target["labels"]

            # Draw each box
            for box in boxes:
                x_min, y_min, x_max, y_max = box.tolist()
                print('Coord :', box.tolist())
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none"
                )
                ax.add_patch(rect)

            ax.set_title(f"Image {img_idx}")
            plt.axis("off")
            
            # Save image
            plt.axis("off")
            plt.savefig(
                os.path.join(save_dir, f"img_{batch_idx + img_idx}.png"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=100
            )
            plt.close()


def visualize_yolo_predictions(
    images,
    targets=None,
    predictions=None,
    save_dir=None,
    class_names=None,
    anchors=None,
    resized_img_width=576,
    resized_img_height=384,
    grid_width=18,
    grid_height=12,
    conf_threshold=0.5,
    iou_threshold=0.4,
    max_images=BATCH_SIZE
):
    """
    Visualizes a batch of YOLO images with either targets or predictions.
    
    Args:
        images (Tensor): [B, 3, H, W]
        targets (Tensor, optional): [B, A, S_H, S_W, 6]
        predictions (Tensor, optional): [B, A, S_H, S_W, 5+C] (raw logits)
        save_dir (str, optional): If provided, saves images to this directory
        class_names (List[str]): Class labels
        anchors (Tensor): [A, 2] scaled anchors
        resized_img_*: Image size
        grid_*: Grid size
        conf_threshold (float): Threshold for model predictions
        iou_threshold (float): IOU threshold for NMS
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    B, _, H, W = images.shape
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # B, H, W, C
    if targets is not None:
        targets = targets.cpu()
    if predictions is not None:
        predictions = predictions.cpu()

    for b in range(min(B, max_images)):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        img = (images_np[b] * 255).astype("uint8")
        ax.imshow(img)
        ax.set_title(f"Image {b}")
        ax.axis("off")

        # ---------- Visualize Ground Truth ----------
        if targets is not None:
            A = targets.shape[1]
            for a in range(A):
                for i in range(grid_height):
                    for j in range(grid_width):
                        if targets[b, a, i, j, 0] != 1:
                            continue
                        x_cell, y_cell, w_cell, h_cell = targets[b, a, i, j, 1:5]
                        cls = int(targets[b, a, i, j, 5])
                        x_center = (j + x_cell.item()) * resized_img_width / grid_width
                        y_center = (i + y_cell.item()) * resized_img_height / grid_height
                        w = w_cell.item() * resized_img_width / grid_width
                        h = h_cell.item() * resized_img_height / grid_height
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2

                        rect = patches.Rectangle(
                            (x1, y1), w, h,
                            linewidth=2, edgecolor='lime', facecolor='none'
                        )
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5, f"GT: {class_names[cls]}", color='lime',
                                bbox=dict(facecolor='black', alpha=0.4))

        # ---------- Visualize Predictions ----------
        if predictions is not None and anchors is not None:
            pred = predictions[b]  # (A, S_H, S_W, 5+C)
            A = pred.size(0)
            anchors = anchors.to(pred.device)
            obj_scores = torch.sigmoid(pred[..., 0])
            pred_xy = torch.sigmoid(pred[..., 1:3])
            pred_wh = torch.exp(pred[..., 3:5]) * anchors.view(A, 1, 1, 2)
            class_logits = pred[..., 5:]
            class_probs = torch.softmax(class_logits, dim=-1)
            class_conf, class_idx = torch.max(class_probs, dim=-1)
            confidence = obj_scores * class_conf

            grid_j, grid_i = torch.meshgrid(
                torch.arange(grid_width),
                torch.arange(grid_height),
                indexing="xy"
            )
            cell_x = grid_j.unsqueeze(0).unsqueeze(-1).expand(A, -1, -1, -1)
            cell_y = grid_i.unsqueeze(0).unsqueeze(-1).expand(A, -1, -1, -1)

            x = (pred_xy[..., 0:1] + cell_x) * (resized_img_width / grid_width)
            y = (pred_xy[..., 1:2] + cell_y) * (resized_img_height / grid_height)
            w = pred_wh[..., 0:1] * (resized_img_width / grid_width)
            h = pred_wh[..., 1:2] * (resized_img_height / grid_height)

            boxes = torch.cat([x, y, w, h, confidence.unsqueeze(-1), class_idx.unsqueeze(-1).float()], dim=-1)
            boxes = boxes.reshape(-1, 6)

            mask = boxes[:, 4] >= conf_threshold
            boxes = boxes[mask]

            if boxes.size(0) > 0:
                coords = boxes[:, :4]
                scores = boxes[:, 4]
                labels = boxes[:, 5].int()

                boxes_corners = torch.zeros_like(coords)
                boxes_corners[:, 0] = coords[:, 0] - coords[:, 2]/2
                boxes_corners[:, 1] = coords[:, 1] - coords[:, 3]/2
                boxes_corners[:, 2] = coords[:, 0] + coords[:, 2]/2
                boxes_corners[:, 3] = coords[:, 1] + coords[:, 3]/2

                keep = torch_nms(boxes_corners, scores, iou_threshold)
                boxes_nms = boxes[keep]

                for box in boxes_nms:
                    x, y, w, h, score, cls = box
                    x1 = x - w / 2
                    y1 = y - h / 2
                    rect = patches.Rectangle(
                        (x1, y1), w, h,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, f"PRED: {class_names[int(cls)]} {score:.2f}",
                            color='red', bbox=dict(facecolor='white', alpha=0.6))

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"img_{b}.png"), bbox_inches="tight", pad_inches=0, dpi=100)
            plt.close()
        else:
            plt.show()


def visualize_dataloader_all(
    loader,
    model=None,
    device=DEVICE,
    save_dir="visualizations_all",
    class_names=None,
    anchors=None,
    resized_img_width=576,
    resized_img_height=384,
    grid_width=18,
    grid_height=12,
    conf_threshold=0.5,
    iou_threshold=0.4
):
    """
    Visualizes all images in a DataLoader using ground truth and/or predictions.
    
    Args:
        loader: PyTorch DataLoader
        model: model to generate predictions (optional)
        device: "cuda" or "cpu"
        save_dir: where to save images
        (other args: same as visualize_yolo_predictions)
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.eval() if model else None

    idx = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if model:
            with torch.no_grad():
                preds = model(images)
        else:
            preds = None

        visualize_yolo_predictions(
            images=images.cpu(),
            targets=targets.cpu(),
            predictions=preds.cpu() if preds is not None else None,
            save_dir=save_dir,
            class_names=class_names,
            anchors=anchors,
            resized_img_width=resized_img_width,
            resized_img_height=resized_img_height,
            grid_width=grid_width,
            grid_height=grid_height,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_images=images.size(0)  # visualize entire batch
        )

        idx += images.size(0)