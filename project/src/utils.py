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

import random

import glob


################################################# ALL CONSTANTS ######################################################

# Device 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
LOAD_MODEL = True
SAVE_MODEL = True

# model checkpoint file name 
CHECKPOINT_FILE_LOAD = "checkpoint_150.pth.tar"

# Anchor boxes scaled between 0 and 1
"""
ANCHORS = [                                                               
	(0.14105, 0.10625), (0.11015, 0.16249), (0.09875, 0.14781),
	(0.08998, 0.13511), (0.08254, 0.12333), (0.07731, 0.19987),
	(0.07835, 0.09580), (0.06312, 0.11607), (0.06459, 0.21892)]
"""

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
NB_EPOCHS = 300

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

################################################### HELPER FUNCTIONS ######################################################

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
	for file_path in glob.glob(os.path.join(save_dir, "*.png")):
		os.remove(file_path)
	global_img_idx = 0
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
				plot_image(img, final_boxes, class_names=CLASS_NAMES, save_dir=save_dir, filename=f"epoch_{epoch}_img_{global_img_idx}.png")

				global_img_idx +=1
				

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
		

################################################### DATASET AND DATA AUGMENTATION ######################################################

class YoloCompatibleTransform:
	def __init__(self, resized_img_width = RESIZED_IMG_WIDTH, resized_img_height = RESIZED_IMG_HEIGHT,
				 train=True, max_rotation=10, max_translation=0.05,
				 brightness=0.3, contrast=0.3, saturation=0.1, hue=0.01, # or 0.2 0.2 0.2 0.1
				 noise_prob=0.3, noise_intensity=0.02,
				 blur_prob=0.3, blur_radius=(0.5, 2.5)): 
		
		self.resized_img_width = resized_img_width
		self.resized_img_height = resized_img_height
		self.train = train

		self.max_rotation = max_rotation
		self.max_translation = max_translation
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue

		self.noise_prob = noise_prob
		self.noise_intensity = noise_intensity
		self.blur_prob = blur_prob
		self.blur_radius = blur_radius

		self.to_tensor = transforms.ToTensor()

	def _add_noise(self, image):
		"""Add random Gaussian noise to the image with varying intensity."""
		if random.random() < self.noise_prob:
			# Convert to numpy array first to get correct shape
			img_array = np.array(image)
			
			# Randomly vary noise intensity between 50-150% of base intensity
			intensity_factor = random.uniform(0.5, 1.5)
			current_intensity = self.noise_intensity * intensity_factor
			
			# Generate noise with correct shape (height, width, channels)
			noise = np.random.normal(0, current_intensity * 255, img_array.shape)
			
			# Apply noise
			noise_img = img_array.astype(np.float32) + noise
			noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
			
			# Random chance to apply different noise types
			noise_type = random.random()
			if noise_type > 0.8:  # 20% chance for salt & pepper noise
				salt_vs_pepper = 0.5
				amount = random.uniform(0.01, 0.04)  # 1-4% of pixels
				# Salt
				num_salt = int(amount * noise_img.size * salt_vs_pepper)
				coords = [np.random.randint(0, i - 1, num_salt) for i in noise_img.shape]
				noise_img[coords[0], coords[1], :] = 255
				# Pepper
				num_pepper = int(amount * noise_img.size * (1 - salt_vs_pepper))
				coords = [np.random.randint(0, i - 1, num_pepper) for i in noise_img.shape]
				noise_img[coords[0], coords[1], :] = 0
				
			return Image.fromarray(noise_img)
		return image

	def _apply_blur(self, image):
		"""Apply various types of blur with random parameters."""
		if random.random() < self.blur_prob:
			blur_type = random.random()
			
			if blur_type < 0.6:  # 60% chance for Gaussian blur
				# Randomly vary blur radius
				radius = random.uniform(self.blur_radius[0], self.blur_radius[1])
				return image.filter(ImageFilter.GaussianBlur(radius))
			
			else:  # 40% chance for box blur
				radius = int(random.uniform(1, 3))
				return image.filter(ImageFilter.BoxBlur(radius))
		return image

	def __call__(self, image, bboxes):
		"""
		Compatible transformations for an image and its associated ground truth boxes of the YOLO dataset.
		
		Inputs:
			image : PIL image or numpy array
			bboxes : list [x, y, width, height, class_label] normalized between 0 and 1 (except class_label between 0 and 12)

		Outputs:
			dictionnary of the transformed image and boxes
		"""
		
		# Convert image to PIL Image if numpy array
		if isinstance(image, np.ndarray):
			image = Image.fromarray(image)
		
		orig_img_width, orig_img_height = image.size #6000x4000
		
		# Initialize transformation parameters
		angle = 0
		translate = (0, 0)
		
		# Apply random transformations only during training
		if self.train:
			# Random horizontal flip
			if random.random() < 0.5:
				image = image.transpose(Image.FLIP_LEFT_RIGHT)
				# Flip bounding boxes' x-coordinates
				for box in bboxes:
					box[0] = 1.0 - box[0]

			# Random vertical flip (optional)
			if random.random() < 0.5:
				image = image.transpose(Image.FLIP_TOP_BOTTOM)
				# Flip bounding boxes' y-coordinates
				for box in bboxes:
					box[1] = 1.0 - box[1]
			
			
			# Random rotation between -max_rotation and +max_rotation degrees
			angle = torch.rand(1).item() * 2 * self.max_rotation - self.max_rotation
			
			# Random translation between -max_translation and +max_translation percent
			tx = (torch.rand(1).item() * 2 - 1) * self.max_translation * self.resized_img_width
			ty = (torch.rand(1).item() * 2 - 1) * self.max_translation * self.resized_img_height
			translate = (tx, ty)
			

		
		# Calculate scaling factor to maintain aspect ratio and pad if the resized image has not the same ratio as the original size
		scale = min(self.resized_img_width / orig_img_width, self.resized_img_height / orig_img_height) # = 0.096
		new_w, new_h = int(orig_img_width * scale), int(orig_img_height * scale) # new_w, new_h = 576, 384 in our case
		
		# Create a blank gray canvas of the target size
		canvas = Image.new('RGB', (self.resized_img_width, self.resized_img_height), (127, 127, 127))
		
		# Resize image (maintains aspect ratio)
		image = image.resize((new_w, new_h))
		
		# Calculate center position for pasting the resized image
		paste_x = (self.resized_img_width - new_w) // 2
		paste_y = (self.resized_img_height - new_h) // 2
		
		
		# Apply rotation and translation if needed
		if angle != 0 or translate != (0, 0):
			# Create larger canvas for rotation/translation to avoid cutting the image
			padding = max(new_w, new_h) // 2
			big_canvas = Image.new('RGB', (self.resized_img_width + 2*padding, self.resized_img_height + 2*padding), (127, 127, 127))
			big_canvas.paste(image, (padding + paste_x, padding + paste_y))
			
			# Apply affine transformation (rotation + translation)
			big_canvas = big_canvas.rotate(angle, resample=Image.BILINEAR, expand=False, 
										translate=(translate[0], translate[1]))
			
			# Crop back to original size
			canvas = big_canvas.crop((padding, padding, padding + self.resized_img_width, padding + self.resized_img_height))
		else:
			# If no rotation/translation, just paste the image directly
			canvas.paste(image, (paste_x, paste_y))
		
		# ======= COLOR TRANSFORMATIONS AFTER resizing =======
		if self.train:
			canvas = transforms.ColorJitter(
				brightness=self.brightness,
				contrast=self.contrast,
				saturation=self.saturation,
				hue=self.hue
			)(canvas)

			canvas = self._add_noise(canvas)
			canvas = self._apply_blur(canvas)


		# Transform bounding boxes
		transformed_bboxes = []
		
		# Calculate transformation matrix for bounding boxes
		# For simplicity, we'll apply transformations in reverse order:
		# 1. Scale and center (always applied)
		# 2. Flip (if applicable)
		# 3. Rotate and translate (if applicable)
		
		for box in bboxes:
			x_center, y_center, w, h, cls = box
			
			# Original image coordinates
			x1 = (x_center - w/2) * orig_img_width
			y1 = (y_center - h/2) * orig_img_height
			x2 = (x_center + w/2) * orig_img_width
			y2 = (y_center + h/2) * orig_img_height
			
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
			if angle != 0 or translate != (0, 0):
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
					px = px_rot + cx + translate[0]
					py = py_rot + cy + translate[1]
					
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
			#if visibility_ratio >= 0.4:  # Minimum size threshold
			transformed_bboxes.append([x_center, y_center, w, h, cls])
		
		# Convert to tensor and normalize
		image_tensor = self.to_tensor(canvas)
		
		return {"image": image_tensor, "bboxes": transformed_bboxes}
		
# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
	def __init__( 
		self, df, image_dir, label_dir, anchors, 
		resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT,
		grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
		num_classes=13, transform=None, ignore_iou_thresh=0.8
	): 
		#df has two columns: image name (LXXXXXXX.jpg) and label name (LXXXXXXX.txt)
		self.label_list = df
		# Image and label directories 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		# Image size treated by YOLO
		self.resized_img_width = resized_img_width
		self.resized_img_height = resized_img_height
		# Transformations 
		self.transform = transform 
		# Grid sizes for each scale 
		self.grid_width = grid_width # 576 // 32 = 18
		self.grid_height = grid_height # 384 // 32 = 12
		# Anchor boxes between 0 and 1
		self.anchors = torch.tensor(anchors) # anchors is a tensor of shape (9, 2)
		# Number of anchor boxes 
		self.num_anchors = len(self.anchors) # 9 anchors
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold (see the end of __getitem__)
		self.ignore_iou_thresh = ignore_iou_thresh

	def __len__(self): 
		return len(self.label_list) 
	
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
				i, j = int(self.grid_height * y), int(self.grid_width * x)
				
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
		
		# Return the image and the target
		return image, target



train_transform = YoloCompatibleTransform(resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT, train=True)
test_transform = YoloCompatibleTransform(resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT, train=False)

"""
# Read full CSV that has 2 columns: image name (LXXXXXXX.JPG) and label name (LXXXXXXX.txt)
full_df = pd.read_csv("./link_images_labels_YOLO.csv")
# Split into train/validation
train_df, val_df = train_test_split(full_df, test_size=0.075, random_state=0)
"""

# Read the full CSV
full_df = pd.read_csv("./link_images_labels_YOLO.csv")

# Liste des images que tu veux dans le validation set
val_images = [
	"L1000850.JPG", "L1000785.JPG", "L1000772.JPG", "L1000894.JPG",
	"L1000899.JPG", "L1000903.JPG", "L1000910.JPG", "L1000981.JPG"
]

# Créer val_df en filtrant les images spécifiques
val_df = full_df[full_df['image_name'].isin(val_images)]

# Le reste va dans le train_df
train_df = full_df[~full_df['image_name'].isin(val_images)]

# Afficher pour vérifier
print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

########################################################## MODEL ##########################################################

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

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

class YOLO_YOCO_Flexible(nn.Module):
    def __init__(self, in_channels=3, num_classes=13, num_anchors=9, grid_ratio=32, scale_factor=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.grid_ratio = grid_ratio
        self.num_anchors = num_anchors

        def s(c):
            return max(8, int(c * scale_factor))

        self.backbone = nn.Sequential(
            CNNBlock(in_channels, s(32), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(32), s(32), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(32), s(64), kernel_size=3, stride=2, padding=1),  # Downsample 1/2
            CNNBlock(s(64), s(64), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(64), s(128), kernel_size=3, stride=2, padding=1), # Downsample 1/4
            CNNBlock(s(128), s(128), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(128), s(256), kernel_size=3, stride=2, padding=1), # Downsample 1/8
            CNNBlock(s(256), s(256), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(256), s(256), kernel_size=3, stride=2, padding=1), # Downsample 1/16
            CNNBlock(s(256), s(256), kernel_size=3, stride=1, padding=1),
        )

        if self.grid_ratio == 32:
            self.extra_down = nn.Sequential(
                CNNBlock(s(256), s(384), kernel_size=3, stride=2, padding=1),  # Downsample 1/32
                CNNBlock(s(384), s(384), kernel_size=3, stride=1, padding=1),
            )
            feature_channels = s(384)
        elif self.grid_ratio == 16:
            self.extra_down = nn.Identity()
            feature_channels = s(256)
        else:
            raise ValueError("Unsupported grid_ratio, only 16 or 32 supported")

        self.head = nn.Sequential(
            CNNBlock(feature_channels, s(256), kernel_size=3, padding=1),
            CNNBlock(s(256), s(256), kernel_size=3, padding=1),
        )

        self.scale_pred = nn.Sequential(
            CNNBlock(s(256), s(256), kernel_size=3, padding=1),
            ScalePrediction(s(256), num_classes, num_anchors)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.extra_down(features)
        x = self.head(features)
        output = self.scale_pred(x)

        # Assert sécurité
        expected_h, expected_w = output.shape[2], output.shape[3]
        if self.grid_ratio == 32:
            assert expected_h == 12 and expected_w == 18, f"Erreur : grille attendue (12, 18) mais obtenue ({expected_h}, {expected_w})"
        elif self.grid_ratio == 16:
            assert expected_h == 24 and expected_w == 36, f"Erreur : grille attendue (24, 36) mais obtenue ({expected_h}, {expected_w})"

        return output

# Initialisation YOLOv3-like robuste
def init_yolo_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for m in model.modules():
        if isinstance(m, ScalePrediction):
            last_conv = m.pred[-1]
            with torch.no_grad():
                bias = last_conv.bias.view(m.num_anchors, 5 + m.num_classes)
                bias[:, 4].fill_(-4.6)  # Objectness bias logit ≈0.01
                bias[:, :4].fill_(0)
                bias[:, 5:].fill_(0)
                last_conv.bias.copy_(bias.view(-1))
    return model

############################################### LOSS ############################################################

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
		self.lambda_no_object = 3.0 # weight for no object loss
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

	
		# Total loss 
		return ( 
			self.lambda_box * box_loss 
			+ self.lambda_object * object_loss 
			+ self.lambda_no_object * no_object_loss 
			+ self.lambda_class * class_loss 
		)

def init_yolo_weights(model):
	"""
	Initialisation complète des poids pour YOLOv3-like :
	- Conv2d : Kaiming normal (He initialization)
	- Bias : 0 sauf pour objectness qui est initialisé à -4.6 (~0.01 après sigmoïde)
	- BatchNorm : gamma=1, beta=0
	"""
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)
	
	# Initialisation spécifique pour objectness (comme YOLOv3 officiel)
	for m in model.modules():
		if isinstance(m, ScalePrediction):
			last_conv = m.pred[-1]  # Dernière couche Conv2d dans ScalePrediction
			if isinstance(last_conv, nn.Conv2d):
				# Poids : Kaiming normal
				nn.init.kaiming_normal_(last_conv.weight, mode='fan_out', nonlinearity='relu')
				# Bias : objectness à -4.6 (~0.01 après sigmoid), classes à 0
				with torch.no_grad():
					bias = last_conv.bias.view(m.num_anchors, 5 + m.num_classes)
					bias[:, 4].fill_(-4.6)  # Objectness logit initialisé bas (≈0.01)
					bias[:, 0:4].fill_(0)   # Tx, Ty, Tw, Th à 0
					bias[:, 5:].fill_(0)    # Class logits à 0
					last_conv.bias.copy_(bias.view(-1))
	return model


################################################### TRAINING AND EVALUATION ############################################################

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
		x = x.to(device) # x is a tensor of shape (batch_size, 3, 416, 416)
		y = y.to(device) # y0 is a tensor of shape (batch_size, 9, 26, 26, 6) (num_anchors, grid_size, grid_size, 6)

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

