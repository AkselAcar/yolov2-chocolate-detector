#################################### IMPORT PACKAGES ####################################

import torch 
import torch.nn as nn 
import torch.optim as optim 

from PIL import Image, ImageFile 
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


 
############################ DEFINE SOME HELPER FUNCTIONS ############################

# Defining a function to calculate Intersection over Union (IoU) 
def iou(box1, box2, is_pred=True):
	if is_pred: 
		# IoU score for prediction and label 
		# box1 (prediction) and box2 (label) are both in [x, y, width, height] format 
		
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
	
	else: # this case is used in the dataset class to calculate IoU of anchor boxes with ground truth boxes (so don't need x, y coordinates), on veut juste la ressemblance de la GT avec les 9 ancres
		# IoU score based on width and height of bounding boxes 
		
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


######################### NONMAX SUPPRESSION TO REMOVE OVERLAPPING BOUNDING BOXES #########################
# Non-maximum suppression function to remove overlapping bounding boxes 
"""
def nms(bboxes, iou_threshold, threshold): 
	# Filter out bounding boxes with confidence below the threshold. 
	bboxes = [box for box in bboxes if box[1] > threshold] 

	# Sort the bounding boxes by confidence in descending order. 
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 

	# Initialize the list of bounding boxes after non-maximum suppression. 
	bboxes_nms = [] 

	while bboxes: 
		# Get the first bounding box. 
		first_box = bboxes.pop(0) 

		# Iterate over the remaining bounding boxes. 
		for box in bboxes: 
		# If the bounding boxes do not overlap or if the first bounding box has 
		# a higher confidence, then add the second bounding box to the list of 
		# bounding boxes after non-maximum suppression. 
			if box[0] != first_box[0] or iou( 
				torch.tensor(first_box[2:]), 
				torch.tensor(box[2:]), 
			) < iou_threshold: 
				# Check if box is not in bboxes_nms 
				if box not in bboxes_nms: 
					# Add box to bboxes_nms 
					bboxes_nms.append(box) 

	# Return bounding boxes after non-maximum suppression. 
	return bboxes_nms
"""
"""
def nms(bboxes, iou_threshold, threshold):
	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	bboxes_nms = []
	while bboxes:
		first_box = bboxes.pop(0)
		bboxes_nms.append(first_box)
		# Keep only boxes that don't overlap significantly with first_box
		bboxes = [
			box for box in bboxes
			if box[0] != first_box[0] or  # Different class
			iou(torch.tensor(first_box[2:]), 
				 torch.tensor(box[2:])) < iou_threshold
		]
	return bboxes_nms
"""
def nms(bboxes, iou_threshold, threshold):
	"""
	Non-maximum suppression to remove overlapping bounding boxes
	
	Args:
		bboxes: List of bounding boxes [class, confidence, x, y, width, height]
		iou_threshold: Threshold for considering boxes as overlapping
		threshold: Confidence threshold to filter out low-confidence boxes
		
	Returns:
		List of bounding boxes after NMS
	"""
	# Filter out boxes with confidence below threshold
	bboxes = [box for box in bboxes if box[1] > threshold]
	
	# If no boxes remain after threshold filtering, return empty list
	if not bboxes:
		return []
	
	# Sort by confidence (higher confidence first)
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	
	bboxes_nms = []
	
	while bboxes:
		# Take the box with highest confidence
		chosen_box = bboxes.pop(0)
		
		# Keep the chosen box
		bboxes_nms.append(chosen_box)
		
		# Filter remaining boxes
		remaining_boxes = []
		for box in bboxes:
			# Keep boxes that are different class OR have IoU less than threshold
			if box[0] != chosen_box[0] or iou(
				torch.tensor(chosen_box[2:]), 
				torch.tensor(box[2:])
			) < iou_threshold:
				remaining_boxes.append(box)
		
		# Update the bboxes list with remaining boxes
		bboxes = remaining_boxes
	
	return bboxes_nms

##################################### CONVERTS CELLS TO BOUNDING BOXES #####################################

# Function to convert cells to bounding boxes, predictions = [B, A, S, S, 5 + C]
# return : Une liste de taille B, où chaque élément est une liste de A × S × S boîtes, et chaque boîte est un vecteur de 6 valeurs : [class, score, x_center, y_center, width, height]
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
	# Batch size used on predictions 
	batch_size = predictions.shape[0] 
	# Number of anchors 
	num_anchors = len(anchors) 
	# List of all the predictions 
	box_predictions = predictions[..., 1:5] 

	# If the input is predictions then we will pass the x and y coordinate 
	# through sigmoid function and width and height to exponent function and 
	# calculate the score and best class. 
	if is_predictions: 
		anchors = anchors.reshape(1, num_anchors, 1, 1, 2).to(predictions.device)
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
		box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors 
		scores = torch.sigmoid(predictions[..., 0:1]) 
		best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
	
	# Else we will just calculate scores and best class. 
	else: 
		scores = predictions[..., 0:1] 
		best_class = predictions[..., 5:6] 

	# Calculate cell indices 
	cell_indices = (torch.arange(s).repeat(batch_size, num_anchors, s, 1).unsqueeze(-1).to(predictions.device))

	# Calculate x, y, width and height with proper scaling 
	x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	# Concatinating the values and reshaping them in 
	# (BATCH_SIZE, num_anchors * S * S, 6) shape 
	converted_bboxes = torch.cat((best_class, scores, x, y, width_height), dim=-1)
	converted_bboxes = converted_bboxes.reshape(batch_size, num_anchors * s * s, 6)

	# Returning the reshaped and converted bounding box list 
	return converted_bboxes.tolist()

###################################### PLOT IMAGES WITH BOUNDING BOXES AND CLASS LABELS ######################################
"""
# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes, save_dir=None, filename="image.png"): 
	# Getting the color map from matplotlib 
	colour_map = plt.get_cmap("tab20b") 
	# Getting 20 different colors from the color map for 13 different classes 
	colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 

	# Reading the image with OpenCV 
	img = np.array(image) 
	# Getting the height and width of the image 
	h, w, _ = img.shape 

	# Create figure and axes 
	fig, ax = plt.subplots(1) 

	# Add image to plot 
	ax.imshow(img) 

	# Plotting the bounding boxes and labels over the image 
	for box in boxes: 
		# Get the class from the box 
		class_pred = box[0] 
		# Get the center x and y coordinates 
		box = box[2:] 
		# Get the upper left corner coordinates 
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2

		# Create a Rectangle patch with the bounding box 
		rect = patches.Rectangle( 
			(upper_left_x * w, upper_left_y * h), 
			box[2] * w, 
			box[3] * h, 
			linewidth=2, 
			edgecolor=colors[int(class_pred)], 
			facecolor="none", 
		) 
		
		# Add the patch to the Axes 
		ax.add_patch(rect) 
		
		# Add class name to the patch 
		plt.text( 
			upper_left_x * w, 
			upper_left_y * h, 
			s=class_labels[int(class_pred)], 
			color="white", 
			verticalalignment="top", 
			bbox={"color": colors[int(class_pred)], "pad": 0}, 
		) 

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		plt.savefig(os.path.join(save_dir, filename))
		plt.close()
	else:
		plt.show()
"""
def plot_image(image, boxes, save_dir=None, filename="image.png"):
	"""
	Function to plot images with bounding boxes and class labels.
	
	Args:
		image: Image as numpy array (H, W, C) in [0, 255]
		boxes: List of bounding boxes [class, confidence, x, y, width, height]
		save_dir: Directory to save the image
		filename: Filename to save the image
	"""
	# Create figure and axes 
	fig, ax = plt.subplots(1, figsize=(12, 12))
	fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
	ax.axis("off")

	# Convert image to uint8 if needed
	if image.dtype != np.uint8:
		image = image.astype(np.uint8)
	
	# Display image
	ax.imshow(image)

	# Get color map
	colour_map = plt.get_cmap("tab20b")
	colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

	# Process each box
	for box in boxes:
		class_pred = int(box[0])
		confidence = box[1]
		
		# Skip if confidence is too low
		if confidence < 0.6:
			continue

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
		if x2 <= x1 or y2 <= y1:
			continue

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
		text = f"{class_labels[class_pred]}: {confidence:.2f}"
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
	checkpoint = torch.load(checkpoint_file, map_location=device) 
	model.load_state_dict(checkpoint["state_dict"]) 
	optimizer.load_state_dict(checkpoint["optimizer"]) 

	for param_group in optimizer.param_groups: 
		param_group["lr"] = lr 

################################## DEFINE SOME CONSTANTS ##################################
# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes scaled between 0 and 1 
# list de 9 tuples
ANCHORS = [                                                               
	(0.14105, 0.10625), (0.11015, 0.16249), (0.09875, 0.14781),
	(0.08998, 0.13511), (0.08254, 0.12333), (0.07731, 0.19987),
	(0.07835, 0.09580), (0.06312, 0.11607), (0.06459, 0.21892)]



# Batch size for training 
batch_size = 1

# Learning rate for training 
learning_rate = 1e-4

# Number of epochs for training 
epochs = 50

# Image size 
image_size = 416

# Grid cell sizes 
GRID_SIZE = image_size // 16

# Class labels 
class_labels = [ 
	"Jelly_White","Jelly_Milk","Jelly_Black","Amandina","Creme_brulee",
	"Triangolo","Tentation_noir","Comtesse","Noblesse","Noir_authentique",
	"Passion_au_lait","Arabia","Stracciatella"
]

############################ DATASET  ############################  
# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
	def __init__( 
		self, df, image_dir, label_dir, anchors, 
		image_size=416, grid_size=GRID_SIZE, 
		num_classes=13, transform=None
	): 
		# Read the csv file with image names and labels 
		self.label_list = df #pd.read_csv(csv_file) #df has two columns: image name (LXXXXXXX.jpg) and label name (LXXXXXXX.txt)
		# Image and label directories 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		# Image size 
		self.image_size = image_size 
		# Transformations 
		self.transform = transform 
		# Grid sizes for each scale 
		self.grid_size = grid_size
		# Anchor boxes # anchors is initially a list of 3 lists, each containing 3 tuples (width, height) normalized between 0 and 1
		self.anchors = torch.tensor(anchors) # anchors is now a tensor of shape (9, 2)
		# Number of anchor boxes 
		self.num_anchors = len(self.anchors) # 9 anchors
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold 
		self.ignore_iou_thresh = 0.5 # MAYBE CHANGE TO 0.7-0.8 ???????????????????????????????????????????

	def __len__(self): 
		return len(self.label_list) 
	
	def __getitem__(self, idx):
		# Getting the label path 
		label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) #self.label_list.iloc[idx, 1] récupère la valeur de la colonne 1 (donc label_name) à la ligne idx
		# We are applying roll to move class label to the last column 
		# 5 columns: x, y, width, height, class_label 
		bboxes = np.roll(np.loadtxt(fname=label_path, 
						delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.loadtxt() charge le fichier label_path (fichier .txt) contenant les annotations YOLO.
																	 # ndmin=2 garantit que même s’il n’y a qu’une seule ligne, on obtient un tableau 2D.
																	 # np.roll() déplace la colonne de la classe à la fin du tableau.
																	 # .tolist() crée une liste de liste(s) à partir du tableau numpy.
		# Getting the image path 
		img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
		image = np.array(Image.open(img_path).convert("RGB")) 

		# Augmentations 
		if self.transform: 
			augs = self.transform(image=image, bboxes=bboxes) 
			image = augs["image"] # image is a tensor of shape (3, 416, 416)
			bboxes = augs["bboxes"] # bboxes is a list of bounding boxes in the format [x, y, width, height, class_label] with x,y,w,h normalized between 0 and 1

		# Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
		# target : [probabilities, x, y, width, height, class_label] 
		# for now targets is empty
		target = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 6)) # targets is a tensor of shape (num_anchors_per_scale, s_i, s_i, 6) et 6 car [probabilities, x, y, width, height, class_label]
										  # en gros pour chaque échelle on crée une grille dont chaque case contient 6 valeurs (probabilité, x, y, width, height, class_label)
										  # targets shape:  torch.Size([9, 26, 26, 6])

		# Identify anchor box and cell for each bounding box 
		for box in bboxes: 
			# Calculate iou of each bounding box WITH ANCHOR BOXES
			iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False) # iou_anchors is a tensor of shape (9,)
													  
			# Selecting the best anchor box 
			anchor_indices = iou_anchors.argsort(descending=True, dim=0) # anchor_indices is a tensor of shape (9,)
			x, y, width, height, class_label = box # x,y,width,height,class_label are all normalized between 0 and 1

			# Assigning the bounding box to the best matching anchor box
			has_anchor = False
			
			for anchor_idx in anchor_indices:
				# Identifying the cell to which the bounding box belongs
				i, j = int(self.grid_size * y), int(self.grid_size * x)
				
				# Check if the anchor box is already assigned
				anchor_taken = target[anchor_idx, i, j, 0]
				
				if not anchor_taken and not has_anchor:
					# Set the probability to 1 
					target[anchor_idx, i, j, 0] = 1

					# Calculating the center of the bounding box relative to the cell 
					x_cell, y_cell = self.grid_size * x - j, self.grid_size * y - i  # x_cell, y_cell are the coordinates of the center of the bounding box relative to the cell (0-1)

					# Calculating the width and height of the bounding box relative to the grid 
					width_cell, height_cell = (width * self.grid_size, height * self.grid_size) # width_cell, height_cell are the width and height of the bounding box relative to the grid

					# Identify the box coordinates 
					box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

					# Assigning the box coordinates to the target 
					target[anchor_idx, i, j, 1:5] = box_coordinates 

					# Assigning the class label to the target 
					target[anchor_idx, i, j, 5] = int(class_label) 

					# Set the anchor box as assigned
					has_anchor = True

				# If the anchor box is already assigned, check if the IoU is greater than the threshold 
				elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
					# Set the probability to -1 to ignore the anchor box 
					target[anchor_idx, i, j, 0] = -1
		
		# Return the image and the target (now just a single tensor)
		return image, target

######################################### DATA AUGMENTATION #########################################
class YoloCompatibleTransform:
	def __init__(self, image_size=416, train=True, 
				 max_rotation=10, max_translation=0.1,
				 brightness=0.3, contrast=0.3, saturation=0.1, hue=0.01): # or 0.2 0.2 0.2 0.1
		"""
		Transformations compatibles avec YOLO incluant rotations, translations et ColorJitter.
		Préserve l'alignement des boîtes avec les axes horizontal/vertical.
		
		Args:
			image_size (int): Taille finale de l'image (carrée)
			train (bool): Si True, applique les augmentations aléatoires
			max_rotation (float): Angle maximum de rotation en degrés (±)
			max_translation (float): Translation maximum en pourcentage de l'image (±)
			brightness (float): Variation maximum de luminosité (0-1)
			contrast (float): Variation maximum de contraste (0-1)
			saturation (float): Variation maximum de saturation (0-1)
			hue (float): Variation maximum de teinte (0-0.5)
		"""
		self.image_size = image_size
		self.train = train
		self.max_rotation = max_rotation
		self.max_translation = max_translation
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue
		self.to_tensor = transforms.ToTensor()
		
	def __call__(self, image, bboxes):
		# Convert image to PIL Image if numpy array
		if isinstance(image, np.ndarray):
			image = Image.fromarray(image)
		
		orig_w, orig_h = image.size
		
		# Initialize transformation parameters
		angle = 0
		translate = (0, 0)
		do_flip = False
		
		# Apply random transformations only during training
		if self.train:
			# Random horizontal flip
			do_flip = torch.rand(1).item() > 0.5
			
			# Random rotation between -max_rotation and +max_rotation degrees
			angle = torch.rand(1).item() * 2 * self.max_rotation - self.max_rotation
			
			# Random translation between -max_translation and +max_translation percent
			tx = (torch.rand(1).item() * 2 - 1) * self.max_translation * self.image_size
			ty = (torch.rand(1).item() * 2 - 1) * self.max_translation * self.image_size
			translate = (tx, ty)
			
			# Apply ColorJitter
			jitter = transforms.ColorJitter(
				brightness=self.brightness,
				contrast=self.contrast,
				saturation=self.saturation,
				hue=self.hue
			)
			image = jitter(image)
		
		# Calculate scaling factor to maintain aspect ratio
		scale = min(self.image_size / orig_w, self.image_size / orig_h)
		new_w, new_h = int(orig_w * scale), int(orig_h * scale)
		
		# Create a blank canvas of the target size
		canvas = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
		
		# Resize image (maintains aspect ratio)
		image = image.resize((new_w, new_h))
		
		# Calculate center position for pasting the resized image
		paste_x = (self.image_size - new_w) // 2
		paste_y = (self.image_size - new_h) // 2
		
		# Apply horizontal flip if needed (before rotation/translation)
		if do_flip:
			image = F.hflip(image)
		
		# Apply rotation and translation
		if angle != 0 or translate != (0, 0):
			# Create larger canvas for rotation/translation to avoid cutting the image
			padding = max(new_w, new_h) // 2
			big_canvas = Image.new('RGB', (self.image_size + 2*padding, self.image_size + 2*padding), (0, 0, 0))
			big_canvas.paste(image, (padding + paste_x, padding + paste_y))
			
			# Apply affine transformation (rotation + translation)
			big_canvas = big_canvas.rotate(angle, resample=Image.BILINEAR, expand=False, 
										translate=(translate[0], translate[1]))
			
			# Crop back to original size
			canvas = big_canvas.crop((padding, padding, padding + self.image_size, padding + self.image_size))
		else:
			# If no rotation/translation, just paste the image directly
			canvas.paste(image, (paste_x, paste_y))
		
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
			x1 = (x_center - w/2) * orig_w
			y1 = (y_center - h/2) * orig_h
			x2 = (x_center + w/2) * orig_w
			y2 = (y_center + h/2) * orig_h
			
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
			
			# Step 3: Apply horizontal flip if needed
			if do_flip:
				x1, x2 = self.image_size - x2, self.image_size - x1
			
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
					cx, cy = self.image_size / 2, self.image_size / 2
					
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
			x1 = np.clip(x1, 0, self.image_size)
			y1 = np.clip(y1, 0, self.image_size)
			x2 = np.clip(x2, 0, self.image_size)
			y2 = np.clip(y2, 0, self.image_size)
			
			# Calculate how much of the box is still inside the image (intersection over original box)
			original_area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
			if original_area <= 0:  # Avoid division by zero
				continue
				
			intersection_area = (x2 - x1) * (y2 - y1)
			visibility_ratio = intersection_area / original_area
			
			# If less than 25% of the box is visible, we skip it
			if visibility_ratio < 0.25:
				continue
			
			# Convert back to YOLO format (x_center, y_center, width, height)
			w = (x2 - x1) / self.image_size
			h = (y2 - y1) / self.image_size
			x_center = (x1 + x2) / 2 / self.image_size
			y_center = (y1 + y2) / 2 / self.image_size
			
			# Add transformed box if it's still visible and large enough
			if w > 0.005 and h > 0.005 and visibility_ratio >= 0.25:  # Minimum size threshold
				transformed_bboxes.append([x_center, y_center, w, h, cls])
		
		# Convert to tensor and normalize
		image_tensor = self.to_tensor(canvas)
		image_tensor = (image_tensor - 0.0) / 1.0  # Simple normalization
		
		return {"image": image_tensor, "bboxes": transformed_bboxes}

train_transform = YoloCompatibleTransform(image_size=416, train=True)
test_transform = YoloCompatibleTransform(image_size=416, train=False)

######################################### LOAD DATASET #########################################
"""
# # Creating a dataset object 
dataset = Dataset( 
	csv_file="link_images_labels_YOLO_v3.csv", 
	image_dir="../dataset_project_iapr2025/train", 
	label_dir="labels/", 
	grid_sizes=[13, 26, 52], 
	anchors=ANCHORS, 
	transform=test_transform 
) 

# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=1, 
	shuffle=True, 
) 

# Defining the grid size and the scaled anchors 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
	1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

# Getting a batch from the dataloader 
x, y = next(iter(loader)) 

# Getting the boxes coordinates from the labels 
# and converting them into bounding boxes without scaling 
boxes = [] 
for i in range(y[0].shape[1]): 
	anchor = scaled_anchors[i] 
	boxes += convert_cells_to_bboxes( 
			y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
			)[0] 

# Applying non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 

# Plotting the image with the bounding boxes 
plot_image(x[0].permute(1,2,0).to("cpu"), boxes)
"""
############################################### MODEL  ##############################################
# Defining CNN Block 
class CNNBlock(nn.Module): 
	def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
		super().__init__() 
		self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs) 
		self.bn = nn.BatchNorm2d(out_channels) 
		self.activation = nn.LeakyReLU(0.1) 
		self.use_batch_norm = use_batch_norm 

	def forward(self, x): 
		# Applying convolution 
		x = self.conv(x) 
		# Applying BatchNorm and activation if needed 
		if self.use_batch_norm: 
			x = self.bn(x) 
			return self.activation(x) 
		else: 
			return x
# Defining residual block 
class ResidualBlock(nn.Module): 
	def __init__(self, channels, use_residual=True, num_repeats=1): 
		super().__init__() 
		
		# Defining all the layers in a list and adding them based on number of 
		# repeats mentioned in the design 
		res_layers = [] 
		for _ in range(num_repeats): 
			res_layers += [ 
				nn.Sequential( 
					nn.Conv2d(channels, channels // 2, kernel_size=1), 
					nn.BatchNorm2d(channels // 2), 
					nn.LeakyReLU(0.1), 
					nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1), 
					nn.BatchNorm2d(channels), 
					nn.LeakyReLU(0.1) 
				) 
			] 
		self.layers = nn.ModuleList(res_layers) 
		self.use_residual = use_residual 
		self.num_repeats = num_repeats 
	
	# Defining forward pass 
	def forward(self, x): 
		for layer in self.layers: 
			residual = x 
			x = layer(x) 
			if self.use_residual: 
				x = x + residual 
		return x

# Defining scale prediction class 
class ScalePrediction(nn.Module): 
	def __init__(self, in_channels, num_classes, num_anchors): 
		super().__init__() 
		# Defining the layers in the network 
		self.pred = nn.Sequential( 
			nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1), 
			nn.BatchNorm2d(2*in_channels), 
			nn.LeakyReLU(0.1), 
			nn.Conv2d(2*in_channels, (num_classes + 5) * num_anchors, kernel_size=1), 
		) 
		self.num_classes = num_classes 
		self.num_anchors = num_anchors
	
	# Defining the forward pass and reshaping the output to the desired output 
	# format: (batch_size, 3, grid_size, grid_size, num_classes + 5) 
	def forward(self, x): 
		B, _, H, W = x.shape
		output = self.pred(x) 
		output = output.view(B, self.num_anchors, self.num_classes + 5, H, W)
		output = output.permute(0, 1, 3, 4, 2) 
		return output


# Class for defining YOLOv3 model 
class YOLOv3(nn.Module): 
	def __init__(self, in_channels=3, num_classes=13): 
		super().__init__() 
		self.num_classes = num_classes 
		self.in_channels = in_channels 

		# Layers list for YOLOv3 
		self.model = nn.Sequential(
			CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1), 
			CNNBlock(32, 64, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(64, num_repeats=1), 
			CNNBlock(64, 128, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(128, num_repeats=2), 
			CNNBlock(128, 256, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(256, num_repeats=8), 
			CNNBlock(256, 512, kernel_size=3, stride=2, padding=1), 
			ResidualBlock(512, num_repeats=8), 
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(1024, num_repeats=4), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1), 
			ResidualBlock(1024, use_residual=False, num_repeats=1), 
			CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0), 
			ScalePrediction(512, num_classes=num_classes, num_anchors=len(ANCHORS))
		) 
	
	# Forward pass for YOLOv3 with route connections and scale predictions 
	def forward(self, x):
		x = self.model(x)
		return x
# input.shape = (batch_size, 3, 416, 416)
# output = liste de |s| tenseurs de forme (batch_size, num_anchors dans chaque scale, grid_size, grid_size, num_classes + 5)

##################################### TESTING THE MODEL #####################################
# Setting number of classes and image size 
num_classes = 13
IMAGE_SIZE = 416


######################################### TRAINING THE MODEL #########################################
# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss()                    # pour les erreurs continues (x, y, w, h, objectness)
		self.bce = nn.BCEWithLogitsLoss()          # pour l'objectness = score (avec logits)
		self.cross_entropy = nn.CrossEntropyLoss() # pour la classification (classes)
		self.sigmoid = nn.Sigmoid()                # pour forcer x, y, objectness ∈ [0,1]
	
	def forward(self, pred, target, anchors): # target = (batch_size, 9, 26, 26, 6) = (batch_size, num_anchors, grid_size, grid_size, 6), 6 = (objectness, x, y, w, h, class_label), (x,y) relatif to the cell, (w,h) relative to the grid size
		                                      # pred = (batch_size, 9, 26, 26, num_classes + 5)
						      				  # anchors sera le scale_anchors entre 0 et 26
		# Identifying which cells in target have objects 
		# and which have no objects
		obj = target[..., 0] == 1 # obj.sum = nombre d'objets dans la batch
		no_obj = target[..., 0] == 0

		# Calculating No object loss 
		# On applique la Binary Cross Entropy entre le score prédit (logit) et le label (0), pour les cellules sans objet.
		# Cela apprend au modèle à ne pas détecter d’objets là où il n’y en a pas.
		no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

		
		# Reshaping anchors to match predictions 
		anchors = anchors.reshape(1, pred.size(1), 1, 1, 2).to(pred.device) # pred.size(1) = 9 ==> anchors = (1, 9, 1, 1, 2) ce qui est compatible avec pred[..., 3:5].shape
					#             │     │            │  │  │
		    		#             │     │            │  │  └─ largeur et hauteur (w, h)
					#             │     │            │  └──── broadcasté sur chaque cellule colonne (W)
					#             │     │            └─────── broadcasté sur chaque cellule ligne (H)
					#             │     └──────────────────── num_anchors (ex: 9)
					#             └────────────────────────── batch_size (à broadcast)

		# Box prediction confidence 
		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
							torch.exp(pred[..., 3:5]) * anchors 
							],dim=-1) 
		# Calculating intersection over union for prediction and target 
		ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
		# Calculating Object loss 
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), 
							   ious * target[..., 0:1][obj]) #target[..., 0:1] == 1 quand il y a un objet (sinon 0). But : la sortie pred[..., 0] doit être proche de IoU si l’objet est bien prédit.

		
		# Predicted box coordinates 
		pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
		# Target box coordinates 
		target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
		# Calculating box coordinate loss 
		box_loss = self.mse(pred[..., 1:5][obj], 
							target[..., 1:5][obj]) 

		
		# Claculating class loss 
		class_loss = self.cross_entropy((pred[..., 5:][obj]), 
									     target[..., 5][obj].long()) 

		# Total loss 
		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)
	
def evaluate(loader, model, loss_fn, scaled_anchors, device):
	model.eval()
	losses = []
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			with torch.amp.autocast(device):
				outputs = model(x)
				loss = loss_fn(outputs, y, scaled_anchors)
				losses.append(loss.item())
	mean_loss = sum(losses) / len(losses) if losses else 0
	return mean_loss

# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
	# Creating a progress bar 
	progress_bar = tqdm(loader, leave=True) 

	# Initializing a list to store the losses 
	losses = [] 
	# Iterating over the training data 
	for _, (x, y) in enumerate(progress_bar): 
		x = x.to(device) # x is a tensor of shape (batch_size, 3, 416, 416)
		y = y.to(device) # y0 is a tensor of shape (batch_size, 9, 26, 26, 6) (num_anchors, grid_size, grid_size, 6)

		with torch.amp.autocast(device): 
			# Getting the model predictions 
			outputs = model(x) # tensor (batch_size, 9, grid_size, grid_size, num_classes + 5)
			# Calculating the loss at each scale 
			loss = loss_fn(outputs, y, scaled_anchors) # scaled_anchors is a tensor of shape (9, 2) (num_anchors, 2)

		# Add the loss to the list
		losses.append(loss.item()) 

		# Reset gradients 
		optimizer.zero_grad() 

		# Backpropagate the loss 
		scaler.scale(loss).backward() 

		# Optimization step 
		scaler.step(optimizer) 

		# Update the scaler for next iteration 
		scaler.update() 

		# update progress bar with loss 
		mean_loss = sum(losses) / len(losses) 
		progress_bar.set_postfix(loss=mean_loss)



# Creating the model from YOLOv3 class 
model = YOLOv3().to(device) 
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Nombre de paramètres entraînables : {num_params}")
# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

# Defining the loss function 
loss_fn = YOLOLoss() 

# Defining the scaler for mixed precision training 
scaler = torch.amp.GradScaler(device)


from sklearn.model_selection import train_test_split

# Read full CSV that has 2 columns: image name (LXXXXXXX.JPG) and label name (LXXXXXXX.txt)
full_df = pd.read_csv("./link_images_labels_YOLO_v3.csv")
# Split into train/validation (80/20)
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = Dataset(
	df=train_df,
	image_dir="../dataset_project_iapr2025/train/",
	label_dir="./labels/",
	anchors=ANCHORS,
	transform=train_transform
)
val_dataset = Dataset(
	df=val_df,
	image_dir="../dataset_project_iapr2025/train/",  # Same as train
	label_dir="./labels/",
	anchors=ANCHORS,
	transform=test_transform
)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=batch_size,
	shuffle=True,
	pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=batch_size,
	shuffle=False,
	pin_memory=True,
)

# Scaling the anchors 
scaled_anchors = torch.tensor(ANCHORS) * GRID_SIZE
scaled_anchors = scaled_anchors.to(device)  # si tu es sur GPU
"""
# Helper function to save validation images
def save_val_images(model, loader, device, anchors, grid_sizes, save_dir, epoch, num_images=3):
	model.eval()
	os.makedirs(save_dir, exist_ok=True)
	with torch.no_grad():
		x, _ = next(iter(loader))
		x = x.to(device)
		outputs = model(x)
		for img_idx in range(min(num_images, x.size(0))):
			all_boxes = []
			for scale_idx in range(3):
				anchor = torch.tensor(anchors[scale_idx]).to(device) #anchors[scale_idx]
				s = grid_sizes[scale_idx]
				pred = outputs[scale_idx][img_idx].unsqueeze(0)
				boxes = convert_cells_to_bboxes(pred, anchor, s, is_predictions=True)[0]
				all_boxes.extend(boxes)
			scores = [box[1] for box in all_boxes]
			print(f"Max score: {max(scores):.4f}, Min score: {min(scores):.4f}")
			boxes_nms = nms(all_boxes, iou_threshold=0.5, threshold=0.6)
			# Convert tensor to image
			img = x[img_idx].permute(1, 2, 0).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			plot_image(img, boxes_nms, save_dir=save_dir, filename=f"epoch_{epoch}_img_{img_idx}.png")
	model.train()
"""
from torchvision.ops import nms as torch_nms

def save_val_images(model, loader, device, anchors, grid_sizes, save_dir, epoch, num_images=3):
	model.eval()
	os.makedirs(save_dir, exist_ok=True)
	with torch.no_grad():
		x, _ = next(iter(loader))
		x = x.to(device)
		outputs = model(x)
		
		for img_idx in range(min(num_images, x.size(0))):
			# Prepare tensors for torchvision NMS
			all_boxes = []
			all_scores = []
			all_labels = []
			
			for scale_idx in range(3):
				anchor = torch.tensor(anchors[scale_idx]).to(device)
				s = grid_sizes[scale_idx]
				pred = outputs[scale_idx][img_idx].unsqueeze(0)
				
				# Convert predictions to boxes [batch_size, num_anchors*s*s, 6]
				boxes = convert_cells_to_bboxes(pred, anchor, s, is_predictions=True)
				boxes = torch.tensor(boxes[0]).to(device)  # Get first (and only) batch item
				
				# Split into components
				labels = boxes[:, 0]        # class indices
				scores = boxes[:, 1]        # confidences
				coords = boxes[:, 2:6]      # [x_center, y_center, width, height]
				
				# Convert from center to corner coordinates [x1, y1, x2, y2]
				boxes_corners = torch.zeros_like(coords)
				boxes_corners[:, 0] = coords[:, 0] - coords[:, 2]/2  # x1
				boxes_corners[:, 1] = coords[:, 1] - coords[:, 3]/2  # y1
				boxes_corners[:, 2] = coords[:, 0] + coords[:, 2]/2  # x2
				boxes_corners[:, 3] = coords[:, 1] + coords[:, 3]/2  # y2
				
				all_boxes.append(boxes_corners)
				all_scores.append(scores)
				all_labels.append(labels)
			
			if len(all_boxes) > 0:
				all_boxes = torch.cat(all_boxes)
				all_scores = torch.cat(all_scores)
				all_labels = torch.cat(all_labels)
				
				# NEW: Single NMS across all classes
				keep = torch_nms(all_boxes, all_scores, iou_threshold=0.5)
				
				# Get top scoring boxes
				boxes_nms = all_boxes[keep]
				scores_nms = all_scores[keep]
				labels_nms = all_labels[keep]
				
				# Convert back to [class, score, x_center, y_center, width, height] format
				final_boxes = []
				for box, score, label in zip(boxes_nms, scores_nms, labels_nms):
					width = box[2] - box[0]
					height = box[3] - box[1]
					x_center = box[0] + width/2
					y_center = box[1] + height/2
					final_boxes.append([label.item(), score.item(), 
									  x_center.item(), y_center.item(), 
									  width.item(), height.item()])
			else:
				final_boxes = []
			
			# Convert tensor to image
			img = x[img_idx].permute(1, 2, 0).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			
			# Plot and save image with predictions
			plot_image(img, final_boxes, save_dir=save_dir, 
					 filename=f"epoch_{epoch}_img_{img_idx}.png")
	
	model.train()

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
##################################### JUST TO CHECK THE TRANSFORMATIONS #####################################
def save_gt_images(loader, class_labels, save_dir="gt_visualizations"):
	"""
	Saves images with ground truth bounding boxes from a DataLoader
	
	Args:
		loader: PyTorch DataLoader (train or val)
		class_labels: List of class names
		save_dir: Directory to save visualized images
	"""
	os.makedirs(save_dir, exist_ok=True)
	print(f"Saving ground truth images to {save_dir}/")
	# Color map for different classes
	colour_map = plt.get_cmap("tab20b")
	colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]
	
	for batch_idx, (images, targets) in enumerate(tqdm(loader)):
		# Convert images from tensor to numpy and denormalize
		images = images.permute(0, 2, 3, 1).cpu().numpy() # shape (batch_size, channels, height, width) to (batch_size, height, width, channels)
		images = (images * 255).astype(np.uint8)
		
		# Process each image in the batch
		for img_idx in range(images.shape[0]):
			img = images[img_idx]
			fig, ax = plt.subplots(1, figsize=(12, 12))
			ax.imshow(img)
			
			# Get boxes from the unique scale
			all_boxes = []
			target = targets[img_idx]
			anchor_indices = torch.where(target[..., 0] == 1)  # Cells with objects
			
			for anchor, i, j in zip(*anchor_indices):
				# par ex anchor = 0, i = 7, j = 15
				# Get box coordinates : x,y in [0,1] relative to the cell, w,h in [0,26] so relative to the grid size
				x, y, w, h = target[anchor, i, j, 1:5]
				class_id = target[anchor, i, j, 5].int() # in [0, 12]
				
				# Convert from grid coordinates to image coordinates
				cell_size = 1/26
				
				# Calculate box coordinates in image space
				x_center = (j + x) * cell_size # remet entre 0 et 1 relative to the grid size
				y_center = (i + y) * cell_size
				width = w * cell_size # remet entre 0 et 1 relative to the grid size
				height = h * cell_size

				all_boxes.append([
					class_id.item(),  # class
					1.0,             # confidence (ground truth)
					x_center.item(),
					y_center.item(),
					width.item(),
					height.item()
				])
			
			# Draw all ground truth boxes
			for box in all_boxes: # all_boxes est toutes les boites d'une image normalisées entre 0 et 1
				class_id, conf, x, y, w, h = box
				
				# Skip if confidence is too low (shouldn't happen for GT)
				if conf < 0.5:
					continue
				
				# Calculate coordinates
				x1 = (x - w/2) * img.shape[1] # trouve le coin supérieur gauche de la boite en pixels (416x416)
				y1 = (y - h/2) * img.shape[0]
				width = w * img.shape[1]
				height = h * img.shape[0]
				
				# Create rectangle patch
				rect = patches.Rectangle(
					(x1, y1), width, height,
					linewidth=2,
					edgecolor=colors[int(class_id)],
					facecolor="none"
				)
				ax.add_patch(rect)
				
				# Add label
				plt.text(
					x1, y1,
					s=f"{class_labels[int(class_id)]} {conf:.2f}",
					color="white",
					verticalalignment="top",
					bbox={"color": colors[int(class_id)], "pad": 0},
				)
			
			# Save image
			plt.axis("off")
			plt.savefig(
				os.path.join(save_dir, f"batch_{batch_idx}_img_{img_idx}.png"),
				bbox_inches="tight",
				pad_inches=0,
				dpi=100
			)
			plt.close()

# Usage example
#save_gt_images(
#    loader=train_loader,
#    class_labels=class_labels,
#    save_dir="train_gt_visualizations"
#)
#print(f"Saved ground truth visualizations to train_gt_visualizations/")

if load_model:
	load_checkpoint("checkpoint.pth.tar", model, optimizer, learning_rate)

# Training the model 
for e in range(1, epochs+1): 
	print("Epoch:", e) 
	training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors) 

	# Validation
	val_loss = evaluate(val_loader, model, loss_fn, scaled_anchors)
	print(f"Validation Loss: {val_loss:.4f}")
	
	# Save validation images
	save_val_images(model, val_loader, device, ANCHORS, s, save_dir="val_images", epoch=e)

	# Saving the model 
	if save_model: 
		save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
