from src.Constants import *
from src.utils import iou
import numpy as np
import torch
from PIL import Image

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
					# Setting this to -1 will enable in the loss not to penalize if the model has found a good anchor but not the best
					target[anchor_idx, i, j, 0] = -1
		
		# Return the image and the target
		return image, target