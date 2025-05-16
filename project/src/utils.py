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
	

def convert_cell_to_bboxes(prediction, scaled_anchors=SCALED_ANCHORS,
						   grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
						   resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT,
						   num_classes=NUM_CLASSES, conf_threshold = 0.6):
	"""
	Function to convert a prediction (output) of the model to a tensor with all the bounding boxes of dimension 6
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
	

def save_val_images(model, loader, device, scaled_anchors, save_dir, epoch, class_names = CLASS_NAMES,
					grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
					resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT):
	"""
	Function to save to apply the model on images of a loader, and plot/save them by calling the function plot_image
	"""
	
	model.eval()
	os.makedirs(save_dir, exist_ok=True)
	for file_path in glob.glob(os.path.join(save_dir, "*.png")):
		os.remove(file_path)
	global_img_idx = 0
	with torch.no_grad():
		for _, (x, _) in enumerate(loader):
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




################################################### TRAINING AND EVALUATION ############################################################
def train_one_epoch(loader, model, optimizer, loss_fn, scaled_anchors=SCALED_ANCHORS, device=DEVICE):
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

