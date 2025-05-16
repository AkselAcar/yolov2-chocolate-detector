from src.Constants import *

import numpy as np
from PIL import Image, ImageFilter
import random
from torchvision import transforms

class YoloCompatibleTransform:
	def __init__(self, resized_img_width = RESIZED_IMG_WIDTH, resized_img_height = RESIZED_IMG_HEIGHT,
				 train=True, max_rotation=10, max_translation=0.05,
				 brightness=0.3, contrast=0.3, saturation=0.1, hue=0.01,
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
		
		# ======= COLOR TRANSFORMATIONS AFTER resizing => better computationnaly =======
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
		

######################################### CHECK THE TRANSFORMATIONS ######################################################

def save_transformed_images_for_checking(loader, class_names=CLASS_NAMES, save_dir="transformation_visualizations", grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT):
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

	total_img_count = 0
	
	for batch_idx, (images, targets) in enumerate(tqdm(loader)):
		# Convert images from tensor to numpy and denormalize
		images = images.permute(0, 2, 3, 1).cpu().numpy() # shape (batch_size, channels, height, width) to (batch_size, height, width, channels)
		images = (images * 255).astype(np.uint8)
		
		# Process each image in the batch
		for img_idx in range(images.shape[0]):
			img = images[img_idx]
			fig, ax = plt.subplots(1, figsize=(12, 8))
			ax.imshow(img)
			
			# Get boxes from the unique scale
			all_boxes = []
			target = targets[img_idx]
			anchor_indices = torch.where(target[..., 0] == 1)  # Cells with objects
			
			for anchor, i, j in zip(*anchor_indices):
				# for ex anchor = 0, i = 7, j = 15
				# Get box coordinates : x,y in [0,1] relative to the cell, w,h in [0,grid_width-1] and [0, grid_height-1] so relative to the grid size
				x, y, w, h = target[anchor, i, j, 1:5]
				class_id = target[anchor, i, j, 5].int() # in [0, 12]
				
				
				# Calculate box coordinates normalized to [0,1] with respect to the resized image size
				x_center = (j + x) / grid_width # normalize to [0,1]
				y_center = (i + y) / grid_height
				width = w / grid_width
				height = h / grid_height

				all_boxes.append([
					class_id.item(),  # class
					1.0,             # confidence (ground truth so 1)
					x_center.item(),
					y_center.item(),
					width.item(),
					height.item()
				])
			
			# Draw all ground truth boxes
			for box in all_boxes:
				class_id, conf, x, y, w, h = box
				
				# Calculate coordinates
				x1 = (x - w/2) * img.shape[1] # finds the upper-left corner of the box in pixels
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
					s=f"{class_names[int(class_id)]} {conf:.2f}",
					color="white",
					verticalalignment="top",
					bbox={"color": colors[int(class_id)], "pad": 0},
				)
			
			# Save image
			plt.axis("off")
			plt.savefig(
				os.path.join(save_dir, f"img_{total_img_count}.png"),
				bbox_inches="tight",
				pad_inches=0,
				dpi=100
			)
			plt.close()

			total_img_count += 1