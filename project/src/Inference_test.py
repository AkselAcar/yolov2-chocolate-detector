from src.Constants import *
from src.utils import convert_cell_to_bboxes

import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import csv


class YOLOTestDataset(torch.utils.data.Dataset):
	def __init__(self, img_dir, filenames, transform=None):
		self.img_dir = img_dir
		self.filenames = filenames
		self.transform = transform

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		fname = self.filenames[idx]
		img_path = os.path.join(self.img_dir, fname)
		image = Image.open(img_path).convert("RGB")

		if self.transform:
			image_dict = self.transform(image, [])
			image = image_dict["image"]
		else:
			image = T.ToTensor()(image)

		#dummy_boxes = torch.zeros((GRID_HEIGHT, GRID_WIDTH, 2 * 5 + NUM_CLASSES))
		return image, fname # dummy_boxes, fname
	

CLASS_NAMES_FOR_SUBMISSION = [ 
	"Jelly White","Jelly Milk","Jelly Black","Amandina","Crème brulée",
	"Triangolo","Tentation noir","Comtesse","Noblesse","Noir authentique",
	"Passion au lait","Arabia","Stracciatella"
]

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
		writer.writerow(["id"] + CLASS_NAMES_FOR_SUBMISSION)
		writer.writerows(predictions)

	print(f" Submission file saved to: {output_csv_path}")

