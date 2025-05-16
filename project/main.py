from src.utils import *
from src.Constants import *

train_transform = YoloCompatibleTransform(resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT, train=True)
test_transform = YoloCompatibleTransform(resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT, train=False)


# Read the full CSV
full_df = pd.read_csv("./src/link_images_labels_YOLO.csv")
# List of images to be used for validation (3 classic backgrounds, 5 noisy)
val_images = [
	"L1000850.JPG", "L1000785.JPG", "L1000772.JPG", "L1000894.JPG",
	"L1000899.JPG", "L1000903.JPG", "L1000910.JPG", "L1000981.JPG"
]

# Create a validation dataframe
val_df = full_df[full_df['image_name'].isin(val_images)]

# The rest is for training
train_df = full_df[~full_df['image_name'].isin(val_images)]

# Create datasets
train_dataset = Dataset(
	df=train_df,
	image_dir="./dataset_project_iapr2025/train/",
	label_dir="./src/labels/",
	anchors=ANCHORS,
	transform=train_transform
)
val_dataset = Dataset(
	df=val_df,
	image_dir="./dataset_project_iapr2025/train/",  # Same as train
	label_dir="./src/labels/",
	anchors=ANCHORS,
	transform=test_transform
)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
	val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	pin_memory=True,
)

# Grid 24x36 (stride 16)
model = YOLO_YOCO_Flexible(grid_ratio=STRIDE_TOT, scale_factor=0.5).to(DEVICE)
init_yolo_weights(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters : {num_params}")

# Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-5) 

# Defining the loss function 
loss_fn = YOLOLoss() 

min_val_loss = 100

if LOAD_MODEL:
	load_checkpoint(CHECKPOINT_FILE_LOAD, model, optimizer, LEARNING_RATE)


# Training the model
for e in range(1, NB_EPOCHS+1): 
	print("Epoch:", e) 
	training_loop(train_loader, model, optimizer, loss_fn, scaled_anchors=SCALED_ANCHORS, device=DEVICE) 

	# Validation
	val_loss = evaluate(val_loader, model, loss_fn, scaled_anchors=SCALED_ANCHORS, device=DEVICE)
	print(f"Validation Loss: {val_loss:.4f}")
	if e==1:
		min_val_loss = val_loss
	# Save validation images
	save_val_images(model, val_loader, device=DEVICE, scaled_anchors=SCALED_ANCHORS, save_dir="val_images", epoch=e, class_names = CLASS_NAMES, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT)

	if val_loss < min_val_loss:
		min_val_loss = val_loss
		# Saving the model 
		if SAVE_MODEL: 
			save_checkpoint(model, optimizer, filename=f"checkpoint_{e}.pth.tar")


test_dir = "./dataset_project_iapr2025/test"
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".JPG")])

test_dataset = YOLOTestDataset(
	img_dir=test_dir,
	filenames=test_files,
	transform=test_transform
)

test_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	pin_memory=True,
)


save_val_images(model, test_loader, device=DEVICE, scaled_anchors=SCALED_ANCHORS, save_dir="test_images", epoch=1000, class_names = CLASS_NAMES, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, resized_img_width=RESIZED_IMG_WIDTH, resized_img_height=RESIZED_IMG_HEIGHT)


generate_submission_csv_with_nms(
	model,
	test_loader,
	output_csv_path="final_submission.csv",
	conf_threshold=0.7,  # Adjust based on validation
	iou_threshold=0.4    # Same as validation NMS
)


# Usage example
save_gt_images(
	loader=train_loader,
	class_names=CLASS_NAMES,
	save_dir="train_gt_visualizations"
)
print(f"Saved ground truth visualizations to train_gt_visualizations/")