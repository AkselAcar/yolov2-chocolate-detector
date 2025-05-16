from src.utils import *

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