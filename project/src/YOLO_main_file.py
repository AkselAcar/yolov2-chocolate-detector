import pandas as pd
import os

def convert_labels(csv_path, output_dir, img_size=640):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        filename, cx, cy, w, h, cls = row
        orig_w, orig_h = 6000, 4000
        
        # Calculate scaling factors
        scale_w = img_size / orig_w
        scale_h = img_size / orig_h
        
        # Scale coordinates
        scaled_cx = cx * scale_w
        scaled_cy = cy * scale_h
        scaled_w = w * scale_w
        scaled_h = h * scale_h
        
        # Normalize
        norm_cx = scaled_cx / img_size
        norm_cy = scaled_cy / img_size
        norm_w = scaled_w / img_size
        norm_h = scaled_h / img_size
        
        # Write to file
        txt_path = os.path.join(output_dir, filename.replace('.JPG', '.txt'))
        with open(txt_path, 'a') as f:
            f.write(f"{cls} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")

convert_labels('../labels_YOLO.csv', 'labels')

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class ChocolateDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.JPG')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size)) / 255.0
        
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.JPG', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cx, cy, w, h, int(cls)])
        
        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(boxes) if boxes else torch.zeros((0, 5))
    


import torch
import torch.nn as nn
import numpy as np

# Configuration
ANCHORS = np.array([[40,74], [50,61], [52,78],
                   [57,86], [41,140], [63,94],
                   [90,67], [49,127], [70,103]])  # Your anchors
STRIDE = 8  # 640px / 80x80 grid
NUM_CLASSES = 13
IMG_SIZE = 640

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, depthwise=False, stride=1):
        super().__init__()
        self.depthwise = depthwise
        
        if self.depthwise:
            # Depthwise convolution
            self.dw_conv = nn.Conv2d(in_c, in_c, kernel_size=3, 
                                   stride=stride, padding=1, 
                                   groups=in_c, bias=False)
            self.bn1 = nn.BatchNorm2d(in_c)
            # Pointwise convolution
            self.pw_conv = nn.Conv2d(in_c, out_c, kernel_size=1, 
                                   stride=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_c)
        else:
            # Standard convolution
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, 
                                stride=stride, padding=1, 
                                bias=False)
            self.bn = nn.BatchNorm2d(out_c)
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        if self.depthwise:
            x = self.dw_conv(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.pw_conv(x)
            x = self.bn2(x)
            x = self.act(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
        return x
    
class YOLOv5(nn.Module):
    def __init__(self, num_classes=13, anchors=None):
        super().__init__()
        assert len(anchors) == 9, "Must provide exactly 9 anchors"
        self.num_anchors = len(anchors) if anchors is not None else 9  # Use your 9 anchors
        
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256),
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 1024),
        )
        
        self.head = nn.Sequential(
            ConvBlock(1024, 512),
            # Ensure output channels match num_anchors * (5 + num_classes)
            nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1)  
        )
        
        # Register anchors properly
        if anchors is not None:
            if isinstance(anchors, np.ndarray):  # Handle NumPy inputs
                anchors = anchors.tolist()
            self.register_buffer('anchors', torch.tensor(anchors).float())
        else:
            # Default anchors (replace with your calculated values)
            self.register_buffer('anchors', torch.tensor([
                [40,74], [50,61], [52,78],
                [57,86], [41,140], [63,94],
                [90,67], [49,127], [70,103]
            ]).float())

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
"""
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.obj_scale = 1.0
        self.noobj_scale = 0.5
        
    def forward(self, preds, targets, model):
        device = preds.device
        bs, _, h, w = preds.shape
        
        # Convert predictions to (bs, anchors, h, w, 5+13)
        preds = preds.view(bs, model.num_anchors, 5 + NUM_CLASSES, h, w).permute(0,1,3,4,2)
        
        # Initialize losses
        obj_loss = torch.tensor(0., device=device)
        cls_loss = torch.tensor(0., device=device)
        box_loss = torch.tensor(0., device=device)

        # Get predictions
        x = torch.sigmoid(preds[..., 0])  # Center x
        y = torch.sigmoid(preds[..., 1])  # Center y
        w_pred = preds[..., 2]  # Width
        h_pred = preds[..., 3]  # Height
        obj = preds[..., 4]  # Objectness
        cls = preds[..., 5:]  # Class scores

        for bi in range(bs):
            # Get targets for this image (shape: [N, 5] - cx, cy, w, h, class)
            img_targets = targets[bi]
            if img_targets.shape[0] == 0:  # No targets
                continue
                
            # Convert normalized coordinates to grid coordinates
            target_xy = img_targets[:, :2] * torch.tensor([w, h], device=device)
            target_wh = img_targets[:, 2:4] * torch.tensor([w, h], device=device)
            target_cls = img_targets[:, 4].long()
            
            # Find best anchor for each target
            ious = wh_iou(target_wh, model.anchors)
            best_anchors = ious.argmax(dim=1)
            
            # Assign targets to grid cells
            gi, gj = target_xy.long().T
            
            # Filter valid grid positions
            #valid_mask = (gi >= 0) & (gi < w) & (gj >= 0) & (gj < h)
            
            #for idx in torch.where(valid_mask)[0]:
            #    a = best_anchors[idx]
            #    i, j = gi[idx], gj[idx]
            #    cls_idx = target_cls[idx]
            #    
            #    # Box coordinates loss
            #    tx = target_xy[idx, 0] - i.float()
            #    ty = target_xy[idx, 1] - j.float()
            #    tw = torch.log(target_wh[idx, 0] / model.anchors[a, 0] + 1e-16)
            #    th = torch.log(target_wh[idx, 1] / model.anchors[a, 1] + 1e-16)
            #    
            #    box_loss += self.mse(x[bi, a, j, i], tx) + \
            #              self.mse(y[bi, a, j, i], ty) + \
            #              self.mse(w_pred[bi, a, j, i], tw) + \
            #              self.mse(h_pred[bi, a, j, i], th)
            #    
            #    # Objectness loss
            #    obj_target = torch.ones_like(obj[bi, a, j, i])
            #    obj_loss += self.bce(obj[bi, a, j, i], obj_target)
            #    
            #    # Classification loss
            #    cls_target = torch.zeros_like(cls[bi, a, j, i])
            #    cls_target[cls_idx] = 1.0
            #    cls_loss += self.bce(cls[bi, a, j, i], cls_target)
                
            # Process all valid grid cells simultaneously
            valid_mask = (gi >= 0) & (gi < w) & (gj >= 0) & (gj < h)
            valid_gi = gi[valid_mask]
            valid_gj = gj[valid_mask]
            valid_a = best_anchors[valid_mask]

            # Vectorized loss calculation
            tx = target_xy[valid_mask, 0] - valid_gi.float()
            ty = target_xy[valid_mask, 1] - valid_gj.float()
            tw = torch.log(target_wh[valid_mask, 0] / model.anchors[valid_a, 0] + 1e-16)
            th = torch.log(target_wh[valid_mask, 1] / model.anchors[valid_a, 1] + 1e-16)

            box_loss = self.mse(x[bi, valid_a, valid_gj, valid_gi], tx).mean() + \
                       self.mse(y[bi, valid_a, valid_gj, valid_gi], ty).mean() + \
                       self.mse(w_pred[bi, valid_a, valid_gj, valid_gi], tw).mean() + \
                       self.mse(h_pred[bi, valid_a, valid_gj, valid_gi], th).mean()
            
        total_loss = (box_loss + self.obj_scale * obj_loss + cls_loss) / bs
        return total_loss
"""
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # Track individual losses
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.obj_scale = 1.0
        self.noobj_scale = 0.5

    def forward(self, preds, targets, model):
        device = preds.device
        bs, _, h, w = preds.shape
        
        # Reshape predictions
        preds = preds.view(bs, model.num_anchors, 5 + NUM_CLASSES, h, w).permute(0,1,3,4,2)
        
        # Initialize losses
        obj_loss = torch.tensor(0., device=device)
        cls_loss = torch.tensor(0., device=device)
        box_loss = torch.tensor(0., device=device)

        # Split predictions
        x = torch.sigmoid(preds[..., 0])
        y = torch.sigmoid(preds[..., 1])
        w_pred = preds[..., 2]
        h_pred = preds[..., 3]
        obj = preds[..., 4]
        cls = preds[..., 5:]

        for bi in range(bs):
            img_targets = targets[bi]
            if img_targets.shape[0] == 0: continue  # Skip empty images
            
            # Convert targets to grid coordinates
            target_xy = img_targets[:, :2] * torch.tensor([w, h], device=device)
            target_wh = img_targets[:, 2:4] * torch.tensor([w, h], device=device)
            target_cls = img_targets[:, 4].long()
            
            # Find best anchors
            ious = wh_iou(target_wh, model.anchors)
            best_anchors = ious.argmax(dim=1)
            
            # Grid indices and validity check
            gi, gj = target_xy.long().T
            valid_mask = (gi >= 0) & (gi < w) & (gj >= 0) & (gj < h)
            valid_gi, valid_gj = gi[valid_mask], gj[valid_mask]
            valid_a = best_anchors[valid_mask]
            
            if not valid_mask.any(): continue  # Skip if no valid targets

            # Box target calculations
            tx = target_xy[valid_mask, 0] - valid_gi.float()
            ty = target_xy[valid_mask, 1] - valid_gj.float()
            tw = torch.log(target_wh[valid_mask, 0]/model.anchors[valid_a, 0] + 1e-16)
            th = torch.log(target_wh[valid_mask, 1]/model.anchors[valid_a, 1] + 1e-16)

            # ------ Vectorized Loss Calculations ------
            # Box loss
            box_preds = [
                x[bi, valid_a, valid_gj, valid_gi],
                y[bi, valid_a, valid_gj, valid_gi],
                w_pred[bi, valid_a, valid_gj, valid_gi],
                h_pred[bi, valid_a, valid_gj, valid_gi]
            ]
            box_targets = [tx, ty, tw, th]
            
            box_loss += sum(self.mse(p, t).sum() for p, t in zip(box_preds, box_targets))

            # Objectness loss
            obj_preds = obj[bi, valid_a, valid_gj, valid_gi]
            obj_loss += self.bce(obj_preds, torch.ones_like(obj_preds)).sum()

            # Classification loss
            cls_preds = cls[bi, valid_a, valid_gj, valid_gi]
            cls_targets = torch.zeros_like(cls_preds)
            cls_targets[torch.arange(len(cls_preds)), target_cls[valid_mask]] = 1.0
            cls_loss += self.bce(cls_preds, cls_targets).sum()

        return (box_loss + self.obj_scale * obj_loss + cls_loss) / bs
    
def wh_iou(wh1, wh2):
    # Returns IoU matrix of shape (targets, anchors)
    wh1 = wh1.unsqueeze(1)  # (n, 1, 2)
    wh2 = wh2.unsqueeze(0)  # (1, m, 2)
    inter = torch.min(wh1[..., 0], wh2[..., 0]) * torch.min(wh1[..., 1], wh2[..., 1])
    union = (wh1[..., 0] * wh1[..., 1]) + (wh2[..., 0] * wh2[..., 1]) - inter
    return inter / union
"""
def wh_iou(wh1, wh2):
    # wh1: (N, 2), wh2: (M, 2)
    min_w = torch.min(wh1[:, None, 0], wh2[None, :, 0])
    min_h = torch.min(wh1[:, None, 1], wh2[None, :, 1])
    inter = min_w * min_h
    union = wh1[:, None, 0]*wh1[:, None, 1] + wh2[None, :, 0]*wh2[None, :, 1] - inter
    return inter / union  # (N, M)
"""
# Initialize model with explicit anchors
model = YOLOv5(num_classes=13, anchors=ANCHORS)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = YOLOLoss()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def box_iou(boxes1, boxes2):
    """
    Calculate intersection over union (IoU) between two sets of boxes in (x1, y1, x2, y2) format
    
    Args:
        boxes1: Tensor[N, 4]
        boxes2: Tensor[M, 4]
    
    Returns:
        iou: Tensor[N, M]
        """
    # Get areas of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Find intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Calculate union area
    union = area1[:, None] + area2 - inter

    # Return IoU
    return inter / (union + 1e-16)

def decode_predictions(preds, model, conf_thresh=0.25, iou_thresh=0.5):
    """Convert model output to bounding boxes with NMS"""
    # preds: (bs, anchors*(5+13), h, w)
    bs, _, h, w = preds.shape
    device = preds.device
    
    # Reshape with correct number of anchors
    preds = preds.view(bs, model.num_anchors, 5 + NUM_CLASSES, h, w).permute(0,1,3,4,2)
    
    # Create grid with proper dimensions
    grid_x = torch.arange(w, device=device).view(1, 1, 1, -1)
    grid_y = torch.arange(h, device=device).view(1, 1, -1, 1)
    
    # Decode boxes using model's registered anchors
    boxes = torch.zeros_like(preds[..., :4])
    boxes[..., 0] = (torch.sigmoid(preds[..., 0]) + grid_x) / w  # cx
    boxes[..., 1] = (torch.sigmoid(preds[..., 1]) + grid_y) / h  # cy
    boxes[..., 2] = torch.exp(preds[..., 2]) * model.anchors[:, 0].view(1, -1, 1, 1, 1) / IMG_SIZE
    boxes[..., 3] = torch.exp(preds[..., 3]) * model.anchors[:, 1].view(1, -1, 1, 1, 1) / IMG_SIZE
    # Convert to (x1, y1, x2, y2)
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2]/2  # x1
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3]/2  # y1
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2]/2  # x2
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3]/2  # y2
    
    # Combine scores
    obj_scores = preds[..., 4].sigmoid()  # Objectness
    class_scores = preds[..., 5:].sigmoid().max(dim=-1)[0]  # Class confidence
    total_scores = obj_scores * class_scores
    
    # Filter and NMS
    all_boxes = []
    for bi in range(bs):
        batch_boxes = []
        mask = total_scores[bi] > conf_thresh
        for idx in torch.where(mask):
            a, i, j = idx
            box = boxes_xyxy[bi, a, j, i].cpu().detach().numpy()
            score = total_scores[bi, a, j, i].item()
            cls = preds[bi, a, j, i, 5:].argmax().item()
            batch_boxes.append([*box, score, cls])
        
        # Non-Max Suppression
        if batch_boxes:
            boxes_tensor = torch.tensor([b[:4] for b in batch_boxes])
            scores = torch.tensor([b[4] for b in batch_boxes])
            classes = torch.tensor([b[5] for b in batch_boxes])
            
            keep = []
            for c in classes.unique():
                mask = classes == c
                boxes_c = boxes_tensor[mask]
                scores_c = scores[mask]
                
                # Sort by score
                sorted_ids = torch.argsort(scores_c, descending=True)
                while sorted_ids.numel() > 0:
                    keep.append(mask.nonzero()[sorted_ids[0]])
                    if sorted_ids.numel() == 1:
                        break
                    ious = box_iou(boxes_c[sorted_ids[0]].unsqueeze(0),
                                   boxes_c[sorted_ids[1:]])
                    remove_ids = torch.where(ious > iou_thresh)[1] + 1
                    sorted_ids = sorted_ids[~torch.isin(sorted_ids, remove_ids)]
            
            if keep:
                keep = torch.cat(keep)
                batch_boxes = [batch_boxes[k] for k in keep]
        
        all_boxes.append(batch_boxes)
    
    return all_boxes

def plot_predictions(images, pred_boxes, true_boxes, epoch):
    """Plot predictions vs ground truth with proper tensor handling"""
    plt.figure(figsize=(15, 10))
    for i in range(min(4, len(images))):  # Plot first 4 images
        plt.subplot(2, 2, i+1)
        
        # Convert image tensor
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Ensure valid image range
        
        plt.imshow(img)
        ax = plt.gca()
        
        # Plot ground truth (already in normalized coordinates)
        for box in true_boxes[i]:
            if box.shape[0] == 0:
                continue
            cx, cy, w, h, cls = box
            x = (cx - w/2) * IMG_SIZE
            y = (cy - h/2) * IMG_SIZE
            rect = patches.Rectangle((x, y), w*IMG_SIZE, h*IMG_SIZE,
                                    linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f'GT: {int(cls)}', color='green')
        
        # Plot predictions
        for box in pred_boxes[i]:
            x1, y1, x2, y2, conf, cls = box
            x = x1 * IMG_SIZE
            y = y1 * IMG_SIZE
            w = (x2 - x1) * IMG_SIZE
            h = (y2 - y1) * IMG_SIZE
            rect = patches.Rectangle((x, y), w, h,
                                   linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f'P: {int(cls)} ({conf:.2f})', color='red')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'output/training_plots/epoch_{epoch+1}_predictions.png')
    plt.close()

from tqdm.auto import tqdm

def train(model, dataloader, optimizer, epochs=100):
    model.train()
    device = next(model.parameters()).device
    
    # Create output directory
    os.makedirs('output/training_plots', exist_ok=True)
    
    # Main training loop with progress bars
    with tqdm(range(epochs), desc="Total Training", unit="epoch") as epoch_pbar:
        for epoch in epoch_pbar:
            total_loss = 0
            model.train()
            
            # Batch processing
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch") as batch_pbar:
                for batch_idx, (images, targets) in enumerate(batch_pbar):
                    # Move data to device
                    images = images.to(device)
                    targets = [t.to(device) for t in targets]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, targets, model)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(dataloader)
            epoch_pbar.set_postfix({'avg_loss': f"{avg_loss:.4f}"})
            
            # Visualize predictions after each epoch
            with torch.no_grad():
                model.eval()
                # Get sample batch
                sample_images, sample_targets = next(iter(dataloader))
                sample_images = sample_images.to(device)
                
                # Run inference
                sample_outputs = model(sample_images)
                pred_boxes = decode_predictions(sample_outputs, model)
                
                # Convert to CPU for plotting
                sample_images_cpu = sample_images.cpu()
                sample_targets_cpu = [t.cpu() for t in sample_targets]
                
                # Plot and save
                plot_predictions(
                    sample_images_cpu,
                    pred_boxes,
                    sample_targets_cpu,
                    epoch
                )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'output/checkpoint_epoch_{epoch+1}.pth')
    
    print("Training completed!")


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = [l for l in labels if l.shape[0] > 0]
    return images, labels

from torch.utils.data import DataLoader
# Dataset (STILL NEEDED)
dataset = ChocolateDataset('../dataset_project_iapr2025/train', 'labels')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)



# Model + Anchors (NEW CODE)
model = YOLOv5(num_classes=13, anchors=ANCHORS)  # Includes your anchors
loss_fn = YOLOLoss()  # Uses anchors in calculations
# Test with dummy input
dummy = torch.randn(1, 3, 640, 640)
output = model(dummy)
print("Output shape:", output.shape)  # Should be (1, 162, 80, 80)
pred_boxes = decode_predictions(output, model)
print("Decoded boxes:", len(pred_boxes[0]))  # Should show multiple boxes



# Training Loop (UPDATED)
train(model, dataloader, optimizer)  # Uses both dataset and anchors


