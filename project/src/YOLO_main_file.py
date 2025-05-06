import pandas as pd
import os

def convert_labels(csv_path, output_dir, img_size=640):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        filename, cx, cy, w, h, cls = row
        orig_w, orig_h = 6000, 4000
        
        # Calculate scaling factors
        scale_w = img_size / orig_w # so 640/6000 = 0.10666666666666667
        scale_h = img_size / orig_h # so 640/4000 = 0.16
        
        # Scale coordinates on resized image
        scaled_cx = cx * scale_w
        scaled_cy = cy * scale_h
        scaled_w = w * scale_w
        scaled_h = h * scale_h
        
        # Normalize coordinates to [0, 1] !!!
        norm_cx = scaled_cx / img_size
        norm_cy = scaled_cy / img_size
        norm_w = scaled_w / img_size
        norm_h = scaled_h / img_size
        
        # Write to file : labels/filename.txt => one file per image !!!
        txt_path = os.path.join(output_dir, filename.replace('.JPG', '.txt'))
        with open(txt_path, 'a') as f:
            f.write(f"{cls} {norm_cx} {norm_cy} {norm_w} {norm_h}\n")

#convert_labels('../labels_YOLO.csv', 'labels') # ATTENTION TO RUN ONCE PSK SINON CA RAJOUTE DES LIGNES FAUDRAIT LE METTRE DANS UN IF

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class ChocolateDataset(Dataset):
    def __init__(self, img_dir, label_dir, filenames, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.filenames = filenames
        
        # Filter existing files
        self.image_files = [f for f in self.filenames 
                          if os.path.exists(os.path.join(img_dir, f)) and 
                          os.path.exists(os.path.join(label_dir, f.replace('.JPG', '.txt')))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load boxes (original coordinates)
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.JPG', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cx, cy, w, h, int(cls)])
        
        # Convert to PIL for transformations
        image_pil = Image.fromarray(image)
        
        # Apply transformations
        if self.transform:
            image_tensor, boxes = self.transform(image_pil, boxes)
        else:
            # Convert to tensor directly
            image_tensor = TF.to_tensor(image_pil)
        
        # Resize and normalize
        image_tensor = TF.resize(image_tensor, [self.img_size, self.img_size])
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image_tensor, torch.tensor(boxes) if boxes is not None else torch.zeros((0, 5))
    
from torchvision import transforms as T
class RandomAffineWithBoxes:
    def __init__(self, degrees=15, translate=(0.1, 0.1), color_jitter=(0.2, 0.2, 0.2, 0.1), img_size=640):
        self.degrees = degrees
        self.translate = translate
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(
            brightness=color_jitter[0],
            contrast=color_jitter[1],
            saturation=color_jitter[2],
            hue=color_jitter[3]
        )
        self.to_tensor = T.ToTensor()

    def __call__(self, image, boxes):
        # Store original in case we need to revert
        original_image = image.copy()
        original_boxes = boxes.copy()
        
        # Apply color jitter first
        image = self.color_jitter(image)
        
        # Random rotation and translation parameters
        angle = np.random.uniform(-self.degrees, self.degrees)
        max_trans_x = int(self.translate[0] * self.img_size)
        max_trans_y = int(self.translate[1] * self.img_size)
        translate = (np.random.randint(-max_trans_x, max_trans_x),
                     np.random.randint(-max_trans_y, max_trans_y))

        # Apply affine transformation to image
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=0,
            fill=(127, 127, 127)  # Gray padding for new areas
        )

        # Convert boxes to tensor if needed
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        boxes = np.array(boxes)

        new_boxes = []
        for box in boxes:
            if len(box) == 0:
                continue
                
            cls_id = box[4] if len(box) > 4 else 0
            cx, cy, w, h = box[:4]

            # Convert normalized coordinates to absolute pixels
            cx_pix = cx * self.img_size
            cy_pix = cy * self.img_size
            w_pix = w * self.img_size
            h_pix = h * self.img_size

            # Generate original bounding box corners
            corners = torch.tensor([
                [cx_pix - w_pix/2, cy_pix - h_pix/2],  # Top-left
                [cx_pix + w_pix/2, cy_pix - h_pix/2],  # Top-right
                [cx_pix + w_pix/2, cy_pix + h_pix/2],  # Bottom-right
                [cx_pix - w_pix/2, cy_pix + h_pix/2],  # Bottom-left
            ], dtype=torch.float32)

            # Rotation matrix (angle in radians)
            angle_rad = np.deg2rad(angle)
            rot_matrix = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ], dtype=torch.float32)

            # Translation vector
            translation = torch.tensor(translate, dtype=torch.float32)

            # Rotate corners around image center
            center = torch.tensor([self.img_size/2, self.img_size/2])
            rotated_corners = (corners - center) @ rot_matrix.T + center

            # Apply translation
            translated_corners = rotated_corners + translation

            # Clamp to image boundaries
            translated_corners[:, 0] = translated_corners[:, 0].clamp(0, self.img_size)
            translated_corners[:, 1] = translated_corners[:, 1].clamp(0, self.img_size)

            # Calculate new bounding box
            xmin, _ = translated_corners[:, 0].min(dim=0)
            xmax, _ = translated_corners[:, 0].max(dim=0)
            ymin, _ = translated_corners[:, 1].min(dim=0)
            ymax, _ = translated_corners[:, 1].max(dim=0)

            # Convert back to normalized coordinates
            new_cx = ((xmin + xmax) / 2) / self.img_size
            new_cy = ((ymin + ymax) / 2) / self.img_size
            new_w = (xmax - xmin) / self.img_size
            new_h = (ymax - ymin) / self.img_size

            # Check validity
            if (new_w > 0.01 and new_h > 0.01 and 
                0 <= new_cx <= 1 and 0 <= new_cy <= 1):
                new_boxes.append([new_cx, new_cy, new_w, new_h, cls_id])

        # Revert if all boxes were invalid
        if len(new_boxes) == 0:
            return TF.to_tensor(original_image), original_boxes

        return self.to_tensor(image), torch.tensor(new_boxes)
    
class ValTransform:
    def __call__(self, image, boxes):
        return TF.to_tensor(image), torch.tensor(boxes) if boxes is not None else torch.zeros((0, 5))
    

import torch
import torch.nn as nn
import numpy as np

# Configuration
ANCHORS = np.array([[40,74], [50,61], [52,78],
                   [57,86], [41,140], [63,94],
                   [90,67], [49,127], [70,103]])  # Anchors with respect to the 640x640 image size (from kmeans)
STRIDE = 16  # 640px / 40x40 grid
NUM_CLASSES = 13
IMG_SIZE = 640

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        # Standard convolution
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, 
                            stride=stride, padding=1, 
                            bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class YOLOv5(nn.Module):
    def __init__(self, num_classes=13, anchors=None):
        super().__init__()
        assert len(anchors) == 9, "Must provide exactly 9 anchors"
        self.num_anchors = len(anchors) if anchors is not None else 9  # Use your 9 anchors
        anchors = np.array(anchors) / STRIDE  # Convert to grid cell units so between 
        
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
            ConvBlock(1024, 512, stride=2),
            # Ensure output channels match num_anchors * (5 + num_classes)
            nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1)  
        )
        with torch.no_grad():
            self.head[-1].bias.fill_(0.0)  # Bias for objectness to ~0.5 after sigmoid
            self.head[-1].weight.fill_(0.01)

        # Register anchors properly
        if anchors is not None:
            if isinstance(anchors, np.ndarray):  # Handle NumPy inputs
                anchors = anchors.tolist()
            self.register_buffer('anchors', torch.tensor(anchors).float())
        else:
            # Default anchors
            self.register_buffer('anchors', torch.tensor(ANCHORS/STRIDE).float())

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
    # input: (bs, 3, 640, 640)
    # output: (bs, 9*(5+13), 40, 40) = (bs, 162, 40, 40)


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # Track individual losses Pour les boîtes (coordonnées)
        self.bce = nn.BCEWithLogitsLoss(reduction='none') # Pour objectness et classification pour les sorties de classification binaire ou multi-label (non sigmoidées).
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.obj_scale = 5.0
        self.noobj_scale = 0.5

    def forward(self, preds, targets, model):
        device = preds.device
        bs, _, h, w = preds.shape
        
        # Reshape predictions : preds est de forme [B, A*(5+C), H, W] = [8, 162, 40, 40] to [B, A, H, W, 5+C] = [8, 9, 40, 40, 18]
        preds = preds.view(bs, model.num_anchors, 5 + NUM_CLASSES, h, w).permute(0,1,3,4,2)
        
        # Initialize losses
        cls_loss = torch.tensor(0., device=device)
        box_loss = torch.tensor(0., device=device)

        # Split predictions
        x = torch.sigmoid(preds[..., 0]) # shape [B, A, H, W] => need sigmoid in inference
        y = torch.sigmoid(preds[..., 1])
        w_pred = preds[..., 2]
        h_pred = preds[..., 3]
        obj = preds[..., 4] # objectness score = confidence score about the presence of an object
        cls = preds[..., 5:]

        # Create target_obj tensor for objectness (0 = no object, 1 = object)
        target_obj = torch.zeros_like(obj, device=device) # shape [B, A, H, W]

        for bi in range(bs): # loop over batch size because targets are not stacked
            img_targets = targets[bi] # shape is [N, 5] where N is the number of targets for image bi [cx, cy, w, h, cls] with cls in [0, 12] and cx, cy, w, h in [0, 1]
            if img_targets.shape[0] == 0: 
                continue  # Skip empty images

            # Rescale targets to grid coordinates
            target_xy = img_targets[:, :2] * torch.tensor([w, h], device=device) # shape [N, 2] where N is the number of targets for image bi. They belong to the interval [0, 40] in continuous space
            target_wh = img_targets[:, 2:4] * torch.tensor([w, h], device=device) # shape [N, 2] where N is the number of targets for image bi. They belong to the interval [0, 40]
            target_cls = img_targets[:, 4].long() # shape [N] where N is the number of targets for image bi. They belong to the interval [0, 12]
            
            # Find best anchors
            ious = wh_iou(target_wh, model.anchors) # shape [N, 9] where N is the number of targets for image bi and 9 is the number of anchors
            best_anchors = ious.argmax(dim=1) # shape [N] where N is the number of targets for image bi. best_anchors[i] is the best anchor for target i

            # Grid indices and validity check ( the 4 lines should be useless)
            gi, gj = target_xy.long().T # they have shape [N] where N is the number of targets for image bi, they are the indices of the grid cells where the targets are located (so discrete space)
            valid_mask = (gi >= 0) & (gi < w) & (gj >= 0) & (gj < h)
            valid_gi, valid_gj = gi[valid_mask], gj[valid_mask]
            valid_a = best_anchors[valid_mask] 
            
            if valid_mask.any():
                # Set target_obj to 1 for positive anchors. Recall : target_obj is of shape [B, A, H, W]
                target_obj[bi, valid_a, valid_gj, valid_gi] = 1.0 # so in the tensor we put a 1 in every cell of the grid where there is an object (target) in the image bi, at the best anchor (valid_a).


                # Box target calculations
                tx = target_xy[valid_mask, 0] - valid_gi.float() # offset between 0 and 1 in the grid cells where there are objects (so discrete space). Shape [N] where N is the number of targets for image bi
                ty = target_xy[valid_mask, 1] - valid_gj.float()
                tw = torch.log(target_wh[valid_mask, 0]/model.anchors[valid_a, 0] + 1e-16) # logarithm of the ratio between the target width (in continuous [0, 40]) and the anchor width (in continuous [0, 40]). Shape [N] where N is the number of targets for image bi
                th = torch.log(target_wh[valid_mask, 1]/model.anchors[valid_a, 1] + 1e-16)

                # ------ Vectorized Loss Calculations ------
                # Box loss
                box_preds = [
                    x[bi, valid_a, valid_gj, valid_gi], # x y w h are pred of shape [B, A, H, W]
                    y[bi, valid_a, valid_gj, valid_gi],
                    w_pred[bi, valid_a, valid_gj, valid_gi],
                    h_pred[bi, valid_a, valid_gj, valid_gi]
                ] # shape : 4 x [N] where N is the number of targets for image bi.

                box_targets = [tx, ty, tw, th]
                
                box_loss += sum(self.mse(p, t).sum() for p, t in zip(box_preds, box_targets)) # sum of 4 losses mse of each box coordinate (x, y, w, h) between pred and target
                # this will make the 4 first indices in the 5 of the output of the model (bs, 9*(5+13), 40, 40) close to the target (tx, ty, tw, th) in the grid cell (valid_gi, valid_gj) for the best anchor (valid_a)

                # Classification loss
                cls_preds = cls[bi, valid_a, valid_gj, valid_gi] # shape [N, 13] where N is the number of targets for image bi
                cls_targets = torch.zeros_like(cls_preds)
                cls_targets[torch.arange(len(cls_preds)), target_cls[valid_mask]] = 1.0 # shape [N, 13] where N is the number of targets for image bi with 1 in the column of the class of the target and 0 elsewhere
                #cls_loss += self.bce(cls_preds, cls_targets).sum()
                cls_loss += self.ce(cls_preds, target_cls[valid_mask])
            
        # Objectness loss with scaling for positives and negatives
        obj_loss = self.bce(obj, target_obj) # obj is of shape [B, A, H, W] and corresponds to the objectness score (confidence score) of the model for each anchor and each grid cell. target_obj is of shape [B, A, H, W] and corresponds to the presence of an object (1) or not (0) in the grid cell at the best anchor.
        # this will make the 5th index in the 5 of the output of the model (bs, 9*(5+13), 40, 40) close to 1 if there is an object in the grid cell (valid_gi, valid_gj) for the best anchor (valid_a) and close to 0 otherwise.
        
        pos_loss = (obj_loss * target_obj).sum() * self.obj_scale # target_obj == 1 → position avec un objet; target_obj == 0 → position sans objet
        neg_loss = (obj_loss * (1 - target_obj)).sum() * self.noobj_scale
        obj_loss = pos_loss + neg_loss
        #obj_loss = (obj_loss * (target_obj * self.obj_scale + (1 - target_obj) * self.noobj_scale)).sum()
    
        # Total loss
        total_loss = (box_loss + obj_loss + cls_loss) / bs

        return total_loss
    
def wh_iou(wh1, wh2):
    # Returns IoU matrix of shape (targets, anchors)  # wh1: (N, 2), wh2: (M, 2)
    wh1 = wh1.unsqueeze(1)  # (n, 1, 2)
    wh2 = wh2.unsqueeze(0)  # (1, m, 2)
    inter = torch.min(wh1[..., 0], wh2[..., 0]) * torch.min(wh1[..., 1], wh2[..., 1])
    union = (wh1[..., 0] * wh1[..., 1]) + (wh2[..., 0] * wh2[..., 1]) - inter
    return inter / union


from tqdm.auto import tqdm
from torchvision.ops import nms

def decode_predictions(preds, model, img_size=640, conf_threshold=0.2, iou_threshold=0.5): # preds = (bs, 9*(5+13), 40, 40)
    device = preds.device
    stride = img_size // preds.shape[2]  # Calculate stride based on grid size
    bs, _, h_grid, w_grid = preds.shape
    num_anchors = model.num_anchors
    num_classes = NUM_CLASSES

    # Reshape predictions to (B, A, H, W, 5+C)
    preds = preds.view(bs, num_anchors, 5 + num_classes, h_grid, w_grid).permute(0, 1, 3, 4, 2)
    anchors = model.anchors.view(1, num_anchors, 1, 1, 2)  # Reshape anchors for broadcasting

    # Generate grid coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(h_grid, device=device), 
                                    torch.arange(w_grid, device=device), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, h, w, 2)
    # on a Crée une grille de coordonnées (i,j) pour chaque cellule de la grille.

    # Split predictions
    tx = torch.sigmoid(preds[..., 0]) # shape [B, A, H, W]
    ty = torch.sigmoid(preds[..., 1])
    tw = preds[..., 2]
    th = preds[..., 3]
    obj = torch.sigmoid(preds[..., 4])
    cls = torch.sigmoid(preds[..., 5:])

    # Calculate box coordinates
    cx = (grid[..., 0] + tx) * stride / img_size
    cy = (grid[..., 1] + ty) * stride / img_size
    width = (anchors[..., 0] * torch.exp(tw)) * stride / img_size
    height = (anchors[..., 1] * torch.exp(th)) * stride / img_size

    # Combine and reshape
    boxes = torch.stack([cx, cy, width, height], dim=-1).view(bs, -1, 4)
    conf = (obj * cls.max(dim=-1)[0]).view(bs, -1)
    cls_ids = cls.argmax(dim=-1).view(bs, -1)

    
    # Apply confidence threshold
    batch_boxes = []
    for b in range(bs):
        mask = conf[b] >= conf_threshold
        filtered = torch.cat([boxes[b][mask], 
                            conf[b][mask].unsqueeze(1), 
                            cls_ids[b][mask].unsqueeze(1)], dim=1)
        if filtered.shape[0] > 0:
            keep = nms(filtered[:, :4], filtered[:, 4], iou_threshold)
            filtered = filtered[keep]
        batch_boxes.append(filtered)
    
    return batch_boxes


import shutil
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def train(model, train_loader, valid_loader, optimizer, epochs=100, device='cuda'):
    model.to(device)
    best_loss = float('inf')
    loss_fn = YOLOLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Store metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        with torch.no_grad():
            val_progress = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, targets in val_progress:
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                
                outputs = model(images)
                loss = loss_fn(outputs, targets, model)
                val_loss += loss.item()
                
                # Calculate metrics
                pred_boxes = decode_predictions(outputs, model)
                for i, (preds, gt) in enumerate(zip(pred_boxes, targets)):
                    gt = gt.cpu().numpy()
                    preds = preds.cpu().numpy()
                    
                    # For each ground truth, find matching prediction
                    matched = set()
                    for gt_idx, gt_box in enumerate(gt):
                        best_iou = 0.0
                        best_pred = -1
                        for pred_idx, pred_box in enumerate(preds):
                            if pred_idx in matched:
                                continue
                            iou = calculate_iou(gt_box[:4], pred_box[:4])
                            if iou > 0.5 and pred_box[5] == gt_box[4]:
                                if iou > best_iou:
                                    best_iou = iou
                                    best_pred = pred_idx
                        if best_pred != -1:
                            total_tp += 1
                            matched.add(best_pred)
                        else:
                            total_fn += 1
                    
                    # Remaining predictions are false positives
                    total_fp += len(preds) - len(matched)
        
        avg_val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(avg_val_loss)
        
        # Calculate precision/recall
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f'output/best_model.pth')
        
        # Visualize validation results
        visualize_validation(model, valid_loader, device, epoch+1)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Precision: {precision:.2f} | Recall: {recall:.2f}')
        print('-' * 50)
        
        # Save metrics
        torch.save(metrics, 'output/training_metrics.pth')
    
    # Plot training curves
    plot_metrics(metrics)
    return metrics

def visualize_validation(model, valid_loader, device, epoch):
    model.eval()
    save_dir = f'output/val_vis/epoch_{epoch}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Get one batch
    images, targets = next(iter(valid_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        pred_boxes = decode_predictions(outputs, model)
    
    # Convert images to numpy
    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    
    for idx in range(images_np.shape[0]):
        img = (images_np[idx] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw ground truth (green)
        gt = targets[idx].cpu().numpy()
        for box in gt:
            cx, cy, w, h, cls_id = box
            x1 = int((cx - w/2) * 640)
            y1 = int((cy - h/2) * 640)
            x2 = int((cx + w/2) * 640)
            y2 = int((cy + h/2) * 640)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw predictions
        preds = pred_boxes[idx].cpu().numpy()
        for box in preds:
            cx, cy, w, h, conf, cls_id = box
            x1 = int((cx - w/2) * 640)
            y1 = int((cy - h/2) * 640)
            x2 = int((cx + w/2) * 640)
            y2 = int((cy + h/2) * 640)
            
            # Check if correct (IoU > 0.5 and class match)
            is_correct = False
            for gt_box in gt:
                iou = calculate_iou(gt_box[:4], [cx, cy, w, h])
                if iou > 0.5 and cls_id == gt_box[4]:
                    is_correct = True
                    break
            
            color = (0, 0, 255) if not is_correct else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        cv2.imwrite(f'{save_dir}/val_{idx}.jpg', img)

def calculate_iou(box1, box2):
    # box format: (cx, cy, w, h)
    box1 = [
        box1[0] - box1[2]/2,  # x1
        box1[1] - box1[3]/2,  # y1
        box1[0] + box1[2]/2,  # x2
        box1[1] + box1[3]/2   # y2
    ]
    box2 = [
        box2[0] - box2[2]/2,
        box2[1] - box2[3]/2,
        box2[0] + box2[2]/2,
        box2[1] + box2[3]/2
    ]
    
    # Calculate intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def plot_metrics(metrics):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['precision'], label='Precision')
    plt.plot(metrics['recall'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_metrics.png')
    plt.close()

# Cette fonction permet : 
# de gérer des tailles de cibles variables (nombre d’objets différent par image),
# d'empiler les images en un batch [B, 3, 640, 640],
# de garder les labels comme liste de torch.Tensor (non empilés).
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = [l for l in labels if l.shape[0] > 0]
    return images, labels

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# Load all filenames
all_data = pd.read_csv("../labels_YOLO.csv")
all_files = all_data['filename'].unique()
train_files, valid_files = train_test_split(all_files, test_size=0.2)

# Create datasets
train_transform = RandomAffineWithBoxes(
    degrees=15,
    translate=(0.1, 0.1),
    color_jitter=(0.2, 0.2, 0.2, 0.1),
    img_size=640
)

# For validation dataset (no transformations)
valid_transform = lambda img, boxes: (TF.to_tensor(img), torch.tensor(boxes))

train_dataset = ChocolateDataset(
    img_dir='../dataset_project_iapr2025/train',
    label_dir='labels',
    filenames=train_files,
    transform=train_transform
)

valid_dataset = ChocolateDataset(
    img_dir='../dataset_project_iapr2025/train',
    label_dir='labels',
    filenames=valid_files,
    transform=valid_transform
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)

# Initialize model and optimizer
model = YOLOv5(num_classes=13, anchors=ANCHORS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Start training
metrics = train(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    epochs=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Start training
train(model, train_loader, valid_loader, optimizer, epochs=100)