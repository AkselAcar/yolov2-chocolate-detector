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
        image = cv2.imread(img_path) # original image 6000x4000
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size)) / 255.0 # Normalize to [0, 1] and resize to 640x640
        
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.JPG', '.txt'))
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cx, cy, w, h, int(cls)]) # Ajoute chaque boîte à la liste boxes sous la forme [cx, cy, w, h, classe]
        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(boxes) if boxes else torch.zeros((0, 5)) # L’image est convertie en tenseur PyTorch, puis permutée de [H, W, Channels] à [Channels, H, W] (ordre requis par PyTorch).
        # shape de boxes = [0, 5] si pas de boîte, sinon [N, 5] où N est le nombre de boîtes


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

def decode_predictions(preds, model, img_size=640, conf_threshold=0.2): # preds = (bs, 9*(5+13), 40, 40)
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
        batch_boxes.append(filtered)
    
    return batch_boxes


import shutil
import os
# Modified train function with visualization
def train(model, dataloader, optimizer, epochs=100):
    
    model.train()
    device = next(model.parameters()).device
    

    for epoch in tqdm(range(epochs), desc="Total Training"):
        total_loss = 0
        model.train()
        
        for batch_idx, (images, targets) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'output/checkpoint_epoch_{epoch+1}.pth')

        # Visualization after each epoch
        model.eval()
        with torch.no_grad():
            viz_images, viz_targets = next(iter(dataloader)) # one batch for visualization
            viz_images = viz_images.to(device)
            outputs = model(viz_images)
            pred_boxes = decode_predictions(outputs, model) # sh


            # Supprimer le dossier s'il existe
            save_dir = 'output/training_plots'
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            for idx in range(viz_images.size(0)):
                # Convert tensor to OpenCV format
                img = viz_images[idx].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Draw ground truth
                gt = viz_targets[idx].cpu().numpy()
                for box in gt:
                    cx, cy, w, h, cls_id = box
                    x1 = int((cx - w/2) * 640)
                    y1 = int((cy - h/2) * 640)
                    x2 = int((cx + w/2) * 640)
                    y2 = int((cy + h/2) * 640)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'GT:{int(cls_id)}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw predictions
                boxes = pred_boxes[idx].cpu().numpy()
                for box in boxes:
                    cx, cy, w, h, conf, cls_id = box
                    x1 = int((cx - w/2) * 640)
                    y1 = int((cy - h/2) * 640)
                    x2 = int((cx + w/2) * 640)
                    y2 = int((cy + h/2) * 640)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'P:{int(cls_id)} {conf:.2f}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                cv2.imwrite(f'{save_dir}/image_{idx}.jpg', img)
        
        model.train()


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
# Dataset (STILL NEEDED)
dataset = ChocolateDataset('../dataset_project_iapr2025/train', 'labels')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

# Initialize model with explicit anchors
model = YOLOv5(num_classes=13, anchors=ANCHORS)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = YOLOLoss()


# Training Loop (UPDATED)
train(model, dataloader, optimizer)  # Uses both dataset and anchors

