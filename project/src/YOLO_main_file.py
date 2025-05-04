import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as TF

# Paramètres globaux
IMG_W, IMG_H = 6000, 4000  # dimensions originales des images
IMG_RESIZED_W, IMG_RESIZED_H = 640, 416  # divisible par 32
S_W = IMG_RESIZED_W // 32  # 640 / 32 = 20, 32 psk on fait 5 maxpool dans le CNN qui nous sort une image de 20x13
S_H = IMG_RESIZED_H // 32  # 416 / 32 = 13
C = 13  # nombre de classes de chocolats
B = 2   # nombre de boîtes par cellule (YOLOv1-style), on gardera celle avec la plus haute confiance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w_min_orig_size = 200 #pixels sur 6000
w_max_orig_size = 1000 #pixels sur 6000
h_min_orig_size = 200 #pixels sur 4000
h_max_orig_size = 1000 #pixels sur 4000

w_min = w_min_orig_size / IMG_W # largeur minimale normalisée
w_max = w_max_orig_size / IMG_W # largeur maximale normalisée
h_min = h_min_orig_size / IMG_H # hauteur minimale normalisée
h_max = h_max_orig_size / IMG_H # hauteur maximale normalisée


# === Dataset YOLO ===
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_csv, filenames, transform=None):
        self.img_dir = img_dir  # chemin des images
        self.annotations = pd.read_csv(label_csv)  # lecture du CSV d'annotations
        self.annotations = self.annotations[self.annotations['filename'].isin(filenames)]
        self.filenames = filenames
        self.transform = transform  # transformation PyTorch (ex: ToTensor)

    def __len__(self):
        return len(self.filenames)  # nombre d'images
    
    def __getitem__(self, idx):
        fname = self.filenames[idx] # nom de l'image
        img_path = os.path.join(self.img_dir, fname) # chemin complet
        image = Image.open(img_path).convert("RGB") # ouverture et conversion RGB

        # Collecte les annotations de cette image
        annots = self.annotations[self.annotations['filename'] == fname] # lignes de cette image
        boxes_raw = []
        class_ids = []

        for _, row in annots.iterrows():
            cx = row['cx'] / IMG_W # position x normalisée (0-1) sur l'image entière
            cy = row['cy'] / IMG_H # position y normalisée
            w = row['width'] / IMG_W  # largeur relative de la box par rapport à l'image entière
            h = row['height'] / IMG_H # hauteur relative de la box
            boxes_raw.append([cx, cy, w, h]) # box entièrement normalisée
            class_ids.append(int(row['class'])) # id de classe entre 0 et C-1 (=12 ici)

        image = image.resize((IMG_RESIZED_W, IMG_RESIZED_H))
        # Applique la transformation affine sur l'image ET les boîtes, transform a été créée de manière à prendre l'image redimensionnée
        if self.transform:
            image, boxes_raw = self.transform(image, boxes_raw)

        # Initialisation de la grille YOLO
        boxes = torch.zeros((S_H, S_W, B * 5 + C)) # TARGET = GROUND TRUTH (vide pour l'instant, construite plus tard) de forme (S_H, S_W, 23). Le NN voudra trouver ces boxes
                           #(13,  20, (x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, class1, class2, ..., class13))
        # Remplit la grille avec les boxes transformées
        for box, class_id in zip(boxes_raw, class_ids):
            cx, cy, w, h = box
            # Clip w et h aux bornes définies (utile si rotation ou bruit)
            w = min(max(w, w_min), w_max)
            h = min(max(h, h_min), h_max)

            grid_x = int(cx * S_W) # index colonne de la cellule qui contient le centre de la box
            grid_y = int(cy * S_H) # index ligne de la cellule qui contient le centre de la box
            if grid_x >= S_W or grid_y >= S_H:
                continue # hors grille, on saute (safety)

            x_cell = cx * S_W - grid_x # position relative dans la cellule (0-1)
            y_cell = cy * S_H - grid_y 

            # Sélection de la meilleure boîte (ou la première libre ici)
            best_b = 0
            for b in range(B):
                if boxes[grid_y, grid_x, b * 5 + 4] == 0:
                    best_b = b
                    break

            boxes[grid_y, grid_x, best_b * 5 : best_b * 5 + 5] = torch.tensor([x_cell, y_cell, w, h, 1])
            boxes[grid_y, grid_x, B * 5 + class_id] = 1

        return image, boxes, fname # image: (3,H,W), target: (S,S,23)

    
class YOLOv4Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential( # feature extractor simple, stride=2 est équivalent à un maxpool 2x2 => réduit l'image de moitié 5 fois D d'où les 32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
        )

        self.head = nn.Sequential( # prédiction directe des (x,y,w,h,conf,classes)
            nn.Conv2d(1024, 512, 1), nn.LeakyReLU(0.1), # 1x1 conv, 23 sorties par cellule
            nn.Conv2d(512, (B * 5 + C), 1)
        )

        self._initialize_weights() # initialisation des poids

    def forward(self, x):
        x = self.backbone(x) # (B, 256, H/32, W/32)
        x = self.head(x) # (B, 23, S_H, S_W)
        x = x.permute(0, 2, 3, 1) # (B, S_H, S_W, 23) pour correspondre au target
        # appliquer sigmoid sur x, y, et conf
        for b in range(B):
            x[..., b * 5 + 0] = torch.sigmoid(x[..., b * 5 + 0])  # x
            x[..., b * 5 + 1] = torch.sigmoid(x[..., b * 5 + 1])  # y
            #x[..., b * 5 + 4] = torch.sigmoid(x[..., b * 5 + 4])  # conf (commenté car new loss with BCE)
            
            x[..., b * 5 + 2] = w_min + (w_max - w_min) * torch.sigmoid(x[..., b * 5 + 2])  # w
            x[..., b * 5 + 3] = h_min + (h_max - h_min) * torch.sigmoid(x[..., b * 5 + 3])  # h
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# === Fonction IOU === convertit (x, y, w, h) en coins (x1, y1, x2, y2) et calcule le rapport intersection over union de target et prediction
def bbox_iou(box1, box2):
    # box1 et box2 sont des tenseurs contenant (x, y, w, h)
    x1, y1, w1, h1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    x2, y2, w2, h2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Conversion en coordonnées coin sup. gauche et inf. droite
    b1_x1 = x1 - w1 / 2
    b1_y1 = y1 - h1 / 2
    b1_x2 = x1 + w1 / 2
    b1_y2 = y1 + h1 / 2
    b2_x1 = x2 - w2 / 2
    b2_y1 = y2 - h2 / 2
    b2_x2 = x2 + w2 / 2
    b2_y2 = y2 + h2 / 2

    # détermine les coins du rectangle d'intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # calcule l'aire d'intersection (0 si pas d'intersection)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    # calcule l'aire d'union (A1 + A2 - A_intersection)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)  # pour éviter division par 0
"""
# === Nouvelle fonction de perte YOLO avec B boîtes ===
def yolo_loss(pred, target, lambda_coord=5.0, lambda_noobj=0.5, lambda_count=1.0): # pred et target sont de forme (B, S_H, S_W, 23) = sortie du réseau
    total_loss = 0.0

    for b in range(B): # => b = 0 ou 1 si B=2
        # Indices pour la boîte b
        box_pred = pred[..., b * 5 : b * 5 + 5]
        box_target = target[..., b * 5 : b * 5 + 5]

        obj_mask = box_target[..., 4] == 1 # target = 1 si la boîte b est présente
        noobj_mask = box_target[..., 4] == 0 # target = 0 si la boîte b n'est pas présente

    # Coordonnées
        if obj_mask.sum() > 0:
            coord_loss = lambda_coord * F.mse_loss(box_pred[obj_mask][..., 0:2], box_target[obj_mask][..., 0:2]) + \
                         lambda_coord * F.mse_loss(
                            torch.sqrt(torch.clamp(box_pred[obj_mask][..., 2:4], min=1e-6)),# compare sqrt(w,h) comme dans le paper pour stabiliser l'apprentissage, et clamp pour éviter les valeurs négatives (le réseau doit sortir des valeurs positives mais il n'a aucune contrainte à le faire)
                            torch.sqrt(box_target[obj_mask][..., 2:4])
                         )
            # Confiance : on veut que la boite responsable prévoie une confiance proche de 1 et les autres boites proches de 0
            conf_loss_obj = F.mse_loss(box_pred[obj_mask][..., 4], box_target[obj_mask][..., 4])
        else: # Si pas d'objet dans la cellule, on ne calcule pas la perte (pour éviter loss nan) PEUT ETRE BOFFFFF
            coord_loss = torch.tensor(0.0, device=pred.device)
            conf_loss_obj = torch.tensor(0.0, device=pred.device)

        conf_loss_noobj = lambda_noobj * F.mse_loss(box_pred[noobj_mask][..., 4], box_target[noobj_mask][..., 4])

        total_loss += coord_loss + conf_loss_obj + conf_loss_noobj
    # Classification : partagée entre les B boîtes
    class_mask = target[..., B * 5 + 0 : B * 5 + C].sum(dim=-1) > 0 # on sélectionne les indices du one hot encoding pour sélectionner les cellules qui contiennent au moins un objet
    class_loss = F.cross_entropy(
        pred[class_mask][..., B * 5 : B * 5 + C],
        torch.argmax(target[class_mask][..., B * 5 : B * 5 + C], dim=-1)
    ) if class_mask.sum() > 0 else torch.tensor(0.0, device=pred.device)

    total_loss += class_loss

    # === 🔢 Ajout : count_loss ===
    true_counts = target[..., B * 5 : B * 5 + C].sum(dim=(0, 1))  # (C,)
    pred_class_probs = pred[..., B * 5 : B * 5 + C]  # (S_H, S_W, C)
    pred_conf = pred[..., [b * 5 + 4 for b in range(B)]].max(dim=-1).values.unsqueeze(-1)  # (S_H, S_W, 1)
    pred_counts = (pred_class_probs * pred_conf).sum(dim=(0, 1))  # (C,)

    count_loss = F.mse_loss(pred_counts, true_counts)
    total_loss += lambda_count * count_loss

    return total_loss

"""
"""
def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5, lambda_count=1.0): # pred et target sont de forme (B, S_H, S_W, 23) = sortie du réseau
    total_loss = 0.0

    for b in range(B): # => b = 0 ou 1 si B=2
        # Indices pour la boîte b
        box_pred = pred[..., b * 5 : b * 5 + 5]
        box_target = target[..., b * 5 : b * 5 + 5]

        obj_mask = box_target[..., 4] == 1  # boîte b est responsable
        cell_has_any_object = target[..., 4::5].any(dim=-1)  # (S_H, S_W)
        noobj_mask = ~cell_has_any_object  # cellule totalement vide

        # --- Coordinate + Confidence (obj) ---
        if obj_mask.sum() > 0:
            coord_loss = lambda_coord * (
                F.mse_loss(box_pred[obj_mask][..., 0:2], box_target[obj_mask][..., 0:2]) +
                F.mse_loss(
                    torch.sqrt(torch.clamp(box_pred[obj_mask][..., 2:4], min=1e-6)),
                    torch.sqrt(box_target[obj_mask][..., 2:4])
                )
            )
            conf_loss_obj = F.binary_cross_entropy_with_logits(
                box_pred[obj_mask][..., 4],
                box_target[obj_mask][..., 4]
            )
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)
            conf_loss_obj = torch.tensor(0.0, device=pred.device)

        # --- Confidence (noobj) ---
        conf_loss_noobj = F.binary_cross_entropy_with_logits(
            box_pred[noobj_mask][..., 4],
            box_target[noobj_mask][..., 4]
        )

        total_loss += coord_loss + conf_loss_obj + lambda_noobj * conf_loss_noobj

    # --- Classification ---
    class_mask = target[..., B * 5 : B * 5 + C].sum(dim=-1) > 0
    if class_mask.sum() > 0:
        class_loss = F.cross_entropy(
            pred[class_mask][..., B * 5 : B * 5 + C],
            torch.argmax(target[class_mask][..., B * 5 : B * 5 + C], dim=-1)
        )
    else:
        class_loss = torch.tensor(0.0, device=pred.device)

    total_loss += class_loss

    # --- Counting loss ---
    true_counts = target[..., B * 5 : B * 5 + C].sum(dim=(0, 1))  # (C,)
    pred_class_probs = torch.sigmoid(pred[..., B * 5 : B * 5 + C])  # (S_H, S_W, C)
    pred_conf_logits = pred[..., [b * 5 + 4 for b in range(B)]]  # (S_H, S_W, B)
    pred_conf = torch.sigmoid(pred_conf_logits).max(dim=-1).values.unsqueeze(-1)  # (S_H, S_W, 1)
    pred_counts = (pred_class_probs * pred_conf).sum(dim=(0, 1))  # (C,)
    count_loss = F.mse_loss(pred_counts, true_counts)

    total_loss += lambda_count * count_loss

    return total_loss
"""


def yolo_loss(pred, target, lambda_coord=10, lambda_noobj=5, lambda_count=1.0):
    total_loss = 0.0

    for b in range(B):
        box_pred = pred[..., b*5 : b*5+5]
        box_target = target[..., b*5 : b*5+5]

        obj_mask = box_target[..., 4] == 1
        noobj_mask = box_target[..., 4] == 0  # Corrected: per anchor

        # Coordinate loss (unchanged)
        if obj_mask.sum() > 0:
            coord_loss = lambda_coord * (
                F.mse_loss(box_pred[obj_mask][..., :2], box_target[obj_mask][..., :2]) +
                F.mse_loss(
                    torch.sqrt(box_pred[obj_mask][..., 2:4].clamp(min=1e-6)),
                    torch.sqrt(box_target[obj_mask][..., 2:4])
                )
            )
            conf_loss_obj = F.binary_cross_entropy_with_logits(
                box_pred[obj_mask][..., 4], box_target[obj_mask][..., 4]
            )
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)
            conf_loss_obj = torch.tensor(0.0, device=pred.device)

        # No-object confidence loss (corrected mask)
        conf_loss_noobj = F.binary_cross_entropy_with_logits(
            box_pred[noobj_mask][..., 4], box_target[noobj_mask][..., 4]
        )
        total_loss += coord_loss + conf_loss_obj + lambda_noobj * conf_loss_noobj

    # --- Classification Loss ---
    class_mask = target[..., B*5 : B*5+C].sum(dim=-1) > 0
    if class_mask.sum() > 0:
        class_loss = F.cross_entropy(
            pred[class_mask][..., B*5 : B*5+C],
            torch.argmax(target[class_mask][..., B*5 : B*5+C], dim=-1)
        )
    else:
        class_loss = torch.tensor(0.0, device=pred.device)
    total_loss += class_loss

    # --- Counting Loss (Fixed) ---
    true_counts = target[..., B*5 : B*5+C].sum(dim=(0, 1))  # (C,)
    
    # Get class probabilities (softmax for mutually exclusive classes)
    pred_class_probs = F.softmax(pred[..., B*5 : B*5+C], dim=-1)  # (S_H, S_W, C)
    # Get max confidence per cell across all boxes
    pred_conf = torch.sigmoid(pred[..., 4::5]).max(dim=-1).values  # (S_H, S_W)
    # Aggregate predicted counts
    pred_counts = (pred_class_probs * pred_conf.unsqueeze(-1)).sum(dim=(0, 1))  # (C,)
    # Use Poisson loss for counting
    count_loss = F.poisson_nll_loss(pred_counts, true_counts, log_input=False)
    total_loss += lambda_count * count_loss

    return total_loss

import shutil
def save_validation_predictions(model, dataset, output_dir, score_thresh=0.3, iou_thresh=0.5):
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)

    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for i in range(len(dataset)):
        img_tensor, target, fname = dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            pred = model(img_tensor).squeeze(0).cpu().numpy()

        img_np = np.transpose(img_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
        plt.figure(figsize=(6, 4))
        plt.imshow(img_np)
        ax = plt.gca()

        boxes_info = []  # nouveau : stocke les infos utiles pour chaque boîte

    for row in range(S_H):
        for col in range(S_W):
            for b in range(B):
                offset = b * 5
                cell = pred[row, col]
                
                # Apply sigmoid to confidence
                conf = 1 / (1 + np.exp(-cell[offset + 4]))
                
                # Apply softmax to class scores
                class_scores = cell[B*5:]
                class_probs = np.exp(class_scores) / np.sum(np.exp(class_scores))
                score = conf * np.max(class_probs)
                class_id = np.argmax(class_probs)

                if score > score_thresh:
                        x, y, w, h = cell[offset + 0], cell[offset + 1], cell[offset + 2], cell[offset + 3]
                        #class_id = np.argmax(class_scores)
                        cx = (col + x) / S_W
                        cy = (row + y) / S_H
                        x1 = (cx - w / 2) * IMG_RESIZED_W
                        y1 = (cy - h / 2) * IMG_RESIZED_H
                        x2 = (cx + w / 2) * IMG_RESIZED_W
                        y2 = (cy + h / 2) * IMG_RESIZED_H

                        if target[row, col, B * 5:].sum() > 0:
                            true_class = torch.argmax(target[row, col, B * 5:]).item()
                        else:
                            true_class = -1  # signifie qu'aucun objet n'est présent
                        boxes_info.append({
                            'box': [x1, y1, x2, y2],
                            'score': score,
                            'class_id': class_id,
                            'true_class': true_class,
                            'row': row,
                            'col': col
                        })

        # NMS et affichage final
        keep = non_max_suppression([info['box'] for info in boxes_info], [info['score'] for info in boxes_info], iou_thresh)

        for i in keep:
            info = boxes_info[i]
            x1, y1, x2, y2 = info['box']
            class_id = info['class_id']
            true_class = info['true_class']
            color = 'black' if class_id == true_class and true_class != -1 else 'red'
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=1)
            ax.add_patch(rect)

        for x in range(1, S_W):
            plt.axvline(x * IMG_RESIZED_W / S_W, color='white', linewidth=0.5)
        for y in range(1, S_H):
            plt.axhline(y * IMG_RESIZED_H / S_H, color='white', linewidth=0.5)

        plt.axis('off')
        plt.savefig(os.path.join(output_dir, fname.replace(".JPG", ".png")))
        plt.close()

def non_max_suppression(boxes, scores, iou_threshold=0.5, img_width=IMG_RESIZED_W, img_height=IMG_RESIZED_H):
    """
    Args:
        boxes: List of [x1, y1, x2, y2] in pixel coordinates
        scores: List of confidence scores
        iou_threshold: Threshold for IoU to consider overlap
        img_width: Width of the resized image
        img_height: Height of the resized image
    """
    if len(boxes) == 0:
        return []
    
    # Convert to tensors and clamp coordinates to image boundaries
    boxes = torch.tensor(boxes)
    boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_width)   # x1
    boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_height)  # y1
    boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=img_width)   # x2
    boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=img_height)  # y2
    
    scores = torch.tensor(scores)
    
    # Calculate areas of all boxes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort indices by descending scores
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        # Pick the highest-scoring box
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # Compute IoU between the picked box and the rest
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # Keep boxes with IoU <= threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return keep


all_data = pd.read_csv("../labels_YOLO.csv")
all_files = all_data['filename'].unique()
train_files, valid_files = train_test_split(all_files, test_size=0.2) # , random_state=42 si on veut la même répartition à chaque fois


class RandomAffineWithBoxes:
    def __init__(self, degrees=0, translate=(0.0, 0.0)):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, image, boxes):
        original_image = image.copy()
        original_boxes = boxes.copy()

        angle = np.random.uniform(-self.degrees, self.degrees)
        translate_x = int(self.translate[0] * image.size[0])
        translate_y = int(self.translate[1] * image.size[1])
        translate = (np.random.randint(-translate_x, translate_x+1), np.random.randint(-translate_y, translate_y+1)) # np.random.randint(a, b) est inclusif pour a et exclusif pour b d'où les +1

        image = TF.affine(image, angle=angle, translate=translate, scale=1.0, shear=0)

        new_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            cx_pix = cx * IMG_RESIZED_W
            cy_pix = cy * IMG_RESIZED_H

            corners = torch.tensor([
                [cx_pix - w*IMG_RESIZED_W/2, cy_pix - h*IMG_RESIZED_H/2],
                [cx_pix + w*IMG_RESIZED_W/2, cy_pix - h*IMG_RESIZED_H/2],
                [cx_pix + w*IMG_RESIZED_W/2, cy_pix + h*IMG_RESIZED_H/2],
                [cx_pix - w*IMG_RESIZED_W/2, cy_pix + h*IMG_RESIZED_H/2]
            ], dtype=torch.float)

            angle_rad = -angle * np.pi / 180.0 # signe inversé car axe y vers le bas
            rot_matrix = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ], dtype=torch.float)

            center = torch.tensor([IMG_RESIZED_W / 2, IMG_RESIZED_H / 2], dtype=torch.float)
            rotated = (corners - center) @ rot_matrix.T + center
            rotated += torch.tensor(translate, dtype=torch.float)

            x_coords = rotated[:, 0].clamp(0, IMG_RESIZED_W)
            y_coords = rotated[:, 1].clamp(0, IMG_RESIZED_H)
            new_cx = (x_coords.min() + x_coords.max()) / 2 / IMG_RESIZED_W
            new_cy = (y_coords.min() + y_coords.max()) / 2 / IMG_RESIZED_H
            new_w = (x_coords.max() - x_coords.min()) / IMG_RESIZED_W
            new_h = (y_coords.max() - y_coords.min()) / IMG_RESIZED_H

            if (
                0 <= new_cx <= 1 and 0 <= new_cy <= 1 and
                0 < new_w <= 1 and 0 < new_h <= 1
            ): # on vérifie que la boîte est toujours dans l'image
                new_boxes.append([new_cx, new_cy, new_w, new_h])

        # Si toutes les boîtes sont invalides, on annule la transformation
        if len(new_boxes) == 0:
            return TF.to_tensor(original_image), original_boxes
        
        return TF.to_tensor(image), new_boxes

# Remplace la transformation simple par la version custom
train_transform = RandomAffineWithBoxes(degrees=20, translate=(0.1, 0.1))
valid_transform = lambda img, boxes: (T.ToTensor()(img), boxes)

train_dataset = YOLODataset("../dataset_project_iapr2025/train", "../labels_YOLO.csv", train_files, train_transform)
valid_dataset = YOLODataset("../dataset_project_iapr2025/train", "../labels_YOLO.csv", valid_files, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = YOLOv4Small()
print(model)
print(f"Nombre total de paramètres entraînables : {count_parameters(model):,}")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 30

best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, targets, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = yolo_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    for images, targets, _ in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = yolo_loss(outputs, targets, lambda_coord=5.0, lambda_noobj=5.0, lambda_count=1.0)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(valid_loader)
    print(f"Valid Loss: {avg_val_loss:.4f}")

    # Sauvegarde des prédictions visuelles
    save_validation_predictions(model, valid_dataset, "images_validation_YOLO", score_thresh=0.6, iou_thresh=0.6)

    # Sauvegarde du meilleur modèle complet (weights + optimizer + epoch)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss
        }
        torch.save(checkpoint, "checkpoint_yolov1.pth")
        print(f"✔ Best model saved at epoch {epoch+1} with val_loss = {avg_val_loss:.4f}")
