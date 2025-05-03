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

# Paramètres globaux
IMG_W, IMG_H = 6000, 4000  # dimensions originales des images
IMG_RESIZED_W, IMG_RESIZED_H = 640, 416  # divisible par 32
S_W = IMG_RESIZED_W // 32  # 640 / 32 = 20, 32 psk on fait 5 maxpool dans le CNN qui nous sort une image de 20x13
S_H = IMG_RESIZED_H // 32  # 416 / 32 = 13
C = 13  # nombre de classes de chocolats
B = 2   # nombre de boîtes par cellule (YOLOv1-style), on gardera celle avec la plus haute confiance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        fname = self.filenames[idx]  # nom de l'image
        img_path = os.path.join(self.img_dir, fname)  # chemin complet
        image = Image.open(img_path).convert("RGB")  # ouverture et conversion RGB
        image = image.resize((IMG_RESIZED_W, IMG_RESIZED_H))  # redimensionnement
        
        boxes = torch.zeros((S_H, S_W, B * 5 + C))  # TARGET = GROUND TRUTH (vide pour l'instant, construite plus tard) de forme (S_H, S_W, 23). Le NN voudra trouver ces boxes
                            #(13,  20, (x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2, class1, class2, ..., class13))
        annots = self.annotations[self.annotations['filename'] == fname]  # lignes de cette image
        for _, row in annots.iterrows():
            cx = row['cx'] / IMG_W # position x normalisée (0-1) sur l'image entière
            cy = row['cy'] / IMG_H # position y normalisée
            class_id = int(row['class']) # id de classe entre 0 et C-1 (=12 ici)

            grid_x = int(cx * S_W) # index colonne de la cellule qui contient le centre de la box
            grid_y = int(cy * S_H) # index ligne de la cellule qui contient le centre de la box
            if grid_x >= S_W or grid_y >= S_H:
                continue # hors grille, on saute (safety)

            x_cell = cx * S_W - grid_x # position relative dans la cellule (0-1)
            y_cell = cy * S_H - grid_y
            w = row['width'] / IMG_W # largeur relative des boîtes par rapport à l'image entière
            h = row['height'] / IMG_H # hauteur relative des boîtes

            # Génère la target
            gt_box = torch.tensor([x_cell, y_cell, w, h])

            # Calcule l'IoU avec les B boites possibles
            ious = []
            for b in range(B):
                pred_box = boxes[grid_y, grid_x, b * 5 : b * 5 + 4]
                if boxes[grid_y, grid_x, b * 5 + 4] == 1:
                    ious.append(bbox_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)))
                else:
                    ious.append(torch.tensor(0.0))

            best_b = torch.tensor(ious).argmax().item()

            boxes[grid_y, grid_x, best_b * 5 : best_b * 5 + 5] = torch.tensor([x_cell, y_cell, w, h, 1])
            boxes[grid_y, grid_x, B * 5 + class_id] = 1


        if self.transform:
            image = self.transform(image)  # transformation image (ToTensor)

        return image, boxes, fname  # image: (3,H,W), target: (S,S,23)

    
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
            x[..., b * 5 + 4] = torch.sigmoid(x[..., b * 5 + 4])  # conf
            
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

# === Nouvelle fonction de perte YOLO avec B boîtes ===
def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5): # pred et target sont de forme (B, S_H, S_W, 23) = sortie du réseau
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
        else: # Si pas d'objet dans la cellule, on ne calcule pas la perte (pour éviter loss nan)
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
    return total_loss


# === Affichage et sauvegarde des prédictions === affiche les boites prédites avec un score de confiance supérieur à un seuil
def save_validation_predictions(model, dataset, output_dir, score_thresh=0.3, iou_thresh=0.5):
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

        boxes = []
        scores = []
        classes = []
        trues = []

        for row in range(S_H):
            for col in range(S_W):
                for b in range(B):
                    offset = b * 5
                    cell = pred[row, col]
                    conf = cell[offset + 4]
                    class_scores = cell[B * 5:]
                    score = conf * np.max(class_scores)
                    if score > score_thresh:
                        x, y, w, h = cell[offset + 0], cell[offset + 1], cell[offset + 2], cell[offset + 3]
                        class_id = np.argmax(cell[B * 5:])
                        cx = (col + x) / S_W
                        cy = (row + y) / S_H
                        bw = w
                        bh = h
                        x1 = (cx - bw / 2) * IMG_RESIZED_W
                        y1 = (cy - bh / 2) * IMG_RESIZED_H
                        x2 = (cx + bw / 2) * IMG_RESIZED_W
                        y2 = (cy + bh / 2) * IMG_RESIZED_H
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        classes.append(class_id)
                        trues.append(torch.argmax(target[row, col, B * 5:]).item())

        keep = non_max_suppression(boxes, scores, iou_threshold=iou_thresh)
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            color = 'black' if classes[i] == trues[i] else 'red'
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=2)
            ax.add_patch(rect)

        for x in range(1, S_W):
            plt.axvline(x * IMG_RESIZED_W / S_W, color='white', linewidth=0.5)
        for y in range(1, S_H):
            plt.axhline(y * IMG_RESIZED_H / S_H, color='white', linewidth=0.5)

        plt.axis('off')
        plt.savefig(os.path.join(output_dir, fname.replace(".JPG", ".png")))
        plt.close()
        

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou <= iou_threshold]

    return keep


# === Boucle d'entraînement ===

all_data = pd.read_csv("../labels_YOLO.csv")
all_files = all_data['filename'].unique()
train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_transform = T.Compose([
    T.Resize((IMG_RESIZED_H, IMG_RESIZED_W)),
    T.RandomAffine(degrees=180, translate=(0.1, 0.1)),  # rotation jusqu'à +/-180°, translation jusqu'à ±10% sur x et y
    T.ToTensor()
])

valid_transform = T.Compose([
    T.Resize((IMG_RESIZED_H, IMG_RESIZED_W)),
    T.ToTensor()
])

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
    print(f"Train Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    for images, targets, _ in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = yolo_loss(outputs, targets)
            val_loss += loss.item()
    print(f"Valid Loss: {val_loss / len(valid_loader):.4f}")

    # Sauvegarde des prédictions visuelles
    save_validation_predictions(model, valid_dataset, "images_validation_YOLO")

