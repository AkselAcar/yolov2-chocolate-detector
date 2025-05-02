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
IMG_RESIZED_W, IMG_RESIZED_H = 600, 400  # dimensions d'entraînement
S = 10  # grille SxS pour YOLOv1
C = 13  # nombre de classes de chocolats
B = 2   # nombre de boîtes par cellule (YOLOv1-style)

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
        boxes = torch.zeros((S, S, B * 5 + C))  # target vide de forme (S, S, 23)

        annots = self.annotations[self.annotations['filename'] == fname]  # lignes de cette image
        for _, row in annots.iterrows():
            cx = row['cx'] / IMG_W  # position x normalisée (0-1) ATTENTION A CHANGER SI LES CX ET CY SONT PLUS DIVISES PAR 10 DE BASE DANS LE CSV
            cy = row['cy'] / IMG_H  # position y normalisée
            class_id = int(row['class'])  # id de classe

            grid_x = int(cx * S)  # index colonne de la cellule
            grid_y = int(cy * S)  # index ligne de la cellule
            if grid_x >= S or grid_y >= S:
                continue  # hors grille, on saute

            x_cell = cx * S - grid_x  # position relative dans la cellule (0-1)
            y_cell = cy * S - grid_y
            w = row['width'] / IMG_RESIZED_W  # largeur relative
            h = row['height'] / IMG_RESIZED_H  # hauteur relative

            if boxes[grid_y, grid_x, 4] == 0:  # seulement si aucune boîte n'est présente (indice 4 est la confidence)
                boxes[grid_y, grid_x, 0:5] = torch.tensor([x_cell, y_cell, w, h, 1])  # x,y,w,h,conf
                boxes[grid_y, grid_x, 5 + class_id] = 1  # encodage one-hot de la classe

        if self.transform:
            image = self.transform(image)  # transformation image (ToTensor)

        return image, boxes, fname  # image: (3,H,W), target: (S,S,23)

# === Réseau YOLOv4 simplifié ===
class YOLOv4Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(  # feature extractor simple
            nn.Conv2d(3, 16, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(  # prédiction directe des (x,y,w,h,conf,classes)
            nn.Conv2d(256, (B * 5 + C), 1)  # 1x1 conv, 23 sorties par cellule
        )

    def forward(self, x):
        x = self.backbone(x)  # (B, 256, H/32, W/32)
        x = self.head(x)      # (B, 23, S, S)
        x = x.permute(0, 2, 3, 1)  # (B, S, S, 23) pour correspondre au target
        return x


# === Fonction IOU ===
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

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)  # pour éviter division par 0

# === Fonction de perte simplifiée YOLO ===
def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5):
    obj_mask = target[..., 4] == 1  # masque pour cellules contenant un objet
    noobj_mask = target[..., 4] == 0  # masque pour les autres

    # perte de localisation sur x, y, w, h
    coord_loss = lambda_coord * F.mse_loss(pred[obj_mask][..., 0:2], target[obj_mask][..., 0:2]) + \
                 lambda_coord * F.mse_loss(torch.sqrt(pred[obj_mask][..., 2:4] + 1e-6),
                                           torch.sqrt(target[obj_mask][..., 2:4] + 1e-6))

    # perte sur la confiance (objet présent / absent)
    conf_loss_obj = F.mse_loss(pred[obj_mask][..., 4], target[obj_mask][..., 4])
    conf_loss_noobj = lambda_noobj * F.mse_loss(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])

    # perte de classification (cross entropy entre classes)
    class_loss = F.cross_entropy(pred[obj_mask][..., 5:], torch.argmax(target[obj_mask][..., 5:], dim=-1))

    total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
    return total_loss


# === Affichage et sauvegarde des prédictions ===
def save_validation_predictions(model, dataset, output_dir, score_thresh=0.3):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for i in range(len(dataset)):
        img_tensor, target, fname = dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor).squeeze(0).cpu().numpy()  # (S, S, 23)

        img_np = np.transpose(img_tensor.squeeze(0).cpu().numpy(), (1, 2, 0))
        plt.figure(figsize=(6, 4))
        plt.imshow(img_np) # "dessine" l'image en mémoire, ne la dessine pas vraiment comme plt.show()
        ax = plt.gca()

        for row in range(S):
            for col in range(S):
                cell = pred[row, col]
                conf = cell[4]
                if conf > score_thresh:
                    x, y, w, h = cell[0], cell[1], cell[2], cell[3]
                    class_id = np.argmax(cell[5:])
                    cx = (col + x) / S
                    cy = (row + y) / S
                    bw = w
                    bh = h
                    x1 = (cx - bw / 2) * IMG_RESIZED_W
                    y1 = (cy - bh / 2) * IMG_RESIZED_H
                    x2 = (cx + bw / 2) * IMG_RESIZED_W
                    y2 = (cy + bh / 2) * IMG_RESIZED_H

                    true_class = torch.argmax(target[row, col, 5:]).item()
                    color = 'black' if true_class == class_id else 'red'
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', lw=2)
                    ax.add_patch(rect)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, fname.replace(".JPG", ".png")))
        plt.close()


# === Boucle d'entraînement ===

all_data = pd.read_csv("../labels_YOLO.csv")
all_files = all_data['filename'].unique()
train_files, valid_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_transform = T.Compose([
    T.Resize((IMG_RESIZED_H, IMG_RESIZED_W)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomHorizontalFlip(p=0.5),
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

print("Data_loaders created")

model = YOLOv4Small()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    save_validation_predictions(model, valid_dataset, "images_validation_YOLO", score_thresh=0.3)

