import os
import json
import pandas as pd
import numpy as np
import cv2
from scipy.signal.windows import gaussian as scipy_gaussian

import matplotlib.pyplot as plt

import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datetime import datetime


# 1. Paramètres
CSV_PATH    = "../label_annotated.csv"   # votre export CSV de VIA
IMG_DIR     = "../dataset_project_iapr2025/train"                # dossier de vos .JPG
OUT_DIR     = "../density_maps/"                                 # dossier où sauver les .npz
SIGMA       = 150                                             # écart-type de la gaussienne
CLASS_NAMES = [
    "Jelly_White","Jelly_Milk","Jelly_Black","Amandina","Crème_brûlée",
    "Triangolo","Tentation_noir","Comtesse","Noblesse","Noir_authentique",
    "Passion_au_lait","Arabia","Stracciatella"
]
# 2. Mapping nom → indice
class2idx = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------------------------Create density_maps----------------------------------

# 3. Précompute the Gaussian kernel
kern_size = int(6 * SIGMA) | 1   # kernel size, ensure odd
g1d = scipy_gaussian(kern_size, SIGMA)
kernel2d = np.outer(g1d, g1d)
kernel2d /= kernel2d.sum()       # normalize sum to 1
half_k = kern_size // 2

# 4. Charger le CSV
df = pd.read_csv(CSV_PATH)
# 5. Créer le dossier de sortie
os.makedirs(OUT_DIR, exist_ok=True)

# 6. Boucle sur chaque image
'''

for fname, group in df.groupby("filename"):
    # 6.1 Lire l'image pour connaître ses dimensions
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Impossible de lire l'image : {img_path}")
        continue
    H, W = img.shape[:2]
    
    # 6.2 Initialiser le tenseur de density maps
    density = np.zeros((H, W, len(CLASS_NAMES)), dtype=np.float32)
    
    # 6.3 Pour chaque point annoté, scatter-add le kernel
    for _, row in group.iterrows():
        shape_attr = json.loads(row["region_shape_attributes"])
        x = int(shape_attr["cx"])
        y = int(shape_attr["cy"])
        choco_type = json.loads(row["region_attributes"])["Choco_Type"]
        c = class2idx.get(choco_type, None)
        if c is None:
            continue
        
        # bounds for scatter-add
        y0 = max(0,   y - half_k)
        y1 = min(H,   y + half_k + 1)
        x0 = max(0,   x - half_k)
        x1 = min(W,   x + half_k + 1)
        ky0 = half_k - (y - y0)
        ky1 = half_k + (y1 - y)
        kx0 = half_k - (x - x0)
        kx1 = half_k + (x1 - x)
        
        density[y0:y1, x0:x1, c] += kernel2d[ky0:ky1, kx0:kx1]
    
    # 6.4 Conversion en float16 pour réduire la taille
    density = density.astype(np.float16)
    
    # 6.5 Sauvegarder au format compressé .npz
    out_path = os.path.join(OUT_DIR, fname.replace(".JPG", ".npz"))
    np.savez_compressed(out_path, density=density)
    
    print(f"Saved compressed density map for {fname} → {out_path}")

'''
# ─── 1. CONFIG ────────────────────────────────────────────────────────────────
DM_DIR      = OUT_DIR
N_CLASSES = len(CLASS_NAMES)
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 32
EPOCHS    = 30
LR        = 1e-4


class ChocolateDataset(Dataset):
    def __init__(self,
                 files, img_dir, dm_dir,
                 csv_path, class2idx,
                 target_size=(600,400),
                 affine_params=None,
                 augment=True):
        """
        + csv_path    : chemin vers le CSV VIA original
        + class2idx   : dict {classe_str → indice}
        """
        self.files       = files
        self.img_dir     = img_dir
        self.dm_dir      = dm_dir
        self.Th, self.Tw = target_size
        self.ap = affine_params or {
            'translate_frac':0.1,
            'scale_range':(0.9,1.1),
            'shear_deg':10
        }
        self.augment = augment

        # --- NOUVEAU : charger les annotations de points
        df = pd.read_csv(csv_path)
        # on s'assure que filename column matche vos files
        self.annots = {}
        for fname, grp in df.groupby("filename"):
            pts = []
            for _, row in grp.iterrows():
                sa = json.loads(row["region_shape_attributes"])
                ca = json.loads(row["region_attributes"])["Choco_Type"]
                if ca not in class2idx: 
                    continue
                pts.append((sa["cx"], sa["cy"], class2idx[ca]))
            self.annots[fname] = pts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 1) Lecture & resize image
        img  = cv2.imread(os.path.join(self.img_dir, fname))
        img  = img[:,:,::-1].astype(np.float32)/255.0
        H0,W0 = img.shape[:2]
        img  = cv2.resize(img, (self.Tw, self.Th), interpolation=cv2.INTER_AREA)

        # 2) Lire & resize les density maps
        dm = np.load(os.path.join(self.dm_dir, fname.replace(".JPG",".npz")))["density"].astype(np.float32)
        scale_area = (H0 * W0) / (self.Th * self.Tw)
        dm_chw = dm.transpose(2,0,1)
        dm_rs  = np.zeros((dm_chw.shape[0], self.Th, self.Tw), dtype=np.float32)
        for c in range(dm_chw.shape[0]):
            patch     = cv2.resize(dm_chw[c], (self.Tw, self.Th), interpolation=cv2.INTER_AREA)
            dm_rs[c]  = patch * scale_area

        # 3) Générer le mask d'impulsions gaussiennes (pour supervision des logits)
        mask = np.zeros((dm_chw.shape[0], self.Th, self.Tw), dtype=np.float32)

        for (cx, cy, cidx) in self.annots.get(fname, []):
            # coordonnée redimensionnée
            x0 = cx * (self.Tw / W0)
            y0 = cy * (self.Th / H0)
            ix = int(round(x0))
            iy = int(round(y0))

            if 0 <= ix < self.Tw and 0 <= iy < self.Th:
                x1, x2 = ix - half_k, ix + half_k + 1
                y1, y2 = iy - half_k, iy + half_k + 1

                # Découper si nécessaire pour rester dans l'image
                x1_clip, x2_clip = max(0, x1), min(self.Tw, x2)
                y1_clip, y2_clip = max(0, y1), min(self.Th, y2)

                # Portion utile du kernel
                kx1 = x1_clip - x1
                kx2 = kernel2d.shape[1] - (x2 - x2_clip)
                ky1 = y1_clip - y1
                ky2 = kernel2d.shape[0] - (y2 - y2_clip)

                patch = kernel2d[ky1:kernel2d.shape[0]-ky2, kx1:kernel2d.shape[1]-kx2]

                # Additionner dans la carte
                mask[cidx, y1_clip:y2_clip, x1_clip:x2_clip] += patch

        # 4) Application éventuelle des mêmes affines à img, dm_rs et mask
        if self.augment:
            tx = random.uniform(-self.ap['translate_frac'], self.ap['translate_frac']) * self.Tw
            ty = random.uniform(-self.ap['translate_frac'], self.ap['translate_frac']) * self.Th
            sc = random.uniform(*self.ap['scale_range'])
            sh = random.uniform(-self.ap['shear_deg'], self.ap['shear_deg'])
            a, d = sc, sc
            b     = np.tan(np.deg2rad(sh))
            c_    = 0.0
            M     = np.array([[a, b, tx],
                              [c_, d, ty]], dtype=np.float32)
            det   = a*d - b*c_

            img = cv2.warpAffine(img, M, (self.Tw,self.Th),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

            for c in range(dm_rs.shape[0]):
                dm_rs[c] = cv2.warpAffine(dm_rs[c], M, (self.Tw, self.Th),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0
                                    ) * det
                                    
                mask[c] = cv2.warpAffine(mask[c], M, (self.Tw, self.Th),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0
                                    )

        # 5) Conversion en Tensors
        img_t   = torch.from_numpy(img.transpose(2,0,1)).float()   # (3,H,W)
        dm_t    = torch.from_numpy(dm_rs).float()                   # (C,H,W)
        mask_t  = torch.from_numpy(mask).float()                    # (C,H,W) 0/1

        return img_t, dm_t, mask_t
    
    #-----------------------------DATASET ET DATALOADER-----------------------------
    # 1) Récupère tous tes fichiers .JPG
all_files = [os.path.basename(p) for p in glob(os.path.join(IMG_DIR, "*.JPG"))]

# 2) Split train / val
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# 3) Datasets
train_ds = ChocolateDataset(
    files=train_files,
    img_dir=IMG_DIR,
    dm_dir=DM_DIR,
    csv_path=CSV_PATH,           # <-- nouveau
    class2idx=class2idx,         # <-- nouveau
    target_size=(400,600),
    affine_params={'translate_frac':0.1,
                   'scale_range':(0.99,1.01),
                   'shear_deg':10},
    augment=True
)

val_ds = ChocolateDataset(
    files=val_files,
    img_dir=IMG_DIR,
    dm_dir=DM_DIR,
    csv_path=CSV_PATH,
    class2idx=class2idx,
    target_size=(400,600),
    affine_params={'translate_frac':0.1,
                   'scale_range':(0.9,1.1),
                   'shear_deg':10},
    augment=False
)
# 4) DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

#----------------------------------MODELE----------------------------------
# ─── 1. PARAMÈTRES GAUSSIEN ─────────────────────────────────────────────────────
SIGMA       = 8   # comme défini précédemment mais images resized
kern_size   = (int(6 * SIGMA) | 1)   # taille impair ≃ 6σ
half_k      = kern_size // 2

# Génération du kernel 2D gaussien normalisé
x = np.arange(-half_k, half_k+1)
g1d = np.exp(-(x**2)/(2*SIGMA**2))
g1d /= g1d.sum()
kernel2d = np.outer(g1d, g1d).astype(np.float32)  # shape (kern_size, kern_size)
# Converti en tensor pour charger dans le module PyTorch
kernel_tensor = torch.from_numpy(kernel2d)[None, None, :, :]  # (1,1,k,k)

# ─── 2. DOUBLE CONV ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# ─── 3. UNET AVEC BLUR FIXE ─────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(3,   64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128,256)
        self.down4 = DoubleConv(256,512)
        self.pool  = nn.MaxPool2d(2)
        # Decoder
        self.up3   = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv3 = DoubleConv(512,256)
        self.up2   = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv2 = DoubleConv(256,128)
        self.up1   = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv1 = DoubleConv(128,64)

        # Tête d'impulsions (sans blur)
        self.impulse_head = nn.Conv2d(64, n_classes, kernel_size=1)
        self.impulse_relu = nn.ReLU(inplace=True)

        # Blur gaussien fixe en Conv2d groupé
        self.blur = nn.Conv2d(
            in_channels=n_classes,
            out_channels=n_classes,
            kernel_size=kern_size,
            padding=half_k,
            groups=n_classes,
            bias=False
        )
        # Charger et bloquer le kernel
        with torch.no_grad():
            self.blur.weight.copy_(kernel_tensor.repeat(n_classes,1,1,1))
        self.blur.weight.requires_grad_(False)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        # Decoder
        u3 = self.up3(d4); u3 = torch.cat([u3, d3], dim=1); u3 = self.conv3(u3)
        u2 = self.up2(u3); u2 = torch.cat([u2, d2], dim=1); u2 = self.conv2(u2)
        u1 = self.up1(u2); u1 = torch.cat([u1, d1], dim=1); u1 = self.conv1(u1)

        # Impulsions + ReLU
        impulses = self.impulse_head(u1)
        impulses = self.impulse_relu(impulses)
        # Blur fixe → density maps finales
        density = self.blur(impulses)

        return density, impulses
        
def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        # Kaiming He initialization for ReLU activations
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        # gamma=1, beta=0
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model = UNet(N_CLASSES).to(DEVICE)
model.apply(init_weights)

# ─── 4. LOSS, OPTI, SCHEDULER ───────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ─── 5. FONCTIONS D’ENTRAÎNEMENT & ÉVAL ─────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-channel heatmaps (pixel-wise binary)
      FL(p,t) = -α t (1-p)^γ log(p) - (1-α)(1-t) p^γ log(1-p)
    where p = sigmoid(logit), t∈{0,1}.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,H,W), targets: same shape, binary {0,1}
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        # focal weights
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # pt = p if t==1 else (1-p)
        pt = p * targets + (1 - p) * (1 - targets)
        # focal loss
        loss = -w * (1 - pt).pow(self.gamma) * torch.log(pt)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
            

def train_epoch(loader, model,
                mse_criterion, impulse_criterion,
                optimizer,
                lambda_imp=0.5,
                lambda_l1=1e-3):
    model.train()
    running_loss = 0.0

    for imgs, dms, mask in tqdm(loader, desc="Train", leave=False):
        imgs, dms, mask = imgs.to(DEVICE), dms.to(DEVICE), mask.to(DEVICE)

        # forward → density & logits
        density, logits = model(imgs)

        # compute each loss term
        loss_mse   = mse_criterion(density, dms)         # sur la density floutée
        loss_imp   = impulse_criterion(logits, mask)     # supervision directe des impulsions
        loss_l1    = lambda_l1 * logits.abs().mean()     # régularisation L1

        # total loss
        loss = loss_mse + lambda_imp * loss_imp + loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


SAVE_DIR = "images_training"
os.makedirs(SAVE_DIR, exist_ok=True)
def eval_epoch(loader, model,
               mse_criterion, impulse_criterion,
               lambda_imp=0.5,
               lambda_l1=1e-3):
    model.eval()
    running_loss = 0.0
    visualized = True

    with torch.no_grad():
        for imgs, dms, mask in tqdm(loader, desc="Validation", leave=False):
            imgs, dms, mask = imgs.to(DEVICE), dms.to(DEVICE), mask.to(DEVICE)

            # forward → density & logits
            density, logits = model(imgs)

            # losses
            loss_mse = mse_criterion(density, dms)
            loss_imp = impulse_criterion(logits, mask)
            loss_l1  = lambda_l1 * logits.abs().mean()
            loss     = loss_mse + lambda_imp * loss_imp + loss_l1

            running_loss += loss.item() * imgs.size(0)

            # visualisation
            for i in range(imgs.size(0)):
                img_np   = imgs[i].cpu().permute(1,2,0).numpy()
                gt_den   = dms[i].sum(dim=0).cpu().numpy()
                imp_sum  = logits[i].sum(dim=0).cpu().numpy()
                pred_den = density[i].sum(dim=0).cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_np)
                axes[0].imshow(gt_den, cmap="jet", alpha=0.5)
                axes[0].set_title("GT : total density"); axes[0].axis("off")

                axes[1].imshow(img_np)
                axes[1].imshow(imp_sum, cmap="hot", alpha=0.5)
                axes[1].set_title("Impulsions"); axes[1].axis("off")

                axes[2].imshow(img_np)
                axes[2].imshow(pred_den, cmap="jet", alpha=0.5)
                axes[2].set_title("Prédiction : density"); axes[2].axis("off")

                plt.tight_layout()
                # génération d’un nom de fichier unique
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"validation_{timestamp}.png"
                save_path = os.path.join(SAVE_DIR, filename)

                # sauvegarde et fermeture
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

    loss = running_loss / len(loader.dataset)
    return loss


# ─── 6. BOUCLE PRINCIPALE ───────────────────────────────────────────────────────
best_val_loss = float('inf')
mse_criterion   = nn.MSELoss(reduction="sum")
impulse_criterion = nn.MSELoss(reduction="sum")  # ou "mean"
lambda_imp = 0.5  # poids de la perte sur les impulsions
lambda_l1            = 1e-3

for epoch in range(1, EPOCHS+1):
    tr_loss = train_epoch(train_loader, model,
                          mse_criterion, impulse_criterion,
                          optimizer, lambda_imp, lambda_l1)
    val_loss = eval_epoch(
                          val_loader, model,
                          mse_criterion, impulse_criterion,
                          lambda_imp, lambda_l1)
    scheduler.step(val_loss)

    print(f"Époch {epoch:02d} | "
          f"Train Loss: {tr_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | ")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_choco_unet.pth")
        print("→ Modèle sauvegardé!")