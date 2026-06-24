"""Treina SimpleBaseline para estimação de pose do NAO no dataset sintético COCO-pose.

Arquitetura: ResNet backbone + 3 camadas deconv (SimpleBaseline, ECCV 2018).
Saída: 17 heatmaps (um por keypoint COCO).

Dependências: torch, torchvision, opencv-python, pycocotools (opcional)

Uso:
    cd dataset_generator
    python scripts/train.py
    python scripts/train.py --epochs 100 --batch 32 --backbone resnet50
    python scripts/train.py --backbone resnet18 --epochs 50 --batch 64   # mais rápido
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Dimensões
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 256, 192   # entrada da rede (H, W)
HM_H,  HM_W  = 64,  48   # saída de heatmaps (H/4, W/4)
SIGMA         = 2.0        # desvio padrão do Gaussian GT
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# pares esquerda/direita para flip horizontal
_FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

# ---------------------------------------------------------------------------
# Modelo — SimpleBaseline
# ---------------------------------------------------------------------------

def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleBaseline(nn.Module):
    """ResNet backbone + decoder de 3 deconvs → 17 heatmaps."""

    def __init__(self, backbone: str = "resnet50", num_kp: int = 17):
        super().__init__()
        net = getattr(models, backbone)(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        in_ch = 2048 if backbone in ("resnet50", "resnet101", "resnet152") else 512
        self.decoder = nn.Sequential(
            _deconv_block(in_ch, 256),
            _deconv_block(256,   256),
            _deconv_block(256,   256),
        )
        self.head = nn.Conv2d(256, num_kp, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.backbone(x)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _make_gaussian(cy: float, cx: float, sigma: float = SIGMA) -> np.ndarray:
    hm = np.zeros((HM_H, HM_W), dtype=np.float32)
    x0, y0 = round(cx), round(cy)
    half = int(3 * sigma)
    x1, x2 = max(0, x0 - half), min(HM_W, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(HM_H, y0 + half + 1)
    if x1 >= x2 or y1 >= y2:
        return hm
    xs = np.arange(x1, x2) - x0
    ys = np.arange(y1, y2) - y0
    xx, yy = np.meshgrid(xs, ys)
    hm[y1:y2, x1:x2] = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return hm


class NaoPoseDataset(Dataset):

    def __init__(self, img_root: Path, ann_file: Path, augment: bool = False):
        self.img_root = img_root
        self.augment  = augment
        with open(ann_file) as f:
            raw = json.load(f)
        ann_by_id = {a["image_id"]: a for a in raw["annotations"]}
        self.samples = [
            (im, ann_by_id[im["id"]])
            for im in raw["images"]
            if im["id"] in ann_by_id
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_info, ann = self.samples[idx]
        path = self.img_root / img_info["file_name"]
        img  = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        img = (img - MEAN) / STD

        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)

        flip = self.augment and np.random.rand() < 0.5
        if flip:
            img = img[:, ::-1, :]
            kps[:, 0] = orig_w - kps[:, 0]
            for a, b in _FLIP_PAIRS:
                kps[[a, b]] = kps[[b, a]]

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy())

        sx = (HM_W / orig_w)
        sy = (HM_H / orig_h)
        heatmaps = np.zeros((17, HM_H, HM_W), dtype=np.float32)
        weights  = np.zeros(17, dtype=np.float32)
        for k in range(17):
            x, y, v = kps[k]
            if v == 0:
                continue
            heatmaps[k] = _make_gaussian(y * sy, x * sx)
            weights[k]  = 1.0 if v == 2 else 0.5

        return img_t, torch.from_numpy(heatmaps), torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MaskedMSE(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 17, H, W)   weights: (B, 17)
        loss = ((pred - target) ** 2).mean(dim=(2, 3))   # (B, 17)
        return (loss * weights).sum() / (weights.sum() + 1e-6)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def decode_heatmaps(hm: torch.Tensor) -> torch.Tensor:
    """Argmax → coordenadas (B, 17, 2) em pixels do heatmap."""
    B, K, H, W = hm.shape
    flat = hm.view(B, K, -1)
    idx  = flat.argmax(dim=2)
    return torch.stack([idx % W, idx // W], dim=2).float()


@torch.no_grad()
def pck_accuracy(pred: torch.Tensor, gt: torch.Tensor,
                 weights: torch.Tensor, thr: float = 0.2) -> tuple[float, float]:
    """PCK@thr normalizado pelo diagonal do heatmap."""
    diag    = (HM_H**2 + HM_W**2) ** 0.5
    dist    = (decode_heatmaps(pred) - decode_heatmaps(gt)).norm(dim=2) / diag
    mask    = weights > 0
    correct = ((dist < thr) & mask).float().sum().item()
    total   = mask.float().sum().item()
    return correct, total


# ---------------------------------------------------------------------------
# Loop de treino
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    root   = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device={device}  backbone={args.backbone}  "
          f"epochs={args.epochs}  batch={args.batch}")

    train_ds = NaoPoseDataset(root / "images",
                              root / "annotations/person_keypoints_train.json",
                              augment=True)
    val_ds   = NaoPoseDataset(root / "images",
                              root / "annotations/person_keypoints_val.json",
                              augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=device.type == "cuda")

    model     = SimpleBaseline(backbone=args.backbone).to(device)
    criterion = MaskedMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                            eta_min=args.lr * 0.01)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pck  = 0.0

    for epoch in range(1, args.epochs + 1):

        # ---- treino ----
        model.train()
        train_loss = 0.0
        for imgs, hms, wts in train_dl:
            imgs, hms, wts = imgs.to(device), hms.to(device), wts.to(device)
            loss = criterion(model(imgs), hms, wts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # ---- validação ----
        model.eval()
        val_loss = correct = total = 0.0
        with torch.no_grad():
            for imgs, hms, wts in val_dl:
                imgs, hms, wts = imgs.to(device), hms.to(device), wts.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, hms, wts).item()
                c, t = pck_accuracy(pred, hms, wts)
                correct += c; total += t
        pck = correct / total if total > 0 else 0.0

        print(f"[{epoch:03d}/{args.epochs}]  "
              f"loss={train_loss/len(train_dl):.4f}  "
              f"val={val_loss/len(val_dl):.4f}  "
              f"PCK@0.2={pck:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {"epoch": epoch, "pck": pck,
                "backbone": args.backbone, "state_dict": model.state_dict()}
        torch.save(ckpt, out_dir / "last.pt")
        if pck > best_pck:
            best_pck = pck
            torch.save(ckpt, out_dir / "best.pt")
            print(f"           ↳ novo melhor PCK={best_pck:.3f}  → {out_dir}/best.pt")

    print(f"\n[TRAIN] Concluído. Melhor PCK@0.2 = {best_pck:.3f}")


# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts/
_DATASET_DIR = _SCRIPT_DIR.parent / "output"          # .../output/
_CKPT_DIR    = _SCRIPT_DIR.parent / "checkpoints"     # .../checkpoints/

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(_DATASET_DIR),
                   help="Pasta output/ com images/ e annotations/")
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--output",   default=str(_CKPT_DIR),
                   help="Pasta para salvar checkpoints")
    train(p.parse_args())
