# extract_and_cluster.py

import os
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.cluster import KMeans                # GPU k-means
import vision_transformer as vits          # DINO library
import utils                               

# 1. Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
    help='location of the dataset')
parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--k',      type=int, default=5,
    help='number of clusters')
parser.add_argument('--out_csv', type=str, default='clusters.csv')
parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument("--checkpoint_key", default="teacher", type=str,
    help='Key to use in the checkpoint (example: "teacher")')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Load DINO
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build model according to args.arch/patch_size, adn move to device
model = vits.__dict__[args.arch](
    patch_size=args.patch_size,
    num_classes=0,           # no head
).to(device)

# load pretrained weights
utils.load_pretrained_weights(
    model,
    args.pretrained_weights, # path
    args.checkpoint_key,     # default teacher
    args.arch,               # e.g. "vit_small"
    args.patch_size,         # e.g. 16
)

# evaluation mode
model.eval()


# 3. Customize dataset
class OrganoidDataset(Dataset):
    def __init__(self, img_dir, transform, mask_suffix="_mask"):
        self.img_dir     = Path(img_dir)
        # non-mask pics
        self.img_paths   = sorted(p for p in self.img_dir.glob("*.tif")
                                  if not p.stem.endswith(mask_suffix))
        self.transform   = transform
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # mask files with the same name
        mask_path = self.img_dir / f"{img_path.stem}{self.mask_suffix}.tif" # name+"_mask"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            # min bounding‐box
            ys, xs = np.nonzero(mask)
            if len(ys) > 0 and len(xs) > 0:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                img = img.crop((x0, y0, x1 + 1, y1 + 1))

        # normalization
        img = self.transform(img)
        return img, str(img_path)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])
dataset = OrganoidDataset(args.data_path, transform)
loader  = DataLoader(dataset, batch_size=args.batch_size,
                     num_workers=args.num_workers, pin_memory=True)

# 4. Feature extract 
all_feats = []
all_paths = []
with torch.no_grad():
    for imgs, paths in loader:
        imgs = imgs.to(device)
        feats = model(imgs)                          # (B, D)
        feats = F.normalize(feats, dim=1).cpu().numpy()
        all_feats.append(feats)
        all_paths.extend(paths)

feats = np.concatenate(all_feats, axis=0)          # (N, D)

# 5. Clustering

# sklearn k-means
km = KMeans(n_clusters=args.k, n_init=20, random_state=0)
labels = km.fit_predict(feats)

# save
import pandas as pd
df = pd.DataFrame({
    'path':    all_paths,
    'cluster': labels,
})
df.to_csv(args.out_csv, index=False)
print(f"Clustering result is saved to {args.out_csv}")

