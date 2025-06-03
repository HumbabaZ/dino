#!/usr/bin/env python3
# show_clusters.py
"""
visualization: put images in the same cluster together
command: 
    python show_clusters.py \
       --csv clusters.csv \
       --cluster 5 \
       --nrow 4 \
       --thumb 96 \
       --out cluster5.jpg
    python show_clusters.py --csv clusters.csv --cluster 4 --out output/cluster4.jpg
"""

import argparse
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def show_cluster_montage(
    df: pd.DataFrame,
    cluster_id: int,
    nrow: int = 3,
    thumb_size: tuple[int, int] = (64, 64),
    save_path: Optional[str] = None,
) -> None:
    
    w, h = thumb_size
    num_pat = re.compile(r"_(\d+)\.\w+$")  # numbers in pic names
    font = ImageFont.load_default()        

    # pic paths
    #paths = df[df["cluster"] == cluster_id]["path"].tolist()[: nrow * nrow]
    mask = df["cluster"] == cluster_id  # rows in the cluster
    path_series = df[mask]["path"]  
    paths_total = path_series.tolist() # paths of the pics
    paths=paths_total[: nrow * nrow]
    if len(paths) == 0:
        raise ValueError(f"Cluster {cluster_id} has no picture")

    # read pics and normalization
    imgs: list[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(thumb_size, Image.BILINEAR)
        imgs.append(np.asarray(img))

    # if no enough picture, use blank pics
    total = nrow * nrow
    if len(imgs) < total:
        blank = np.ones((*thumb_size, 3), dtype=np.uint8) * 255
        imgs.extend([blank] * (total - len(imgs)))

    # put together
    rows = [np.hstack(imgs[i * nrow : (i + 1) * nrow]) for i in range(nrow)]
    grid_np = np.vstack(rows)

    # save to file
    if save_path:
        Image.fromarray(grid_np).save(save_path)

    # write numbers on the pics
    grid = Image.fromarray(grid_np)
    draw = ImageDraw.Draw(grid)
    for idx, p in enumerate(paths):
        r, c = divmod(idx, nrow)
        x, y = c * w + 2, r * h + 2  # location: top left +2px
        m = num_pat.search(p)
        label = m.group(1) if m else str(idx)
        draw.text((x, y), label, fill=(255, 0, 0), font=font)

    # save as .png(Matplotlib)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(f"Cluster {cluster_id}  |  has {len(paths_total)} samples")

    # save another version
    if save_path:
        fig_path = Path(save_path).with_suffix(".png")
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)

    # plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a cluster montage")
    parser.add_argument("--csv", required=True, help="Path to clusters.csv")
    parser.add_argument(
        "--cluster", required=True, type=int, help="Cluster ID to visualize"
    )
    parser.add_argument("--nrow", default=3, type=int, help="Grid dimension (nrow x nrow)")
    parser.add_argument(
        "--thumb", default=64, type=int, help="Thumbnail side length in pixels"
    )
    parser.add_argument(
        "--out", default=None, help="If set, save the montage to this path"
    )
    args = parser.parse_args()

    # load CSV
    df = pd.read_csv(args.csv)

    show_cluster_montage(
        df,
        cluster_id=args.cluster,
        nrow=args.nrow,
        thumb_size=(args.thumb, args.thumb),
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
