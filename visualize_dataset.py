"""
Dataset 端到端可视化审查
==========================
在正式训练前最后一道防线: 直接调用 LGDDepthDataset / BucketBatchSampler,
把产出的 tensor 反归一化回图像保存, 检查:
    - bucket 尺寸对齐是否正确
    - low_light_ratio 混合比例是否生效
    - 归一化 / crop 是否引入伪影
    - valid_mask 是否合理

用法:
    python tools/visualize_dataset.py \
        --root /workspace/.../NYUv2_LGD \
        --out_dir ./debug_vis \
        --num 16
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lgd_core.bucket_sampler import BucketBatchSampler
from lgd_core.dataset import LGDDepthDataset


def tensor_to_bgr(rgb_tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] 的 RGB tensor [3, H, W] -> uint8 BGR [H, W, 3]"""
    x = rgb_tensor.detach().cpu().numpy()
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def depth_to_color(depth_tensor: torch.Tensor, max_depth: float = 10.0) -> np.ndarray:
    """[1, H, W] 米 -> 伪彩色 BGR [H, W, 3]"""
    d = depth_tensor.squeeze(0).detach().cpu().numpy()
    d = np.clip(d, 0, max_depth) / max_depth
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def mask_to_gray(mask_tensor: torch.Tensor) -> np.ndarray:
    """[1, H, W] bool -> 灰度 BGR (白=有效, 黑=无效)"""
    m = mask_tensor.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="LGDDepthDataset 根目录")
    parser.add_argument("--out_dir", type=str, default="./debug_vis")
    parser.add_argument("--num", type=int, default=16,
                        help="可视化样本数")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--low_light_ratio", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = LGDDepthDataset(
        root_dir=args.root,
        low_light_ratio=args.low_light_ratio,
        is_train=True,
    )
    sampler = BucketBatchSampler(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    n_low = 0
    n_normal = 0
    saved = 0
    for batch_idx, batch in enumerate(loader):
        B = batch["I_low"].shape[0]
        for b in range(B):
            rgb_bgr = tensor_to_bgr(batch["I_low"][b])
            depth_bgr = depth_to_color(batch["D_gt"][b])
            mask_bgr = mask_to_gray(batch["valid_mask"][b])

            is_low = bool(batch["is_low_light"][b])
            n_low += int(is_low)
            n_normal += int(not is_low)

            # 给图像加标签
            tag = "LOW" if is_low else "NORM"
            label_band = np.zeros((30, rgb_bgr.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                label_band,
                f"[{tag}] {batch['filename'][b]} | shape={rgb_bgr.shape[:2]}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            rgb_with_label = np.concatenate([label_band, rgb_bgr], axis=0)

            # 拼接到深度和 mask 上面要先对齐高度
            h = rgb_with_label.shape[0]
            depth_pad = np.zeros_like(rgb_with_label)
            depth_pad[-depth_bgr.shape[0]:, :, :] = depth_bgr
            mask_pad = np.zeros_like(rgb_with_label)
            mask_pad[-mask_bgr.shape[0]:, :, :] = mask_bgr

            triplet = np.concatenate([rgb_with_label, depth_pad, mask_pad], axis=1)

            out_path = os.path.join(args.out_dir, f"sample_{saved:03d}.png")
            cv2.imwrite(out_path, triplet)
            saved += 1
            if saved >= args.num:
                break
        if saved >= args.num:
            break

    print("\n" + "=" * 60)
    print(f"已保存 {saved} 个样本到 {args.out_dir}/")
    print(f"低光照 : 正常光照 = {n_low} : {n_normal} "
          f"(目标 {args.low_light_ratio:.0%} : {1 - args.low_light_ratio:.0%})")
    print("=" * 60)
    print("\n[审查清单]")
    print("  ✓ 同一 batch 内所有图分辨率一致 (检查 bucket sampler)")
    print("  ✓ RGB 看起来没有伪影 (检查 normalize 反演正确)")
    print("  ✓ 深度伪彩颜色梯度自然 (检查 NEAREST resize 没破坏)")
    print("  ✓ valid_mask 白色区域大致覆盖物体表面 (剔除 0 和 >10m 的像素)")
    print("  ✓ LOW / NORM 标签比例符合 low_light_ratio")


if __name__ == "__main__":
    main()
