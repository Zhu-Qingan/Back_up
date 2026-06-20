"""
LGD-Depth 数据集
=================
读取 tools/build_nyuv2_lowlight.py 离线生成的数据:
    <root>/
        rgb/              # 正常光照 RGB (.png)
        rgb_lowlight/     # 离线合成低光照 RGB (.png)
        depth/            # uint16 毫米深度 (.png)

每个样本会:
    1. 按 mix_ratio 随机决定本次返回 (低光照 / 正常光照)
        默认 70% 低光照 + 30% 正常光照, 让模型同时学到两种模式
    2. 根据原图尺寸找最近的 bucket, 把 RGB / Depth 都调整到 bucket 尺寸
    3. RGB 归一化到 [-1, 1] (Marigold VAE 要求), Depth 保持 float32 米

支持 BucketBatchSampler:
    实现了 get_bucket(idx) 接口, 同一 batch 内 bucket 一致, 避免 collate 失败。
"""

import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .bucket_sampler import BucketIndexMixin, find_nearest_bucket


class LGDDepthDataset(Dataset, BucketIndexMixin):
    def __init__(
        self,
        root_dir: str,
        low_light_ratio: float = 0.7,
        is_train: bool = True,
        depth_unit_divisor: float = 1000.0,
        max_depth_meter: float = 10.0,
    ):
        """
        Args:
            root_dir: 数据根, 内含 rgb/, rgb_lowlight/, depth/
            low_light_ratio: 训练时返回低光照样本的概率 (0.0~1.0)
                            0.7 = 70% 低光照 + 30% 正常光照
                            评估时建议外部直接强制 (例如设 1.0 测低光照, 0.0 测正常)
            is_train: 训练模式. False 时关闭随机性 (固定走低光照, 用于评估)
            depth_unit_divisor: depth/ 目录 PNG 是 uint16, 除以此值得到米
                                build_nyuv2_lowlight.py 默认存毫米, 所以是 1000.0
            max_depth_meter: 深度上限 (米), 超过此值的像素视为无效 (NYUv2 室内一般 10 米)
        """
        self.root_dir = root_dir
        self.low_light_ratio = low_light_ratio
        self.is_train = is_train
        self.depth_unit_divisor = depth_unit_divisor
        self.max_depth_meter = max_depth_meter

        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.low_dir = os.path.join(root_dir, "rgb_lowlight")
        self.depth_dir = os.path.join(root_dir, "depth")

        for d in [self.rgb_dir, self.low_dir, self.depth_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"找不到必需目录: {d}")

        # 取三个目录文件名的交集, 防止有缺失
        rgb_set = {os.path.splitext(f)[0] for f in os.listdir(self.rgb_dir)}
        low_set = {os.path.splitext(f)[0] for f in os.listdir(self.low_dir)}
        depth_set = {os.path.splitext(f)[0] for f in os.listdir(self.depth_dir)}
        common = sorted(rgb_set & low_set & depth_set)
        if not common:
            raise RuntimeError(f"三个目录没有公共文件名, 请检查 {root_dir}")
        self.basenames = common
        print(f"[LGDDepthDataset] {root_dir} 共 {len(self.basenames)} 组有效样本 "
              f"(low_light_ratio={low_light_ratio})")

        # 探测一个样本的尺寸, 提前算出每张图的 bucket (避免训练时反复 IO)
        sample_path = os.path.join(self.rgb_dir, common[0] + ".png")
        sample = cv2.imread(sample_path)
        if sample is None:
            raise RuntimeError(f"读不到样本图: {sample_path}")
        self._default_hw = sample.shape[:2]  # NYUv2 应为 (480, 640)
        self._default_bucket = find_nearest_bucket(*self._default_hw)
        print(f"[LGDDepthDataset] 检测到原始尺寸 {self._default_hw}, "
              f"映射到 bucket {self._default_bucket}")

        # RGB -> [-1, 1] 的标准变换
        self._rgb_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5])

    # ---------------- Dataset 接口 ----------------
    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        name = self.basenames[idx]
        bucket_h, bucket_w = self.get_bucket(idx)

        # 1. 决定本次走低光照还是正常光照
        if self.is_train:
            use_low = random.random() < self.low_light_ratio
        else:
            use_low = True
        rgb_src_dir = self.low_dir if use_low else self.rgb_dir

        # 2. 读 RGB
        rgb_path = os.path.join(rgb_src_dir, name + ".png")
        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is None:
            raise RuntimeError(f"读不到 RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # 3. 读 Depth (uint16 mm -> float32 m)
        depth_path = os.path.join(self.depth_dir, name + ".png")
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise RuntimeError(f"读不到深度: {depth_path}")
        depth = depth_raw.astype(np.float32) / self.depth_unit_divisor  # 米

        # 4. resize 到 bucket (保比例 -> 中心 crop)
        rgb = self._resize_and_crop(rgb, bucket_h, bucket_w, is_depth=False)
        depth = self._resize_and_crop(depth, bucket_h, bucket_w, is_depth=True)

        # 5. 转 tensor + 归一化
        tensor_rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3, H, W] in [0, 1]
        tensor_rgb = self._rgb_normalize(tensor_rgb)                           # [-1, 1]
        tensor_depth = torch.from_numpy(depth).unsqueeze(0).float()            # [1, H, W] 米

        # 6. 有效掩码: 深度 > 0 且 < max_depth
        valid_mask = (tensor_depth > 1e-3) & (tensor_depth < self.max_depth_meter)

        return {
            "I_low": tensor_rgb,         # 兼容旧训练代码的命名 (实际可能是低光照也可能是正常)
            "D_gt": tensor_depth,
            "valid_mask": valid_mask,
            "is_low_light": use_low,
            "filename": name,
        }

    # ---------------- BucketIndexMixin 接口 ----------------
    def get_bucket(self, idx: int) -> Tuple[int, int]:
        # Phase 1: NYUv2 全部走同一个 bucket, 直接返回 default
        # Phase 2 加 KITTI 后, 这里改成按 idx 实际读图尺寸算
        return self._default_bucket

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _resize_and_crop(img: np.ndarray, target_h: int, target_w: int,
                         is_depth: bool) -> np.ndarray:
        """
        保宽高比 resize, 短边对齐 target, 再做中心 crop 到精确尺寸。
        - is_depth=True: 用 NEAREST, 不引入插值伪深度
        - is_depth=False: 用 LINEAR
        """
        H, W = img.shape[:2]
        scale = max(target_h / H, target_w / W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # 中心 crop
        y0 = (new_h - target_h) // 2
        x0 = (new_w - target_w) // 2
        return img[y0:y0 + target_h, x0:x0 + target_w]


# =========================================================
# Sanity check
# =========================================================
if __name__ == "__main__":
    # 服务器路径示例 (后续替换为真实路径)
    root = os.environ.get(
        "NYUV2_LGD_ROOT",
        "/workspace/team_code_codec_new/depth/datasets/NYUv2_LGD",
    )
    if not os.path.isdir(root):
        print(f"[skip] 测试需要数据集, 没找到: {root}")
        raise SystemExit(0)

    from torch.utils.data import DataLoader
    from .bucket_sampler import BucketBatchSampler

    ds = LGDDepthDataset(root_dir=root, low_light_ratio=0.7, is_train=True)
    sampler = BucketBatchSampler(ds, batch_size=2, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    for batch in loader:
        print("I_low:", batch["I_low"].shape,
              "min/max:", batch["I_low"].min().item(), batch["I_low"].max().item())
        print("D_gt:", batch["D_gt"].shape,
              "min/max:", batch["D_gt"].min().item(), batch["D_gt"].max().item())
        print("valid_mask 有效像素比例:",
              batch["valid_mask"].float().mean().item())
        print("is_low_light:", batch["is_low_light"])
        break
