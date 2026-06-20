"""
Aspect-Ratio Bucket Sampler
============================
为 SD 系扩散模型 (Marigold 基于 SD 1.5) 设计的多分辨率训练采样器，
借鉴 SDXL / Qwen-VL 的动态分辨率思想，但适配 UNet 架构。

设计要点:
    1. 预定义若干 bucket (宽高对), 每个 bucket 像素总量接近 512² ≈ 262144
    2. 每个 bucket 宽高都是 8 的倍数 (SD VAE 要求 8 倍下采样)
    3. 同一 batch 内只放同一个 bucket 的图, 保证 collate 不出错
    4. 训练时按 bucket 内样本数加权随机抽 bucket, 避免小 bucket 饿死

Phase 1 (NYUv2-only) 实际只会用到一个 bucket (480×640 → 512×640 类),
但代码已经支持后续 Phase 2 (KITTI) 直接挂上。
"""

from collections import defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler


# =========================================================
# 预定义 bucket 列表
# 每行: (height, width), 都是 8 的倍数, 像素总量 ≈ 262144
# 按宽高比从方形到超宽幅排序, 方便检索
# =========================================================
DEFAULT_BUCKETS: List[Tuple[int, int]] = [
    # 方形 (1:1) - 主用于正方形增广
    (512, 512),   # 262144
    # 4:3 横向 (NYUv2 原生 480:640 = 3:4, 但我们做 NHWC, H=480 W=640 是 3:4)
    (448, 576),   # 258048  ← NYUv2 (480x640) 最近邻
    (480, 640),   # 307200  ← NYUv2 原生比例, 稍大一点
    # 16:9 横向
    (384, 704),   # 270336
    # KITTI 风格 宽幅 (~1:3.3)
    (320, 832),   # 266240
    (288, 928),   # 267264
    (256, 1024),  # 262144
    (256, 1056),  # 270336  ← KITTI (375x1242) 最近邻
]


def find_nearest_bucket(
    h: int,
    w: int,
    buckets: Sequence[Tuple[int, int]] = DEFAULT_BUCKETS,
) -> Tuple[int, int]:
    """
    根据原始图像 (h, w) 找最接近的 bucket。
    距离度量: 宽高比的 log 差 (不受绝对大小影响)。

    返回: (bucket_h, bucket_w)
    """
    src_ratio = h / max(w, 1)
    best = None
    best_diff = float("inf")
    for bh, bw in buckets:
        bucket_ratio = bh / bw
        # 用 log 差对称比较宽高比 (4:3 vs 3:4 在 log 空间距离相同)
        diff = abs(torch.log(torch.tensor(src_ratio)).item()
                   - torch.log(torch.tensor(bucket_ratio)).item())
        if diff < best_diff:
            best_diff = diff
            best = (bh, bw)
    return best


# =========================================================
# 数据集侧 mixin: 每个样本能告诉 sampler 自己属于哪个 bucket
# =========================================================
class BucketIndexMixin:
    """
    Dataset 通过实现 get_bucket(idx) -> (h, w) 来告诉 sampler 自己的归属。
    LGDDepthDataset 会 mix-in 这个接口。
    """

    def get_bucket(self, idx: int) -> Tuple[int, int]:
        raise NotImplementedError(
            "Dataset 必须实现 get_bucket(idx) 才能配合 BucketBatchSampler 使用"
        )


# =========================================================
# Batch Sampler: 按 bucket 分组, 每次产出同一 bucket 的 batch
# =========================================================
class BucketBatchSampler(Sampler[List[int]]):
    """
    用法:
        sampler = BucketBatchSampler(
            dataset, batch_size=4, shuffle=True, drop_last=True
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4,
                            collate_fn=...)

    工作原理:
        1. 启动时扫一遍 dataset, 调 get_bucket(i) 把所有样本分组
        2. 每个 epoch 开始时, 打乱每个 bucket 内的索引
        3. 每个 bucket 切成多个 batch_size 大小的 chunk
        4. 把所有 chunks 拼起来再整体 shuffle 一次, 产出 batch
    """

    def __init__(
        self,
        dataset: BucketIndexMixin,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if not hasattr(dataset, "get_bucket"):
            raise TypeError(
                "dataset 必须实现 get_bucket(idx) 接口 (继承 BucketIndexMixin)"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # 启动时把所有样本按 bucket 分组
        self.bucket_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            bucket = dataset.get_bucket(i)
            self.bucket_to_indices[bucket].append(i)

        # 打印 bucket 分布, 方便确认数据是否均衡
        print("[BucketBatchSampler] 各 bucket 样本数:")
        for bucket, idxs in sorted(self.bucket_to_indices.items()):
            print(f"  {bucket[0]:4d} x {bucket[1]:4d}  →  {len(idxs):6d} 张")

        # 预计算 batch 数
        self._num_batches = 0
        for idxs in self.bucket_to_indices.values():
            n = len(idxs)
            if self.drop_last:
                self._num_batches += n // self.batch_size
            else:
                self._num_batches += (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        """DDP 训练时, 每个 epoch 切换 seed, 保证多卡 shuffle 一致。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # 每个 epoch 独立的 RNG, 保证可复现
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches: List[List[int]] = []
        for bucket, indices in self.bucket_to_indices.items():
            indices = list(indices)  # copy
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[p] for p in perm]

            # 切 batch
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start: start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # 再 shuffle 一次, 让不同 bucket 的 batch 交错出现
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[p] for p in perm]

        yield from all_batches

    def __len__(self) -> int:
        return self._num_batches


# =========================================================
# Sanity check
# =========================================================
if __name__ == "__main__":
    # 测试: 模拟一个 dataset, 验证 bucket 分组正确
    class _MockDataset(BucketIndexMixin):
        def __init__(self):
            # 模拟 100 张 NYUv2 (480x640) + 50 张 KITTI (375x1242)
            self.samples = (
                [(480, 640)] * 100 + [(375, 1242)] * 50
            )

        def __len__(self):
            return len(self.samples)

        def get_bucket(self, idx):
            h, w = self.samples[idx]
            return find_nearest_bucket(h, w)

    ds = _MockDataset()
    sampler = BucketBatchSampler(ds, batch_size=4, shuffle=True, drop_last=True)
    print(f"\n总 batch 数: {len(sampler)}")

    # 抽样几个 batch, 确认同一 batch 内 bucket 相同
    print("\n前 3 个 batch 的 bucket 验证:")
    for i, batch in enumerate(sampler):
        buckets = set(ds.get_bucket(idx) for idx in batch)
        assert len(buckets) == 1, f"❌ 同一 batch 出现多个 bucket: {buckets}"
        print(f"  Batch {i}: indices={batch}, bucket={buckets.pop()}")
        if i >= 2:
            break
    print("\n✓ 同一 batch 内 bucket 一致, sampler 工作正常")
