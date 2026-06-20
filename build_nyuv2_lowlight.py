"""
NYUv2 离线数据构建脚本
=========================
功能：
    1. 从 nyu_depth_v2_labeled.mat (1449 张精标) 提取 RGB + Depth
    2. 用多策略组合，离线合成低光照版本
    3. 抽样生成可视化对比图，供训练前人工审查

输出目录结构：
    <out_root>/
        rgb/              # 正常光照 RGB (.png, 480x640)
        rgb_lowlight/     # 合成低光照 RGB (.png, 480x640)
        depth/            # 深度真值 (.png, uint16, 单位: 毫米)
        visualization/    # 抽样对比图 (每 N 张采样一张, 3 联拼接)
        meta.json         # 每张图采用的退化策略与参数 (可复现)

使用方式 (服务器示例)：
    python tools/build_nyuv2_lowlight.py \
        --mat_path /workspace/team_code_codec_new/depth/datasets/nyu_depth_v2_labeled.mat \
        --out_root /workspace/team_code_codec_new/depth/datasets/NYUv2_LGD \
        --vis_stride 50
"""

import argparse
import json
import os
import random

import cv2
import h5py
import numpy as np
from tqdm import tqdm


# =========================================================
# 低光照退化策略库 (离线一次性生成，不参与训练时随机)
# =========================================================
def degrade_gamma(img, gamma):
    """Gamma 降暗：模拟曝光不足。gamma > 1 时图像变暗。"""
    img_f = img.astype(np.float32) / 255.0
    out = np.power(img_f, gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def degrade_linear(img, alpha):
    """线性降暗：模拟光线不足。alpha < 1 时图像变暗。"""
    out = img.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def add_poisson_gaussian_noise(img, read_noise_sigma, shot_noise_scale):
    """
    暗光下的物理噪声模型：泊松 (shot noise) + 高斯 (read noise)。
    暗光场景下信噪比下降，噪声变得明显，是低光照图最显著的特征之一。

    read_noise_sigma: 读出噪声 (固定方差高斯), 推荐 2~8 (8-bit 空间)
    shot_noise_scale: 散粒噪声强度, 推荐 0.5~2.0
    """
    img_f = img.astype(np.float32)
    # Shot noise: 方差与信号成正比 (泊松近似为高斯)
    shot = np.random.randn(*img.shape).astype(np.float32) * np.sqrt(np.maximum(img_f, 0)) * shot_noise_scale
    # Read noise: 固定方差
    read = np.random.randn(*img.shape).astype(np.float32) * read_noise_sigma
    out = img_f + shot + read
    return np.clip(out, 0, 255).astype(np.uint8)


def shift_color_temperature(img, scale_r, scale_b):
    """色温偏移：模拟非标准光源 (钨丝灯偏黄, 月光偏蓝)。"""
    img_f = img.astype(np.float32)
    # OpenCV 读入是 BGR
    img_f[:, :, 0] *= scale_b  # B
    img_f[:, :, 2] *= scale_r  # R
    return np.clip(img_f, 0, 255).astype(np.uint8)


def sample_degradation_params(rng):
    """
    随机采样一组退化参数。
    每张图都走完整的 [曝光降低 -> 色温偏移 -> 噪声注入] 流水线，
    只是各阶段强度不同，这样能覆盖最广的低光照分布。
    """
    # === 曝光降低：gamma 和 linear 二选一 ===
    if rng.random() < 0.5:
        exposure_mode = "gamma"
        gamma = rng.uniform(2.0, 4.5)
        alpha = None
    else:
        exposure_mode = "linear"
        gamma = None
        alpha = rng.uniform(0.08, 0.35)

    # === 色温偏移：偏暖 / 偏冷 / 不偏 ===
    color_mode = rng.choices(
        ["warm", "cool", "neutral"],
        weights=[0.35, 0.35, 0.30],
    )[0]
    if color_mode == "warm":  # 钨丝灯/路灯，R 多 B 少
        scale_r = rng.uniform(1.05, 1.20)
        scale_b = rng.uniform(0.70, 0.90)
    elif color_mode == "cool":  # 月光/荧光，B 多 R 少
        scale_r = rng.uniform(0.70, 0.90)
        scale_b = rng.uniform(1.05, 1.20)
    else:
        scale_r = 1.0
        scale_b = 1.0

    # === 噪声强度 ===
    read_noise_sigma = rng.uniform(3.0, 10.0)
    shot_noise_scale = rng.uniform(0.6, 1.8)

    return {
        "exposure_mode": exposure_mode,
        "gamma": gamma,
        "alpha": alpha,
        "color_mode": color_mode,
        "scale_r": scale_r,
        "scale_b": scale_b,
        "read_noise_sigma": read_noise_sigma,
        "shot_noise_scale": shot_noise_scale,
    }


def apply_degradation(img_bgr, params):
    """按参数对 BGR 图像执行完整退化流水线。"""
    out = img_bgr.copy()

    # 1. 曝光降低
    if params["exposure_mode"] == "gamma":
        out = degrade_gamma(out, params["gamma"])
    else:
        out = degrade_linear(out, params["alpha"])

    # 2. 色温偏移
    if params["color_mode"] != "neutral":
        out = shift_color_temperature(out, params["scale_r"], params["scale_b"])

    # 3. 噪声注入
    out = add_poisson_gaussian_noise(
        out,
        read_noise_sigma=params["read_noise_sigma"],
        shot_noise_scale=params["shot_noise_scale"],
    )
    return out


# =========================================================
# 可视化：把 (正常 | 低光照 | 深度伪彩) 三联拼接保存
# =========================================================
def colorize_depth(depth_m, max_depth=10.0):
    """深度图 (米) -> 伪彩色 BGR"""
    d = np.clip(depth_m, 0, max_depth) / max_depth
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def save_visualization(rgb_bgr, low_bgr, depth_m, params, out_path):
    depth_vis = colorize_depth(depth_m)
    triplet = np.concatenate([rgb_bgr, low_bgr, depth_vis], axis=1)

    # 在图像底部加一条文字带，写明退化参数
    H, W = triplet.shape[:2]
    band = np.zeros((40, W, 3), dtype=np.uint8)
    text = f"exp={params['exposure_mode']}"
    if params["exposure_mode"] == "gamma":
        text += f"(g={params['gamma']:.2f})"
    else:
        text += f"(a={params['alpha']:.2f})"
    text += f" | color={params['color_mode']}(r={params['scale_r']:.2f},b={params['scale_b']:.2f})"
    text += f" | noise(read={params['read_noise_sigma']:.1f},shot={params['shot_noise_scale']:.2f})"
    cv2.putText(band, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.concatenate([triplet, band], axis=0)
    cv2.imwrite(out_path, final)


# =========================================================
# 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", type=str, required=True,
                        help="nyu_depth_v2_labeled.mat 的完整路径")
    parser.add_argument("--out_root", type=str, required=True,
                        help="输出根目录")
    parser.add_argument("--vis_stride", type=int, default=50,
                        help="每隔多少张采样一张做可视化对比 (默认 50, 即 ~30 张供审查)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子, 保证可复现")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # 准备输出目录
    rgb_dir = os.path.join(args.out_root, "rgb")
    low_dir = os.path.join(args.out_root, "rgb_lowlight")
    depth_dir = os.path.join(args.out_root, "depth")
    vis_dir = os.path.join(args.out_root, "visualization")
    for d in [rgb_dir, low_dir, depth_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"正在加载 NYUv2 mat 文件: {args.mat_path}")
    # NYUv2 v7.3 是 HDF5 格式，必须用 h5py 读 (scipy.io.loadmat 读不了)
    with h5py.File(args.mat_path, "r") as f:
        # 数据布局: images [3, 640, 480, N], depths [640, 480, N]  (matlab 列主序)
        images = f["images"]  # uint8
        depths = f["depths"]  # float32, 单位: 米
        N = images.shape[0]
        print(f"共找到 {N} 组 (RGB, Depth) 对")

        meta = {}

        for i in tqdm(range(N), desc="构建数据"):
            # --- 1. 读取 + 转置回 (H, W, C) ---
            # mat 是 [C, W, H]，需要转置到 [H, W, C]
            img_rgb = np.array(images[i])  # [3, 640, 480]
            img_rgb = np.transpose(img_rgb, (2, 1, 0))  # [480, 640, 3]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            depth = np.array(depths[i])  # [640, 480]
            depth = np.transpose(depth, (1, 0))  # [480, 640], float32 米

            # --- 2. 采样退化参数 + 应用 ---
            params = sample_degradation_params(rng)
            img_low_bgr = apply_degradation(img_bgr, params)

            # --- 3. 保存 ---
            fname = f"{i:05d}"

            # RGB 正常 (.png 无损)
            cv2.imwrite(os.path.join(rgb_dir, f"{fname}.png"), img_bgr)
            # RGB 低光照
            cv2.imwrite(os.path.join(low_dir, f"{fname}.png"), img_low_bgr)
            # 深度: 转 uint16 毫米 (跟 lgd_core/dataset.py 当前的 /1000.0 约定一致)
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, f"{fname}.png"), depth_mm)

            # 可视化抽样
            if i % args.vis_stride == 0:
                save_visualization(
                    img_bgr, img_low_bgr, depth,
                    params,
                    os.path.join(vis_dir, f"vis_{fname}.png"),
                )

            # 记录元信息 (可复现)
            meta[fname] = params

        # 保存所有退化参数, 供事后审计
        with open(os.path.join(args.out_root, "meta.json"), "w") as jf:
            json.dump(meta, jf, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print(f"构建完成！数据已写入: {args.out_root}")
    print(f"  - rgb/             ({N} 张正常光照)")
    print(f"  - rgb_lowlight/    ({N} 张离线合成低光照)")
    print(f"  - depth/           ({N} 张 uint16 毫米深度)")
    print(f"  - visualization/   ({N // args.vis_stride + 1} 张三联对比图, 请人工审查)")
    print(f"  - meta.json        (每张图的退化参数, 供复现)")
    print("=" * 60)
    print("\n[下一步] 请打开 visualization/ 文件夹, 抽查 20~30 张:")
    print("  ✓ 低光照看起来像真实暗光照片 (而不是简单压暗)")
    print("  ✓ 噪声明显但不至于完全看不出内容")
    print("  ✓ 色温偏移自然 (不会出现荧光绿/紫色等奇怪颜色)")
    print("  ✗ 如有问题, 调整 sample_degradation_params() 的范围后重新运行")


if __name__ == "__main__":
    main()
