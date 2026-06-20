LGD-Depth-Project/
├── tools/                                # 新增目录
│   ├── build_nyuv2_lowlight.py          # 离线生成 NYUv2 低光照数据 + 可视化抽样
│   └── visualize_dataset.py             # Dataset 端到端审查工具
├── lgd_core/
│   ├── __init__.py                       # 新增 (聚合导出)
│   ├── bucket_sampler.py                 # 新增 (Aspect-Ratio Bucketing)
│   └── dataset.py                        # 重写 (支持 bucket + 70/30 混合)
└── train_lgd.py                          # 改 (接入 bucket sampler)


# 1) 同步代码到服务器后, 先生成数据 (NYUv2 mat 路径替换为实际)
python tools/build_nyuv2_lowlight.py \
    --mat_path /path/to/nyu_depth_v2_labeled.mat \
    --out_root /workspace/team_code_codec_new/depth/datasets/NYUv2_LGD \
    --vis_stride 50

# 2) 打开 NYUv2_LGD/visualization/ 抽查 ~30 张三联图
#    - 不满意就调 build_nyuv2_lowlight.py 的 sample_degradation_params() 重跑

# 3) 数据通过审查后, 端到端验证 dataset
python tools/visualize_dataset.py \
    --root /workspace/team_code_codec_new/depth/datasets/NYUv2_LGD \
    --out_dir ./debug_vis --num 16

# 4) 检查 ./debug_vis/ 16 张 (RGB | Depth | Mask) 三联图, 重点确认:
#    - 同 batch 同分辨率
#    - LOW/NORM 比例接近 70:30
#    - mask 白色覆盖物体表面

# 5) 全部通过 → 启动训练
python train_lgd.py


关键设计点回顾
退化策略：曝光降低（gamma/linear）→ 色温偏移（暖/冷/中性）→ 物理噪声（泊松+高斯），三阶段流水线，每张图独立采样参数，meta.json 记录可复现
Bucket 设计：8 个 bucket 覆盖方形 → 16:9 → KITTI 超宽幅，每个像素总量 ≈ 262144，全部 8 倍数对齐 SD VAE
70/30 混合：训练时按概率切 rgb_lowlight/ 或 rgb/，dataset 返回 is_low_light flag，后续可以用来做条件分析
审查双保险：build 阶段抽样可视化 + dataset 阶段端到端可视化，两道关卡确保数据进训练时已经被肉眼确认过

服务器上 NYUv2 mat 路径后续需要修改，把 train_lgd.py 第 17 行的 DATASET_ROOT 改一下就行。下一步 Phase 2 加 KITTI 的时候，bucket_sampler.py 不用改，只要扩展 dataset.py 的 get_bucket() 让它按图实际尺寸返回 bucket 即可