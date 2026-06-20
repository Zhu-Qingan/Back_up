import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from MPyACE.model import enhance_net_nopool
from lgd_core.adapter import LGDGuidanceModule
from lgd_core.dataset import LGDDepthDataset
from lgd_core.bucket_sampler import BucketBatchSampler
from lgd_core.loss import LGDCombinedLoss

# 配置路径
MPYACE_WEIGHTS = './MPyACE/weight/Epoch999.pth'
MARIGOLD_PATH = '/workspace/team_code_codec_new/depth/checkpoint/marigold-depth-v1-1'
DATASET_ROOT = '/workspace/team_code_codec_new/depth/datasets/NYUv2_LGD'  # 服务器上 build_nyuv2_lowlight.py 的输出目录
CHECKPOINT_DIR = './checkpoints'

DEVICE = torch.device('cuda')
BATCH_SIZE = 4  # 引入 VAE 解码后，显存压力增大，建议先设为 1
LR = 1e-4
epochs = 100
SAVE_INTERVAL = 5
LOW_LIGHT_RATIO = 0.7  # 70% 低光照 + 30% 正常光照, 验证 adapter 学的是结构性先验
NUM_WORKERS = 4


def train():
    # 1. 加载所有组件 (保持 fp16 以节省显存)
    mpyace = enhance_net_nopool().to(DEVICE)
    mpyace.load_state_dict(torch.load(MPYACE_WEIGHTS, map_location=DEVICE))
    guidance_module = LGDGuidanceModule(mpyace).to(DEVICE)

    vae = AutoencoderKL.from_pretrained(MARIGOLD_PATH, subfolder="vae", torch_dtype=torch.float16).to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(MARIGOLD_PATH, subfolder="unet").to(DEVICE)
    
    # 冻结
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    noise_scheduler = DDPMScheduler.from_pretrained(MARIGOLD_PATH, subfolder="scheduler")
    
    # 预取空文本 Embedding
    tokenizer = CLIPTokenizer.from_pretrained(MARIGOLD_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MARIGOLD_PATH, subfolder="text_encoder").to(DEVICE)
    text_inputs = tokenizer([""], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        empty_text_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    del text_encoder, tokenizer

    # 2. 数据与优化器
    dataset = LGDDepthDataset(
        root_dir=DATASET_ROOT,
        low_light_ratio=LOW_LIGHT_RATIO,
        is_train=True,
    )
    batch_sampler = BucketBatchSampler(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(guidance_module.parameters(), lr=LR)
    
    # 实例化综合 Loss (包含 SILog)
    criterion = LGDCombinedLoss(lambda_latent=1.0, lambda_depth=0.1)
    
    # 在训练开始前，确保保存权重的文件夹存在
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    guidance_module.train()

    for epoch in range(epochs):
        batch_sampler.set_epoch(epoch)  # 每个 epoch 切换 RNG, 保证 shuffle 不重复
        for i, batch in enumerate(train_loader):
            img_low = batch['I_low'].to(DEVICE)
            depth_gt = batch['D_gt'].to(DEVICE) # [B, 1, 512, 512]
            
            # --- 步骤 1: 潜空间编码 ---
            with torch.no_grad():
                # 编码 RGB
                rgb_latents = vae.encode(img_low.to(dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor
                # 编码 Depth (需重复至 3 通道)
                d_min, d_max = depth_gt.min(), depth_gt.max()
                depth_norm = 2.0 * (depth_gt - d_min) / (d_max - d_min + 1e-6) - 1.0
                depth_latents = vae.encode(depth_norm.repeat(1,3,1,1).to(dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor

            # --- 步骤 2: 加噪 ---
            noise = torch.randn_like(depth_latents.to(torch.float32))
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (img_low.shape[0],), device=DEVICE).long()
            noisy_latents = noise_scheduler.add_noise(depth_latents.to(torch.float32), noise, timesteps)

            # --- 步骤 3: 引导特征提取与注入 ---
            adapter_features = guidance_module(img_low)
            mid_res = torch.zeros(img_low.shape[0], 1280, 8, 8, device=DEVICE, dtype=img_low.dtype)
            
            # --- 步骤 4: UNet 预测噪声 ---
            latent_input = torch.cat([noisy_latents, rgb_latents.to(torch.float32)], dim=1)
            noise_pred = unet(
                latent_input, timesteps, 
                encoder_hidden_states=empty_text_embeds.expand(img_low.shape[0], -1, -1),
                down_block_additional_residuals=adapter_features,
                mid_block_additional_residual=mid_res
            ).sample

            # --- 步骤 5: 反推 x0 并解码 (计算 SILog) ---
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(DEVICE)
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            
            # 【防线 1】：限制 alpha_t 的下限，防止除以极小数导致数值爆炸
            alpha_t = torch.clamp(alpha_t, min=1e-4)
            
            pred_z0 = (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # 【防线 2】：截断潜变量 z0。
            # SD 模型的 Latent 值域通常在 -5 到 5 之间，强行截断可以防止 VAE 崩溃
            pred_z0 = torch.clamp(pred_z0, min=-10.0, max=10.0)
            
            # 使用 VAE 解码回像素空间
            with torch.no_grad():
                decoded_res = vae.decode(pred_z0.to(dtype=torch.float16) / vae.config.scaling_factor).sample
                
                # 清洗 VAE 可能产生的极少数 NaN
                if torch.isnan(decoded_res).any():
                    decoded_res = torch.nan_to_num(decoded_res, nan=0.0)
                    
                pred_depth_map = (decoded_res[:, :1, :, :] + 1.0) / 2.0 * (d_max - d_min) + d_min

            # --- 步骤 6: 综合 Loss ---
            total_loss, l_latent, l_depth = criterion(noise_pred, noise, pred_depth_map.to(torch.float32), depth_gt)
            
            # 【防线 3】：NaN 熔断机制
            # 如果深度 Loss 在极端情况下还是崩了，我们用基础的 Latent Loss 兜底，保证训练不断
            if torch.isnan(total_loss):
                total_loss = l_latent
                l_depth_print = "NaN(Skipped)"
            else:
                l_depth_print = f"{l_depth.item():.4f}"

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch[{epoch}] Step[{i}] LatentLoss: {l_latent.item():.4f} SILogLoss: {l_depth_print}")
    
        # ==========================================
        # ★ 新增：Epoch 级别的权重保存逻辑
        # ==========================================
        # 加上 +1 是为了符合人类直觉 (Epoch 1 到 100，而不是 0 到 99)
        current_epoch = epoch + 1 
        
        if current_epoch % SAVE_INTERVAL == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f'lgd_adapter_epoch_{current_epoch}.pth')
            torch.save(guidance_module.state_dict(), save_path)
            print(f"\n---> [Checkpoint] 已保存阶段性权重至: {save_path}\n")

    # 循环结束后，额外保存一个 final 版本作为最终完结纪念
    final_path = os.path.join(CHECKPOINT_DIR, 'lgd_adapter_final.pth')
    torch.save(guidance_module.state_dict(), final_path)
    print(f"\n训练全部结束！最终权重已保存至: {final_path}")

    # torch.save(guidance_module.state_dict(), 'lgd_adapter_final.pth')

if __name__ == "__main__":
    train()