import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random  # ✅ 新增：用于打乱图片顺序

# 直接导入光照增强模块
from model_snn_cnn import HDREnhancer

# ================= 配置区域 =================
CHECKPOINT_PATH = '/root/Single-eye-Emotion-Recognition/single-eye-emotion/results/resnet_cmm_test/save_110.pth'
DATA_ROOT = Path('./SEE/frame')
OUTPUT_DIR = Path('visible_low_light_results')

# 亮度筛选范围：选 15-80 之间
BRIGHTNESS_MIN = 15
BRIGHTNESS_MAX = 80
# ===========================================

def get_brightness(img_path):
    img = cv2.imread(str(img_path))
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def find_appropriate_images(root_path, count=6):
    print(f"🕵️ 正在寻找亮度在 [{BRIGHTNESS_MIN}, {BRIGHTNESS_MAX}] 之间的图片...")
    candidates = []
    
    # ✅ 新增：用于记录已经选过的视频和情绪，避免重复
    used_video_folders = set()
    used_emotions = set()

    all_imgs = list(root_path.rglob('*.jpg'))
    
    # ✅ 核心修改：打乱所有图片的顺序，不再按顺序死磕一个视频
    random.seed(42) # 固定随机种子，保证每次跑出来的6张图是固定的，方便对比。如果想每次随机，删掉这行。
    random.shuffle(all_imgs)

    for p in all_imgs:
        if len(candidates) >= count: 
            break

        # 获取图片的父文件夹（视频编号）和爷爷文件夹（情绪标签）
        # 路径结构一般是：SEE/frame/angry/video1/00001.jpg
        video_folder = str(p.parent)
        emotion_label = p.parent.parent.name 

        # ✅ 核心逻辑：同一个视频只准取1张图
        if video_folder in used_video_folders:
            continue

        b = get_brightness(p)
        if BRIGHTNESS_MIN < b < BRIGHTNESS_MAX:
            # 找到符合亮度的图后，记录下它的视频和情绪
            candidates.append(p)
            used_video_folders.add(video_folder)
            used_emotions.add(emotion_label)

    print(f"✅ 成功挑选出 {len(candidates)} 张来自不同视频的图片！")
    print(f"🎭 包含的情绪类别有: {', '.join(used_emotions)}")
    return candidates

def gamma_correction(image, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image.astype(np.uint8), table)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hdr_module = HDREnhancer(alpha=0.4).to(device)
    
    if Path(CHECKPOINT_PATH).exists():
        print(f"📥 Loading weights: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        hdr_weights = {k.replace('hdr_enhancer.', ''): v 
                       for k, v in state_dict.items() if 'hdr_enhancer' in k}
        if len(hdr_weights) > 0:
            hdr_module.load_state_dict(hdr_weights, strict=False)
            print("✅ 成功加载已训练的 LACM 光照补偿模块权重！")
            
    hdr_module.eval()

    images = find_appropriate_images(DATA_ROOT, count=6)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("⚡ 正在处理图片并生成特征图...")
    processed_results = []

    for img_path in images:
        raw_pil = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(raw_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_tensor = hdr_module(input_tensor)

        img_in = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_out = enhanced_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_in = np.clip(img_in, 0, 1)
        img_out = np.clip(img_out, 0, 1)

        diff = np.abs(img_out - img_in)
        diff = np.mean(diff, axis=2) 

        vis_in = gamma_correction(img_in * 255, gamma=0.5)
        
        p_max = np.percentile(diff, 99.5)
        diff_norm = np.clip(diff / (p_max + 1e-8), 0, 1)
        
        processed_results.append((vis_in, diff_norm))

    print("🎨 正在拼接 3x2 网格大图...")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 14), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for i, (vis_in, diff_norm) in enumerate(processed_results):
        row = i // 2           
        col_base = (i % 2) * 2 

        ax_in = axes[row, col_base]
        ax_in.imshow(vis_in)
        ax_in.axis('off')
        
        ax_map = axes[row, col_base + 1]
        ax_map.imshow(diff_norm, cmap='inferno', vmin=0, vmax=1)
        ax_map.axis('off')
        
        if row == 0:
            ax_in.set_title("Input (Gamma Corrected)", fontsize=18, pad=12)
            ax_map.set_title("Compensation Map", fontsize=18, pad=12)

    plt.tight_layout()

    save_path = OUTPUT_DIR / "LACM_Visualization_Grid_3x2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n✅ 完美收工！组合大图已保存至: {save_path}")

if __name__ == '__main__':
    run()