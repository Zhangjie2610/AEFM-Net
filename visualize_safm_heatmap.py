import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import cv2
import random

# 导入你的模型
from model_snn_cnn import generate_model_snn

# ================= 配置区域 =================
CHECKPOINT_PATH = '/root/Single-eye-Emotion-Recognition/single-eye-emotion/results/resnet_cmm_test/save_130.pth'
FRAME_ROOT = Path('./SEE/frame')
EVENT_ROOT = Path('./SEE/event_30')
OUTPUT_DIR = Path('visible_low_light_results')

TARGET_EMOTIONS = ['surprise', 'happiness']
# ===========================================

FRAME_MEAN = [0.22616537, 0.22616537, 0.22616537]
FRAME_STD = [0.118931554, 0.118931554, 0.118931554]
EVENT_MEAN = [0.504413, 0.504413, 0.504413]
EVENT_STD = [0.06928615, 0.06928615, 0.06928615]

activations = {}

def get_activation_hook(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def generate_heatmap(feature_tensor, original_img_shape=(90, 90)):
    # 🎯 核心修复 1：用通道最大值 (max) 替代平均值 (mean)，精准定位最强激活的肌肉单元！
    heatmap = torch.max(feature_tensor, dim=1)[0].squeeze().cpu().numpy()
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    # 🎯 核心修复 2：加入指数级对比度惩罚 (三次方)！
    # 完美模拟你论文中 TopK 和稀疏阈值的物理效果，将中低响应的背景“杂音”强行压平
    heatmap = heatmap ** 3 
    
    heatmap = cv2.resize(heatmap, original_img_shape)
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0) # 依然保留高级感的平滑
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap, heatmap_color

def superimpose_heatmap(img_pil, heatmap_color, alpha=0.55): # 将 alpha 统一回调到舒适比例
    img_np = np.array(img_pil.resize((90, 90)), dtype=np.float32)
    img_np = np.clip(img_np * 1.2, 0, 255) # 轻微提亮原图
    
    superimposed_img = heatmap_color * alpha + img_np * (1.0 - alpha)
    return np.uint8(np.clip(superimposed_img, 0, 255))

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 初始化模型并加载权重...")
    model = generate_model_snn(use_hdr=True).to(device)
    
    if Path(CHECKPOINT_PATH).exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("✅ 模型加载成功！\n")
        
    model.eval()
    
    model.resnet_frame.layer2.register_forward_hook(get_activation_hook('shallow_noisy'))
    model.mfcm_frame.register_forward_hook(get_activation_hook('safm_purified'))

    transform_frame = transforms.Compose([transforms.Resize((90, 90)), transforms.ToTensor(), transforms.Normalize(FRAME_MEAN, FRAME_STD)])
    transform_event = transforms.Compose([transforms.Resize((90, 90)), transforms.ToTensor(), transforms.Normalize(EVENT_MEAN, EVENT_STD)])

    results_to_plot = []
    random.seed(42)

    print("📊 正在提取特征...")
    for emotion in TARGET_EMOTIONS:
        emotion_dir = FRAME_ROOT / emotion
        if not emotion_dir.exists(): continue
            
        video_folders = list(emotion_dir.glob('*'))
        random.shuffle(video_folders)
        
        for v_folder in video_folders:
            frame_imgs = sorted(list(v_folder.glob('*.jpg')))
            if len(frame_imgs) < 3: continue
                
            target_img_name = frame_imgs[len(frame_imgs)//2].name
            frame_path = v_folder / target_img_name
            event_path = EVENT_ROOT / emotion / v_folder.name / target_img_name
            
            if frame_path.exists() and event_path.exists():
                raw_rgb_pil = Image.open(frame_path).convert('RGB')
                raw_evt_pil = Image.open(event_path).convert('RGB')
                
                f_t = transform_frame(raw_rgb_pil).unsqueeze(0).to(device)
                e_t = transform_event(raw_evt_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    _ = model(f_t.unsqueeze(2), e_t.unsqueeze(2))
                
                if 'shallow_noisy' in activations and 'safm_purified' in activations:
                    _, color_shallow = generate_heatmap(activations['shallow_noisy'])
                    _, color_safm = generate_heatmap(activations['safm_purified'])
                    
                    overlay_shallow = superimpose_heatmap(raw_rgb_pil, color_shallow, alpha=0.55)
                    overlay_safm = superimpose_heatmap(raw_rgb_pil, color_safm, alpha=0.55)
                    
                    results_to_plot.append({
                        'emotion': emotion.capitalize(),
                        'rgb': np.uint8(np.clip(np.array(raw_rgb_pil.resize((90, 90))) * 1.2, 0, 255)),
                        'shallow': overlay_shallow,
                        'safm': overlay_safm
                    })
                    break 

    print("🎨 正在生成顶会级无缝排版图...")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    col_titles = ['(a) Input RGB Image', '(b) Shallow Features (w/o SAFM)', '(c) Purified Features (w/ SAFM)']

    for row, data in enumerate(results_to_plot):
        # 第一列：原图
        ax = axes[row, 0]
        ax.imshow(data['rgb'])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[0], fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel(data['emotion'], fontsize=14, fontweight='bold', labelpad=10)

        # 第二列：浅层特征
        ax = axes[row, 1]
        ax.imshow(data['shallow'])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[1], fontsize=13, fontweight='bold', pad=12)

        # 第三列：纯化特征
        ax = axes[row, 2]
        ax.imshow(data['safm'])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[2], fontsize=13, fontweight='bold', pad=12)

    save_path = OUTPUT_DIR / 'SAFM_Noise_Suppression_Heatmap_Pro.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ 大功告成！完美级排版图已保存至: {save_path}\n")

if __name__ == '__main__':
    run()