import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random
import math
from types import MethodType

from model_snn_cnn import generate_model_snn

# ================= 配置区域 =================
CHECKPOINT_PATH = '/root/Single-eye-Emotion-Recognition/single-eye-emotion/results/resnet_cmm_test/save_110.pth' # 务必换成你的最高轮数权重！
FRAME_ROOT = Path('./SEE/frame')
EVENT_ROOT = Path('./SEE/event_30')
OUTPUT_DIR = Path('visible_low_light_results')

SAMPLE_VIDEOS_PER_CLASS = 100 
# ===========================================

EVENT_MEAN = [0.504413, 0.504413, 0.504413]
EVENT_STD = [0.06928615, 0.06928615, 0.06928615]
FRAME_MEAN = [0.22616537, 0.22616537, 0.22616537]
FRAME_STD = [0.118931554, 0.118931554, 0.118931554]

EMOTIONS = ['happiness', 'sadness', 'angry', 'disgust', 'surprise', 'fear', 'neutral']
CONDITIONS = ['normal', 'Low', 'Overexposed', 'HDR']
SUB_TITLES = ['(a) Normal Conditions', '(b) Low Light Conditions', 
              '(c) Overexposed Conditions', '(d) HDR Conditions']

def get_hooked_forward():
    def hooked_asfm_forward(self, fe, ff):
        B, C, H, W = fe.shape
        mid_C = C // self.reduction
        fe_red = self.conv_reduce(fe)
        ff_red = self.conv_reduce(ff)
        fe_pool = F.adaptive_avg_pool2d(fe_red, 1) + F.adaptive_max_pool2d(fe_red, 1)
        ff_pool = F.adaptive_avg_pool2d(ff_red, 1) + F.adaptive_max_pool2d(ff_red, 1)
        merged_pool = torch.cat([fe_pool, ff_pool], dim=1)
        beta = self.weight_net(merged_pool).squeeze(-1).squeeze(-1)
        
        D = torch.median(beta, dim=1, keepdim=True).values
        beta_tilde = torch.where(beta >= D, beta * self.alpha_amp, beta * self.alpha_low)
        w_e_mid = beta_tilde[:, :mid_C]
        w_f_mid = beta_tilde[:, mid_C:]
        
        self.captured_w_e = w_e_mid.mean(dim=1).detach().cpu().numpy()
        self.captured_w_f = w_f_mid.mean(dim=1).detach().cpu().numpy()
        
        w_e = w_e_mid.unsqueeze(-1).unsqueeze(-1).repeat_interleave(self.reduction, dim=1)
        w_f = w_f_mid.unsqueeze(-1).unsqueeze(-1).repeat_interleave(self.reduction, dim=1)
        fused = torch.cat([fe * w_e, ff * w_f], dim=1)
        return self.restore_conv(fused)
    return hooked_asfm_forward

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
        print("✅ 模型权重加载成功！\n")
    
    model.eval()
    model.asfm.forward = MethodType(get_hooked_forward(), model.asfm)

    transform_frame = transforms.Compose([transforms.Resize((90, 90)), transforms.ToTensor(), transforms.Normalize(FRAME_MEAN, FRAME_STD)])
    transform_event = transforms.Compose([transforms.Resize((90, 90)), transforms.ToTensor(), transforms.Normalize(EVENT_MEAN, EVENT_STD)])

    # 优化矢量图导出时的字体渲染，确保文本可编辑/不失真
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    for idx, condition in enumerate(CONDITIONS):
        print(f"==================================================")
        print(f"🌟 正在处理光照场景: 【 {condition.upper()} 】 ({idx+1}/4)")
        
        results = {}
        random.seed(42)

        for emotion in EMOTIONS:
            results[emotion] = {'w_e': [], 'w_f': []}
            emotion_dir = FRAME_ROOT / emotion
            if not emotion_dir.exists(): continue
                
            video_folders = list(emotion_dir.glob('*'))
            random.shuffle(video_folders)
            
            video_count = 0
            for v_folder in video_folders:
                if condition.lower() not in v_folder.name.lower():
                    continue

                if video_count >= SAMPLE_VIDEOS_PER_CLASS: break
                    
                frame_imgs = sorted(list(v_folder.glob('*.jpg')))
                if len(frame_imgs) < 3: continue
                    
                target_indices = [len(frame_imgs)//4, len(frame_imgs)//2, len(frame_imgs)*3//4]
                
                for i_idx in target_indices:
                    target_img_name = frame_imgs[i_idx].name
                    frame_path = v_folder / target_img_name
                    event_path = EVENT_ROOT / emotion / v_folder.name / target_img_name
                    
                    if frame_path.exists() and event_path.exists():
                        f_img = Image.open(frame_path).convert('RGB')
                        e_img = Image.open(event_path).convert('RGB')
                        
                        with torch.no_grad():
                            _ = model(transform_frame(f_img).unsqueeze(0).to(device), 
                                      transform_event(e_img).unsqueeze(0).to(device))
                        
                        results[emotion]['w_e'].append(float(model.asfm.captured_w_e[0]))
                        results[emotion]['w_f'].append(float(model.asfm.captured_w_f[0]))
                
                video_count += 1
                    
            print(f"  - [{emotion.capitalize()}] 完成")

        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        valid_emotions = [e for e in EMOTIONS if len(results[e]['w_e']) > 0]
        if not valid_emotions:
            continue
            
        labels = [e.capitalize() for e in valid_emotions]
        w_f_raw = np.array([np.mean(results[e]['w_f']) for e in valid_emotions])
        w_e_raw = np.array([np.mean(results[e]['w_e']) for e in valid_emotions])
        
        total_weight = w_f_raw + w_e_raw
        w_f_means = (w_f_raw / total_weight) * 100
        w_e_means = (w_e_raw / total_weight) * 100
        
        x = np.arange(len(labels))
        width = 0.38 
        
        rects1 = ax.bar(x - width/2, w_f_means, width, label='RGB Contribution', color='#4C72B0', edgecolor='black', linewidth=0.5)
        rects2 = ax.bar(x + width/2, w_e_means, width, label='Event Contribution', color='#DD8452', edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Relative Contribution (%)', fontsize=13, fontweight='bold')
        ax.set_title(SUB_TITLES[idx], fontsize=15, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        
        if idx == 1:
            ax.legend(fontsize=11, loc='upper right')
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        max_val = max(np.max(w_f_means), np.max(w_e_means))
        ax.set_ylim(20, math.ceil(max_val + 10)) 
        
        ax.bar_label(rects1, padding=4, fmt='%.1f%%', fontsize=8.5)
        ax.bar_label(rects2, padding=4, fmt='%.1f%%', fontsize=8.5)

    # ================= 核心修改：保存为高清矢量图格式 =================
    
    # 1. 保存为 PDF (期刊排版、LaTeX 论文最常用)
    save_path_pdf = OUTPUT_DIR / 'ASFM_Relative_Weights_2x2_Grid.pdf'
    plt.savefig(save_path_pdf, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    # 2. 保存为 SVG (Microsoft Word 支持最好、最方便插入的格式)
    save_path_svg = OUTPUT_DIR / 'ASFM_Relative_Weights_2x2_Grid.svg'
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    
    # 3. 依然保留一张 PNG 方便你平时快速预览
    save_path_png = OUTPUT_DIR / 'ASFM_Relative_Weights_2x2_Grid.png'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    print(f"\n✅ 完美排版！高清矢量图已保存！\n 📄 PDF路径: {save_path_pdf}\n 🎨 SVG路径: {save_path_svg}")

if __name__ == '__main__':
    run()