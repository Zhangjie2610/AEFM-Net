import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm

from opts import parse_opts
from model_snn_cnn import generate_model_snn
from main import get_inference_utils

# ================= 配置区域 =================
CHECKPOINT_PATH = './results/resnet_cmm_test/save_110.pth'
OUTPUT_FILE = 'tsne_emotion_clusters.svg'

# 英文标签
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
# ===========================================

def run_tsne_svg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 解析参数
    opt = parse_opts()
    opt.device = device
    opt.inference = True
    opt.inference_subset = 'test'
    opt.inference_no_average = True 

    # 2. 加载模型
    print("📥 Loading model...")
    model = generate_model_snn(use_hdr=True).to(device)
    if Path(CHECKPOINT_PATH).exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. 注册 Hook 提取特征
    features_list = []
    def hook_fn(module, input, output):
        features_list.append(output.flatten(1).detach().cpu().numpy())
    hook_handle = model.avgpool.register_forward_hook(hook_fn)

    # 4. 获取数据集
    print("🗂️ Loading test dataset...")
    inference_loader, _ = get_inference_utils(opt)
    labels_list = []

    # 5. 推理并收集特征
    print("🚀 Extracting features...")
    with torch.no_grad():
        for i, (event_inputs, frame_inputs, targets) in enumerate(tqdm(inference_loader)):
            video_ids, segments = zip(*targets)
            _ = model(frame_inputs.to(device), event_inputs.to(device))
            
            for vid in video_ids:
                # 提取情绪标签并首字母大写
                emotion_label = vid.split('_')[6].capitalize()
                labels_list.append(emotion_label)

    hook_handle.remove()
    features_all = np.concatenate(features_list, axis=0)
    labels_all = np.array(labels_list)

    # 6. t-SNE 降维
    print("🌌 Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_all)

    # 7. 绘制学术风 SVG 散点图
    print("🎨 Plotting standalone SVG...")
    plt.figure(figsize=(8, 7.5)) 
    
    # 设置统一的 seaborn 调色板
    palette = sns.color_palette("tab10", len(EMOTION_LABELS))
    
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=labels_all,
        hue_order=EMOTION_LABELS, 
        palette=palette,
        s=60,             
        alpha=0.85,       
        edgecolor='w',    
        linewidth=0.5
    )

    # 坐标轴标签 (纯英文)
    plt.xlabel("t-SNE Dimension 1", fontsize=15, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=15, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=13)

    # ================= 核心修改：保留图注，但去掉黑框 =================
    plt.legend(
        title="Emotions", 
        title_fontsize=14, 
        fontsize=13, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.12), 
        ncol=4,                      
        frameon=False,               # <---- 这里设置为 False，彻底去掉黑色的边框！
        shadow=False
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 Done! High-quality SVG saved as '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    run_tsne_svg()