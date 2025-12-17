import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def visualize_results(original_images, anomaly_maps, scores, threshold=5.0, save_dir=None):
    """
    可视化原始图像、异常热力图和检测结果
    """
    batch_size = len(original_images)
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # 原始图像
        img = original_images[i].permute(1, 2, 0).cpu().numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image\nScore: {scores[i]:.4f}')
        axes[i, 0].axis('off')
        
        # 异常热力图
        anomaly_map = anomaly_maps[i]
        im = axes[i, 1].imshow(anomaly_map, cmap='jet', aspect='auto')
        axes[i, 1].set_title(f'Anomaly Heatmap\nMax: {anomaly_map.max():.4f}')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 检测结果（叠加热力图）
        axes[i, 2].imshow(img)
        # 叠加半透明的热力图
        heatmap_overlay = anomaly_map / (anomaly_map.max() + 1e-8)  # 归一化
        axes[i, 2].imshow(heatmap_overlay, cmap='jet', alpha=0.5, extent=axes[i, 2].get_xlim() + axes[i, 2].get_ylim())
        
        if scores[i] > threshold:
            # 绘制红色边界框表示异常
            h, w = img.shape[:2]
            rect = plt.Rectangle((0, 0), w-1, h-1, fill=False, edgecolor='red', linewidth=3)
            axes[i, 2].add_patch(rect)
            axes[i, 2].set_title(f'ANOMALOUS (Score: {scores[i]:.2f})', color='red', fontsize=12)
        else:
            axes[i, 2].set_title(f'Normal (Score: {scores[i]:.2f})', color='green', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 使用时间戳或随机数避免文件名冲突
        import time
        timestamp = int(time.time() * 1000) % 10000
        save_path = os.path.join(save_dir, f'heatmap_result_{timestamp}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        print(f"   🔥 热力图已保存: {save_path}")
    
    plt.close()  # 关闭图形释放内存

def plot_anomaly_map(anomaly_map, original_image=None, ax=None, title="Anomaly Heatmap"):
    """
    单独绘制异常热力图
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax.imshow(anomaly_map, cmap='jet')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    return ax