import re
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from scipy.interpolate import make_interp_spline

def extract_losses_from_log(log_path):
    """从日志文件中提取各种损失值"""
    step_losses = {
        'train_loss': [],
        'task_loss': [],
        'kl_loss': [],
        'hidden_loss': [],
        'attn_loss': []
    }
    step_values = []
    
    # 用于提取每个epoch的平均测试损失
    epoch_test_losses = []
    epoch_step_values = []
    
    # 正则表达式模式
    step_pattern = re.compile(r'step: (\d+)/\d+.*train_loss:([\d\.]+), task_loss:([\d\.]+), kl_loss:([\d\.]+), hidden_loss:([\d\.]+), attn_loss:([\d\.]+)')
    test_pattern = re.compile(r'step: (\d+)/\d+.*average_test_loss:([\d\.]+)')
    
    with open(log_path, 'r') as f:
        for line in f:
            # 提取每100步的训练损失
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))
                step_values.append(step)
                step_losses['train_loss'].append(float(step_match.group(2)))
                step_losses['task_loss'].append(float(step_match.group(3)))
                step_losses['kl_loss'].append(0.5*float(step_match.group(4)))
                step_losses['hidden_loss'].append(float(step_match.group(5)))
                step_losses['attn_loss'].append(float(step_match.group(6)))
            
            # 提取每个epoch的测试损失
            test_match = test_pattern.search(line)
            if test_match:
                step = int(test_match.group(1))
                epoch_step_values.append(step)
                epoch_test_losses.append(float(test_match.group(2)))
    
    return step_values, step_losses, epoch_step_values, epoch_test_losses

def smooth_curve(x, y, smoothing_factor=300):
    """使用样条插值平滑曲线"""
    if len(x) < 4:  # 样条插值需要至少4个点
        return x, y
    
    # 创建更密集的x点以获得更平滑的曲线
    x_smooth = np.linspace(min(x), max(x), smoothing_factor)
    
    # 使用样条插值
    try:
        spline = make_interp_spline(x, y, k=3)  # k=3表示三次样条
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth
    except:
        # 如果样条插值失败，回退到简单的移动平均
        return x, moving_average(y, window=3)

def moving_average(data, window=3):
    """计算移动平均"""
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='same')

def plot_losses(step_values, step_losses, epoch_step_values, epoch_test_losses, output_path):
    """绘制损失曲线并保存"""
    plt.figure(figsize=(12, 8))
    
    # 绘制每100步的训练损失
    plt.plot(step_values, step_losses['train_loss'], label='Train Loss', color='blue')
    plt.plot(step_values, step_losses['task_loss'], label='1.0 * Task Loss', color='green')
    plt.plot(step_values, step_losses['kl_loss'], label='0.5 * KL Loss', color='red')
    plt.plot(step_values, step_losses['hidden_loss'], label='0.05 * Hidden Loss', color='purple')
    plt.plot(step_values, step_losses['attn_loss'], label='1.0 * Attention Loss', color='orange')
    
    # 平滑并绘制每个epoch的测试损失
    if len(epoch_step_values) > 1:
        x_smooth, y_smooth = smooth_curve(epoch_step_values, epoch_test_losses)
        plt.plot(x_smooth, y_smooth, label='Average Test Loss', color='teal', linestyle='-', linewidth=2.5)
    else:
        plt.plot(epoch_step_values, epoch_test_losses, label='Average Test Loss', color='teal', linestyle='-', linewidth=2.5)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training and Testing Losses over Steps', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整y轴以更好地显示小值
    plt.yscale('log')  # 使用对数刻度，可以更好地显示不同量级的损失
    
    # 添加一些额外的标注
    max_step = max(step_values)
    plt.xlim([0, max_step * 1.05])  # 给x轴留一些额外空间
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Loss plot saved to {output_path}")
    
    # 显示图像
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse training log and plot loss curves')
    parser.add_argument('--log_path', type=str, required=True, help='Path to the training log file')
    parser.add_argument('--output_path', type=str, default='loss_plot.png', help='Path to save the output plot')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file {args.log_path} does not exist!")
        return
    
    # 提取损失值
    step_values, step_losses, epoch_step_values, epoch_test_losses = extract_losses_from_log(args.log_path)
    
    # 如果没有提取到损失值
    if not step_values:
        print("No loss values were extracted from the log file.")
        return
    
    # 绘制并保存图像
    plot_losses(step_values, step_losses, epoch_step_values, epoch_test_losses, args.output_path)

if __name__ == "__main__":
    main()

    # python visualize_loss.py --log_path results/Finetune_TS_T0.40_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_att_loss_hidstate_loss/train.log --output_path plots/loss_plot_1_1.png