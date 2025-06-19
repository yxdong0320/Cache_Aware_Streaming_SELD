import re
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def parse_log_file(log_file_path):
    """
    解析日志文件，提取训练和测试损失值
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        dict: 包含所有损失值的字典
    """
    data = {
        'steps': [],
        'epochs': [],
        'lr': [],
        'train_loss': [],
        'train_distill_loss': [],
        'train_stream_loss': [],
        'train_non_stream_loss': [],
        'train_non_stream_apc_loss': [],
        'test_steps': [],
        'test_epochs': [],
        'average_test_nonstream_loss': [],
        'average_test_stream_loss': []
    }
    
    # 训练损失的正则表达式
    train_pattern = re.compile(
        r'epoch: (\d+), step: (\d+)/\d+, lr:([\d.]+), '
        r'train_loss:([\d.]+), train_distill_loss:([\d.]+), '
        r'train_stream_loss:([\d.]+), train_non_stream_loss:([\d.]+), '
        r'train_non_stream_apc_loss:([\d.]+)'
    )
    
    # 测试损失的正则表达式
    test_pattern = re.compile(
        r'epoch: (\d+), step: (\d+)/\d+, train_time:[\d.]+, test_time:[\d.]+, '
        r'average_train_loss:[\d.]+, average_test_nonstream_loss:([\d.]+), '
        r'average_test_stream_loss:([\d.]+)'
    )
    
    print(f"开始解析日志文件: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 匹配训练损失
            train_match = train_pattern.search(line)
            if train_match:
                epoch, step, lr, train_loss, distill_loss, stream_loss, non_stream_loss, apc_loss = train_match.groups()
                
                data['epochs'].append(int(epoch))
                data['steps'].append(int(step))
                data['lr'].append(float(lr))
                data['train_loss'].append(float(train_loss))
                data['train_distill_loss'].append(float(distill_loss))
                data['train_stream_loss'].append(float(stream_loss))
                data['train_non_stream_loss'].append(float(non_stream_loss))
                data['train_non_stream_apc_loss'].append(float(apc_loss))
                continue
            
            # 匹配测试损失
            test_match = test_pattern.search(line)
            if test_match:
                epoch, step, test_nonstream_loss, test_stream_loss = test_match.groups()
                
                data['test_epochs'].append(int(epoch))
                data['test_steps'].append(int(step))
                data['average_test_nonstream_loss'].append(float(test_nonstream_loss))
                data['average_test_stream_loss'].append(float(test_stream_loss))
    
    print(f"解析完成:")
    print(f"  - 训练损失记录: {len(data['train_loss'])} 条")
    print(f"  - 测试损失记录: {len(data['average_test_nonstream_loss'])} 条")
    
    return data

def plot_loss_curves(data, save_dir, experiment_name="experiment"):
    """
    绘制损失曲线并保存
    
    Args:
        data: parse_log_file返回的数据字典
        save_dir: 保存图片的目录
        experiment_name: 实验名称，用于文件命名
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training and Testing Loss Curves - {experiment_name}', fontsize=16)
    
    # 1. 总体训练损失
    ax1 = axes[0, 0]
    ax1.plot(data['steps'], data['train_loss'], 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    if data['average_test_nonstream_loss']:
        ax1.plot(data['test_steps'], data['average_test_nonstream_loss'], 'r-', 
                linewidth=2, marker='o', markersize=4, label='Test NonStream Loss')
        ax1.plot(data['test_steps'], data['average_test_stream_loss'], 'g-', 
                linewidth=2, marker='s', markersize=4, label='Test Stream Loss')
    ax1.set_title('Overall Training and Testing Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练损失组件
    ax2 = axes[0, 1]
    ax2.plot(data['steps'], data['train_distill_loss'], 'r-', linewidth=1, alpha=0.8, label='Distill Loss')
    ax2.plot(data['steps'], data['train_stream_loss'], 'g-', linewidth=1, alpha=0.8, label='Stream Loss')
    ax2.plot(data['steps'], data['train_non_stream_loss'], 'b-', linewidth=1, alpha=0.8, label='NonStream Loss')
    ax2.set_title('Training Loss Components')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. APC损失
    ax3 = axes[0, 2]
    ax3.plot(data['steps'], data['train_non_stream_apc_loss'], 'purple', linewidth=1, alpha=0.8, label='NonStream APC Loss')
    ax3.set_title('APC Loss')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习率
    ax4 = axes[1, 0]
    ax4.plot(data['steps'], data['lr'], 'orange', linewidth=1, alpha=0.8)
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True, alpha=0.3)
    
    # 5. 测试损失对比
    ax5 = axes[1, 1]
    if data['average_test_nonstream_loss']:
        ax5.plot(data['test_steps'], data['average_test_nonstream_loss'], 'r-', 
                linewidth=2, marker='o', markersize=6, label='NonStream Test Loss')
        ax5.plot(data['test_steps'], data['average_test_stream_loss'], 'g-', 
                linewidth=2, marker='s', markersize=6, label='Stream Test Loss')
        ax5.set_title('Test Loss Comparison')
        ax5.set_xlabel('Steps')
        ax5.set_ylabel('Loss')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Test Data Available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Test Loss Comparison')
    
    # 6. 损失趋势（移动平均）
    ax6 = axes[1, 2]
    if len(data['train_loss']) > 50:
        # 计算移动平均
        window = min(50, len(data['train_loss']) // 10)
        train_loss_ma = np.convolve(data['train_loss'], np.ones(window)/window, mode='valid')
        steps_ma = data['steps'][window-1:]
        
        ax6.plot(steps_ma, train_loss_ma, 'b-', linewidth=2, label=f'Train Loss (MA-{window})')
        
        if data['average_test_nonstream_loss']:
            ax6.plot(data['test_steps'], data['average_test_nonstream_loss'], 'r-', 
                    linewidth=2, marker='o', markersize=4, label='Test NonStream Loss')
    else:
        ax6.plot(data['steps'], data['train_loss'], 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    
    ax6.set_title('Loss Trends (Smoothed)')
    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_loss_curves_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存到: {filepath}")
    
    # 也保存一个不带时间戳的版本，方便覆盖更新
    filename_simple = f"{experiment_name}_loss_curves.png"
    filepath_simple = os.path.join(save_dir, filename_simple)
    plt.savefig(filepath_simple, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return filepath

def save_loss_data(data, save_dir, experiment_name="experiment"):
    """
    保存损失数据到CSV文件
    """
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存训练数据
    train_df = pd.DataFrame({
        'epoch': data['epochs'],
        'step': data['steps'],
        'lr': data['lr'],
        'train_loss': data['train_loss'],
        'train_distill_loss': data['train_distill_loss'],
        'train_stream_loss': data['train_stream_loss'],
        'train_non_stream_loss': data['train_non_stream_loss'],
        'train_non_stream_apc_loss': data['train_non_stream_apc_loss']
    })
    
    train_csv_path = os.path.join(save_dir, f"{experiment_name}_train_losses.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"训练损失数据已保存到: {train_csv_path}")
    
    # 保存测试数据（如果有）
    if data['average_test_nonstream_loss']:
        test_df = pd.DataFrame({
            'epoch': data['test_epochs'],
            'step': data['test_steps'],
            'average_test_nonstream_loss': data['average_test_nonstream_loss'],
            'average_test_stream_loss': data['average_test_stream_loss']
        })
        
        test_csv_path = os.path.join(save_dir, f"{experiment_name}_test_losses.csv")
        test_df.to_csv(test_csv_path, index=False)
        print(f"测试损失数据已保存到: {test_csv_path}")
    
    return train_csv_path

def plot_specific_losses(data, save_dir, experiment_name="experiment"):
    """
    绘制特定的损失对比图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 流式 vs 非流式损失对比
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data['steps'], data['train_stream_loss'], 'g-', linewidth=1, alpha=0.8, label='Train Stream Loss')
    plt.plot(data['steps'], data['train_non_stream_loss'], 'b-', linewidth=1, alpha=0.8, label='Train NonStream Loss')
    if data['average_test_stream_loss']:
        plt.plot(data['test_steps'], data['average_test_stream_loss'], 'g-', 
                linewidth=2, marker='s', markersize=4, label='Test Stream Loss')
        plt.plot(data['test_steps'], data['average_test_nonstream_loss'], 'b-', 
                linewidth=2, marker='o', markersize=4, label='Test NonStream Loss')
    plt.title('Stream vs NonStream Loss Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 蒸馏损失
    plt.subplot(2, 2, 2)
    plt.plot(data['steps'], data['train_distill_loss'], 'r-', linewidth=1, alpha=0.8)
    plt.title('Knowledge Distillation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 3. APC损失
    plt.subplot(2, 2, 3)
    plt.plot(data['steps'], data['train_non_stream_apc_loss'], 'purple', linewidth=1, alpha=0.8)
    plt.title('APC Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 4. 总损失
    plt.subplot(2, 2, 4)
    plt.plot(data['steps'], data['train_loss'], 'k-', linewidth=1, alpha=0.8, label='Total Train Loss')
    if data['average_test_nonstream_loss']:
        plt.plot(data['test_steps'], data['average_test_nonstream_loss'], 'r-', 
                linewidth=2, marker='o', markersize=4, label='Test NonStream Loss')
    plt.title('Total Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    filename = f"{experiment_name}_loss_comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"损失对比图已保存到: {filepath}")
    
    plt.show()
    
    return filepath

def main(log_file_path, save_dir, experiment_name=None):
    """
    主函数：解析日志并生成所有图表
    
    Args:
        log_file_path: 日志文件路径
        save_dir: 保存图片和数据的目录
        experiment_name: 实验名称，如果为None则从日志文件名推断
    """
    if experiment_name is None:
        experiment_name = os.path.splitext(os.path.basename(log_file_path))[0]
    
    print(f"开始处理实验: {experiment_name}")
    
    # 解析日志文件
    data = parse_log_file(log_file_path)
    
    if not data['train_loss']:
        print("错误: 没有找到训练损失数据，请检查日志文件格式")
        return
    
    # 绘制主要损失曲线
    plot_loss_curves(data, save_dir, experiment_name)
    
    # 绘制特定损失对比图
    plot_specific_losses(data, save_dir, experiment_name)
    
    # 保存数据到CSV
    save_loss_data(data, save_dir, experiment_name)
    
    # 打印统计信息
    print(f"\n实验统计信息:")
    print(f"  训练步数: {len(data['train_loss'])}")
    print(f"  最终训练损失: {data['train_loss'][-1]:.4f}")
    if data['average_test_nonstream_loss']:
        print(f"  最终测试损失(NonStream): {data['average_test_nonstream_loss'][-1]:.4f}")
        print(f"  最终测试损失(Stream): {data['average_test_stream_loss'][-1]:.4f}")
    print(f"  结果保存目录: {save_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='解析训练日志并绘制损失曲线')
    parser.add_argument('log_file', type=str, help='日志文件路径')
    parser.add_argument('-o', '--output_dir', type=str, default='./loss_plots', 
                       help='输出目录 (默认: ./loss_plots)')
    parser.add_argument('-n', '--name', type=str, default=None,
                       help='实验名称 (默认: 从日志文件名推断)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"错误: 日志文件不存在: {args.log_file}")
        exit(1)
    
    main(args.log_file, args.output_dir, args.name)

# python visualize_loss_curve_dualmode.py results/Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC/train.log -o plots -n "Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC"