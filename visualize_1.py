import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 假设您有以下数据
configs = ['Offline_P8F2', 'Offline_P4F1', 'Offline_P2F1', 'Offline_P0F1', 'Offline_P0.9F0.1',
            'L10_EM256_RS49','L8_EM256_RS49', 'L4_EM256_RS49',
           'L8_EM256_RS24','L8_EM256_RS9', 'L8_EM256_RS4',
           'L8_EM128_RS49', 'L8_EM64_RS49',
           'L8_EM64_RS24', 'L8_EM64_RS9', 'L8_EM64_RS4',
           ]
performance = [0.41, 0.41, 0.43, 0.47, 0.49,
                0.43, 0.43, 0.45,
               0.44, 0.47, 0.47,
               0.44, 0.45,
               0.47, 0.48, 0.50,
               ]  # 例如F1分数
latency = [2.141, 1.069, 1.049, 1.042, 0.136,
            1.048, 1.043, 1.035,
           0.538, 0.228, 0.129,
           1.043, 1.043,
           0.534, 0.229, 0.127]  # 毫秒

# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 6))

# 设置柱状图
x = np.arange(len(configs))
width = 0.6
bars = ax1.bar(x, performance, width, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Model Configuration', fontsize=12)
ax1.set_ylabel('Performance (SELD Score)', fontsize=12)
ax1.set_ylim(min(performance)* 0.9, max(performance) * 1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=45, ha='right')

# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# 创建第二个Y轴用于时延
ax2 = ax1.twinx()
# 使用更柔和的颜色 - 改用暗红色/砖红色
soft_red = '#C25B56'  # 砖红色
ax2.plot(x, latency, marker='o', color=soft_red, linewidth=2, markersize=8)
ax2.set_ylabel('Latency (s)', color=soft_red, fontsize=12)
ax2.tick_params(axis='y', labelcolor=soft_red)
ax2.set_ylim(0, max(latency) * 1.25)  # 增加上限以便有更多空间放置标签

# 在折线图上添加数值标签 - 调整垂直位置使其更贴近点
offset = max(latency) * 0.05  # 动态计算偏移量
for i, v in enumerate(latency):
    ax2.text(i, v + offset, f'{v:.3f}s', color=soft_red, ha='center', va='bottom', fontsize=9)

plt.title('Performance vs. Latency for Different Model Configurations', fontsize=14)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例 - 调整位置避免遮挡
ax1.legend(['Performance'], loc='upper left')
# 将延迟图例移到图的右上角但稍低一点以避免遮挡
ax2.legend(['Latency'], loc='upper right', bbox_to_anchor=(1.0, 0.95))

plt.savefig('plots/performance_latency_comparison_2.png', dpi=300, bbox_inches='tight')
# plt.show()