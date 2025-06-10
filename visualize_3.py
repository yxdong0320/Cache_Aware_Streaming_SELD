import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data = {
    'Config': [ 'Offline_P2F1', 'Offline_P0F1',
            'L10_EM256_RS49','L8_EM256_RS49', 'L4_EM256_RS49',
            'L8_EM128_RS49', 'L8_EM64_RS49',
    ],
    'Params': [11.42, 11.42,
               12.27, 9.90, 5.14,
               3.13, 1.15,
               ],  # Millions
    'Performance': [0.43, 0.47,
                    0.43, 0.43, 0.45,
                    0.44, 0.45,
                    ],  # SELD score
    'MACs(G/s)': [2.40, 0.80,
                  0.88, 0.70, 0.36,
                  0.27, 0.12,
                    ],  # GMACs
    'Chunksize': [      
                        50, 50,
                        50, 50, 50,
                        50, 50,
    ],
}
df = pd.DataFrame(data)

# 设置图表样式
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# 创建散点图 - 使用单一颜色
scatter = plt.scatter(df['Params'], df['Performance'],
            s=df['MACs(G/s)']*100, # 调整点大小乘数以便更好地可视化
            color='#1f77b4',  # 使用单一颜色
            marker='o', # 所有点都使用圆形
            alpha=0.7, # 透明度
            edgecolors='black')

# 添加标签和图例
plt.xlabel('Number of Parameters (Millions)', fontsize=12)
plt.ylabel('Performance (SELD Score)', fontsize=12)
plt.title('Model Complexity vs. Performance Trade-off', fontsize=14)

# 添加大小图例（显示MACs）
sizes = [0.1, 0.5, 1.5, 4.0]
size_labels = ['0.1 GMACs', '0.5 GMACs', '1.5 GMACs', '4.0 GMACs']
for size, label in zip(sizes, size_labels):
    plt.scatter([], [], s=size*100, c='#1f77b4', alpha=0.7, edgecolors='black', label=label)

plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='MACs(G/s)')

# 为每个点添加标签，使用不同的偏移方向
# 定义一组不同的偏移方向
offsets = [
    (10, 10),    # 右上
    (-10, 10),   # 左上
    (10, -10),   # 右下
    (-10, -10),  # 左下
    (15, 0),     # 右
    (-15, 0),    # 左
    (0, 15),     # 上
    (0, -15)     # 下
]

# 为每个点添加标签，使用不同的偏移方向
for i, row in df.iterrows():
    # 选择一个偏移方向 (循环使用offsets列表)
    offset = offsets[i % len(offsets)]
    
    # 将标签从点偏移
    plt.annotate(row['Config'],
                 xy=(row['Params'], row['Performance']),
                 xytext=offset,
                 textcoords='offset points',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

# 如果需要更宽的尺度范围，可以调整坐标轴范围
plt.xlim(min(df['Params']) * 0.9, max(df['Params']) * 1.1)
plt.ylim(min(df['Performance']) * 0.95, max(df['Performance']) * 1.05)

plt.tight_layout()
plt.savefig('plots/model_complexity_tradeoff_2.png', dpi=300, bbox_inches='tight')

plt.show()