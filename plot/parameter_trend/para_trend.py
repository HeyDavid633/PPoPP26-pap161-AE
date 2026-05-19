import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==========================================
# 1. 字体配置
# ==========================================
simsun_path = '/System/Library/Fonts/Supplemental/Songti.ttc'

if os.path.exists(simsun_path):
    zh_font = font_manager.FontProperties(fname=simsun_path)
    print(f"成功加载字体：{simsun_path}")
else:
    print(f"警告：未找到字体 {simsun_path}，尝试使用系统默认中文字体。")
    zh_font = font_manager.FontProperties(family='STHeiti')

# Times New Roman 字体配置
times_font = font_manager.FontProperties(family='Times New Roman')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 准备数据
# ==========================================
data = [
    {"name": "AlexNet", "date": "2012-09-01", "flops": 1.1e18, "era": "small"},
    {"name": "VGG", "date": "2014-09-01", "flops": 1.5e19, "era": "small"},
    {"name": "ResNet-152", "date": "2015-12-01", "flops": 1.1e20, "era": "small"},
    {"name": "Transformer", "date": "2017-06-01", "flops": 1.0e20, "era": "small"}, 
    {"name": "BERT-Large", "date": "2018-10-01", "flops": 1.0e21, "era": "small"},
    {"name": "GPT-2", "date": "2019-02-01", "flops": 2.5e21, "era": "small"},
    
    {"name": "GPT-3", "date": "2020-06-01", "flops": 3.14e23, "era": "large"},
    {"name": "Switch Transformer", "date": "2021-01-01", "flops": 1.0e23, "era": "large"}, 
    {"name": "PaLM", "date": "2022-04-01", "flops": 1.2e24, "era": "large"},
    {"name": "Llama 2 (70B)", "date": "2023-07-01", "flops": 2.0e23, "era": "large"},
    {"name": "GPT-4", "date": "2023-03-01", "flops": 2.15e25, "era": "large"},
    {"name": "Grok-1", "date": "2023-11-01", "flops": 2.0e24, "era": "large"},
    {"name": "Gemini Ultra", "date": "2023-12-01", "flops": 1.0e25, "era": "large"},
    {"name": "Claude 3", "date": "2024-03-01", "flops": 4e25, "era": "large"},
    {"name": "DeepSeek-V2", "date": "2024-05-01", "flops": 2.5e24, "era": "large"},
    {"name": "Qwen-2 (72B)", "date": "2024-06-01", "flops": 5.0e24, "era": "large"},
    
    {"name": "DeepSeek-R1", "date": "2025-01-01", "flops": 1.2e25, "era": "large"},
    {"name": "Grok-3", "date": "2025-02-01", "flops": 5.0e25, "era": "large"},
    {"name": "GPT-5", "date": "2025-06-01", "flops": 9.0e25, "era": "large"},
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['flops'] = df['flops'].astype(float)

# ==========================================
# 3. 创建画布 (适当扩大以缓解拥挤)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# ==========================================
# 4. 设置背景色 (分界线改为 2020 年)
# ==========================================
ax.axvspan(datetime(2012, 1, 1), datetime(2020, 1, 1), color='#e3f2fd', alpha=0.6)
ax.axvspan(datetime(2020, 1, 1), datetime(2026, 12, 31), color='#ffebee', alpha=0.6)

# 添加背景文字 (横向方式)
ax.text(datetime(2015, 6, 1), 2e26, "深度学习时代 (Deep Learning Era)", 
        color='#1976d2', fontsize=15, fontproperties=zh_font, fontweight='bold', ha='center')
ax.text(datetime(2022, 6, 1), 2e26, "大模型时代 (Large-Scale Era)", 
        color='#d32f2f', fontsize=15, fontproperties=zh_font, fontweight='bold', ha='center')

# ==========================================
# 5. 绘制散点
# ==========================================
df_small = df[df['era'] == 'small']
df_large = df[df['era'] == 'large']

ax.scatter(df_small['date'], df_small['flops'], c='#1976d2', marker='o', s=60, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5, label='早期模型')
ax.scatter(df_large['date'], df_large['flops'], c='#d32f2f', marker='^', s=80, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=5, label='大语言模型')

# ==========================================
# 6. 绘制趋势线
# ==========================================
x_dates = mdates.date2num(df['date'])
y_flops = np.log10(df['flops'])

z = np.polyfit(x_dates, y_flops, 1)
p = np.poly1d(z)

xp = np.linspace(x_dates.min(), x_dates.max(), 100)
yp = p(xp)

ax.plot(mdates.num2date(xp), 10**yp, linestyle='--', color='gray', linewidth=2, alpha=0.6, label='增长趋势')

# ==========================================
# 7. 添加文本标注 (优化位置避免重叠)
# ==========================================
# 定义每个模型的特殊偏移 (避免重叠)
label_offsets = {
    'GPT-4': (-90, 1.15),
    'Claude 3': (-40, 1.05),
    'Gemini Ultra': (-170, 0.95),
    'GPT-3': (0, 1.1),
    'DeepSeek-R1': (-80, 0.90),
    'Grok-3': (10, 0.60),
    'GPT-5': (-80, 1.20),
    'Qwen-2 (72B)': (40, 0.5),
    'DeepSeek-V2': (40, 0.5),
    'Grok-1': (-130, 1.0),
    'Llama 2 (70B)': (-60, 0.90),
    'PaLM': (0, 1.1),
    'Switch Transformer': (-90, 0.90),
    'BERT-Large': (-60, 0.90),
    'GPT-2': (-60, 0.90),
    'Transformer': (-60, 0.90),
    'ResNet-152': (-60, 0.90),
    'VGG': (-60, 0.90),
    'AlexNet': (-60, 0.90),
}

for i, row in df.iterrows():
    # 获取特殊偏移，如果没有则使用默认
    if row['name'] in label_offsets:
        offset_x, offset_y = label_offsets[row['name']]
    else:
        offset_x = 0
        offset_y = 1.1 if row['era'] == 'large' else 0.90

    ax.text(row['date'] + pd.Timedelta(days=offset_x), 
            row['flops'] * offset_y*1.5, 
            row['name'], 
            fontsize=12,
            color='#333333',
            fontproperties=zh_font,
            ha='left',
            weight='bold')

# ==========================================
# 8. 设置坐标轴
# ==========================================
ax.set_yscale('log')
ax.set_ylim(1e18, 1e27)
ax.set_xlim(datetime(2012, 1, 1), datetime(2026, 1, 1))


ax.yaxis.set_minor_locator(NullLocator())
yticks = [10.0**i for i in range(18, 27)]
# 删除原来的 set_yticklabels，改用以下方式：
ax.set_yticks([10.0**i for i in range(18, 27)])

# 单独设置每个刻度标签的字体
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(16)

# 设置 xticklabels 使用 Times New Roman 字体
ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], fontproperties=times_font, fontsize=16)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_xlabel('发布时间', fontsize=16, fontproperties=zh_font)
ax.set_ylabel('训练计算量 (FLOPs)', fontsize=16, fontproperties=zh_font)

ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5, color='gray')

# ==========================================
# 9. Legend 设置 (修复字号问题)
# ==========================================
# 删除原来的 legend 创建代码，改用：
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, 
                   frameon=False)

# 强制设置每个文本项
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')  # 英文用 Times
    text.set_fontproperties(zh_font)         # 中文用宋体
    text.set_fontsize(18)                    # 明确设置字号
    text.set_weight('bold')                  # 加粗确保可见

# ==========================================
# 10. 保存与显示
# ==========================================
# 使用 bbox_inches='tight' 但给 legend 留空间
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为上方 legend 留出空间

output_filename = 'training_compute_trend_2025.png'
try:
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✅ 图表已成功保存为：{output_filename}")
except Exception as e:
    print(f"❌ 保存文件时出错：{e}")

# plt.show()