import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator  # 新增刻度控制模块

# 字体参数配置
FONT_CONFIG = {
    'title': 24,
    'axis_label': 24,
    'tick_label': 24,
    'legend': 24
}

# 估计数据（保持原数据不变）
x = [16, 32, 64, 128, 256, 512]
full = [36.26739789962768, 36.302566379547116, 36.48970317459107, 36.56101776885986, 36.802495056152345, 36.81123249816894]
no_mesh = [36.237874385833734, 36.57108349609375, 36.70331030273438, 36.75691581344605, 36.77123978042603, 36.78017133712768]
no_denoiser = [33.43153452301026, 34.75332764434815, 35.607677848815925, 36.116570192337036, 36.392456520080565, 36.541689655303955]
no_importance = [30.975474327087404, 32.0869763584137, 33.07429472732544, 34.04357278251648, 34.64539299583435, 35.12397266769409]

plt.figure(figsize=(10, 6), facecolor='white')

# 绘制三条折线（样式保持不变）
plt.plot(x, full, 'b', marker='o', linewidth=5, markersize=10, label='Full')
plt.plot(x, no_denoiser, 'orange', marker='o', linewidth=5, markersize=10, label='w/o denoiser')
plt.plot(x, no_importance, 'g', marker='o', linewidth=5, markersize=10, label='w/o MIS')
plt.plot(x, no_mesh, 'r', marker='o', linewidth=5, markersize=10, label='w/o mesh')

# 设置y轴刻度（关键修改部分）
plt.yticks(np.arange(30, 38, 1))  # 从10到30，步长5
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))  # 可选次刻度（更稀疏时不推荐）

# 字体放大处理
# plt.title('TensoIR armadillo', fontsize=FONT_CONFIG['title'], pad=20)
plt.xlabel('Samples', fontsize=FONT_CONFIG['axis_label'])
plt.ylabel('PSNR (dB)', fontsize=FONT_CONFIG['axis_label'])

# 设置刻度标签大小
plt.xticks(fontsize=FONT_CONFIG['tick_label'])
plt.yticks(fontsize=FONT_CONFIG['tick_label'])

# 图例字体放大
legend = plt.legend(
    loc='lower right',
    fontsize=FONT_CONFIG['legend'],
    framealpha=0.9
)

# 保持原有样式
plt.ylim(30, 37)
plt.xscale('log', base=2)
plt.xticks(x, labels=[str(i) for i in x])
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 优化布局
plt.tight_layout()
plt.savefig('ablation_curve_psnr.png', dpi=300)