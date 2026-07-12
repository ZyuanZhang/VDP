import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# 读取数据
inputpath = './XGB-embedding_GO'
df = pd.read_csv(inputpath+'/pr_data.csv', sep='\t')  # 根据你前面的描述，还是制表符

# 统一 recall 横坐标点
mean_recall = np.linspace(0, 1, 100)
interp_precisions = []

plt.figure(figsize=(8, 8))

# 绘制灰色每一折 PR 曲线
for i in range(5):
    recall = df[f'Recall{i}'].dropna()
    precision = df[f'Precision{i}'].dropna()

    # 截取最短长度，防止对不上
    min_len = min(len(recall), len(precision))
    recall = recall.iloc[:min_len]
    precision = precision.iloc[:min_len]

    # 保证 recall 单调递增（插值要求）
    sort_idx = np.argsort(recall)
    recall_sorted = recall.iloc[sort_idx].values
    precision_sorted = precision.iloc[sort_idx].values

    plt.plot(recall_sorted, precision_sorted, color='lightgray', linewidth=2)

    interp_precision = np.interp(mean_recall, recall_sorted, precision_sorted)
    interp_precisions.append(interp_precision)

# 均值 PR 曲线
mean_precision = np.mean(interp_precisions, axis=0)
pr_auc = auc(mean_recall, mean_precision)

# 绘制红色均值曲线
plt.plot(mean_recall, mean_precision, color='red', linewidth=3, label=f'Mean PR (AUC = {pr_auc:.3f})')
# plt.plot(mean_recall, mean_precision, color='red', linewidth=3, label=f'Mean PR (AUC = 0.874)')

# 图像设置
plt.xlabel('Recall', fontsize=30)
plt.ylabel('Precision', fontsize=30)
# plt.title('Mean PR Curve over 5-Fold CV')
plt.legend(loc='lower left', fontsize=20)

# 坐标轴刻度大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 加粗边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

# plt.grid(True)
plt.tight_layout()
plt.plot([0, 1], [1, 0.5], 'k--', lw=2)
plt.savefig(inputpath+'/pr.png', dpi=300)
plt.show()
