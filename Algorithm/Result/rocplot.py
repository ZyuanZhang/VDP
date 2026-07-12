import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# 读取制表符分隔的CSV
inputpath = './XGB-embedding_GO'
df = pd.read_csv(inputpath + '/roc_data.csv', sep='\t')

# 统一FPR
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

plt.figure(figsize=(8, 8))

# 每折灰色曲线 + 插值
for i in range(5):
    fpr = df[f'FPR{i}']
    tpr = df[f'TPR{i}']
    plt.plot(fpr, tpr, color='lightgray', linewidth=2)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

# 平均TPR & AUC
mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

# 绘制均值曲线（红色）
plt.plot(mean_fpr, mean_tpr, color='red', linewidth=3, label=f'Mean ROC (AUC = {mean_auc:.3f})')
# plt.plot(mean_fpr, mean_tpr, color='red', linewidth=3, label=f'Mean ROC (AUC = 0.891)')

# 图设置
plt.plot([0, 1], [0, 1], 'k--', lw=2)
#  坐标轴标签大小
plt.xlabel('False Positive Rate (FPR)', fontsize=30)
plt.ylabel('True Positive Rate (TPR)', fontsize=30)
# 坐标轴刻度大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 标题
# plt.title('Mean ROC Curve over 5-Fold CV', fontsize=30)
# 图例
plt.legend(loc='lower right', fontsize=20)
#  网格
# plt.grid(True)
#   tight_layout
plt.tight_layout()
# 加粗边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.savefig(inputpath+'/roc.png',dpi =300)
plt.show()
