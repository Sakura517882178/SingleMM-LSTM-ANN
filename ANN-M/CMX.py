import pandas as pd
from sklearn.metrics import confusion_matrix

# 读取 CSV
df = pd.read_csv("predictions-ANN-M.csv")  
y_true = df['true'].values
y_pred = df['pred'].values

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 转为 DataFrame，便于保存为 CSV
cm_df = pd.DataFrame(cm)

# 可选：给行列加标签（真实类别/预测类别）
cm_df.index.name = 'True'
cm_df.columns.name = 'Predicted'

# 保存为 CSV
cm_df.to_csv("confusion_matrix-ANN-M.csv", index=True)

print("📄 混淆矩阵已保存为 confusion_matrix.csv")
