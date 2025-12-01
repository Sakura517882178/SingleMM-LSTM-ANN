import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import csv

# ===========================
# Dataset（同训练）
# ===========================
class SpeechFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.X = data["X"]
        self.y = data["y"] if "y" in data else None  # 推理时可能没有标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y
        else:
            return x

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        feats, labels = zip(*batch)
        return torch.stack(feats, dim=0), torch.stack(labels, dim=0)
    else:
        return torch.stack(batch, dim=0)

# ===========================
# 模型定义（与训练一致）
# ===========================
class ANN_LSTM_Attn(torch.nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, lstm_layers=1, fc_hidden=128, num_classes=36, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=False)
        self.attn_weight = torch.nn.Parameter(torch.Tensor(hidden_dim, 1))
        torch.nn.init.xavier_uniform_(self.attn_weight)
        self.fc1 = torch.nn.Linear(hidden_dim, fc_hidden)
        self.layernorm = torch.nn.LayerNorm(fc_hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = torch.tanh(lstm_out @ self.attn_weight)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)
        x = torch.relu(self.fc1(pooled))
        x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# ===========================
# 推理函数
# ===========================
def inference(model_path, npz_file, batch_size=32, output_csv="predictions.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据
    dataset = SpeechFeatureDataset(npz_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 确定类别数（如果训练时固定可直接指定）
    num_classes = 36  # 或根据训练时设置
    model = ANN_LSTM_Attn(input_dim=64, hidden_dim=128, lstm_layers=1,
                          fc_hidden=128, num_classes=num_classes, dropout=0.0).to(device)
    
    # 加载训练好的权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_labels = []  # 如果有标签
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                X, y = batch
                X = X.to(device)
            else:
                X = batch.to(device)
                y = None

            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            if y is not None:
                all_labels.extend(y.numpy())

    # 保存结果到 CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        if all_labels:
            writer.writerow(["pred", "true"])
            for p, t in zip(all_preds, all_labels):
                writer.writerow([p, t])
        else:
            writer.writerow(["pred"])
            for p in all_preds:
                writer.writerow([p])

    print(f"✅ 推理完成，结果已保存到 {output_csv}")
    return all_preds

# ===========================
# 示例调用
# ===========================
if __name__ == "__main__":
    preds = inference(
        model_path="checkpoints-AB/best_lstm_attn_ft.pth",
        npz_file="test_features.npz",
        batch_size=64,
        output_csv="predictions.csv"
    )
