"""
推理脚本：加载 Light/Dark 最佳权重，保存预测结果 CSV
基于 compare-Y6-light-dark.py 的模型架构和参数加载逻辑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os, csv, openpyxl, argparse

# ===========================
# Y6 参数加载 (与 compare-Y6-light-dark.py 一致)
# ===========================
def load_y6_params(xlsx_path, stable_after_ms=240.0):
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    time, current = [], []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] is not None: time.append(float(row[0]))
        if row[2] is not None: current.append(float(row[2]))
    t_arr, I_arr = np.array(time), np.array(current)
    I_stable = I_arr[t_arr > stable_after_ms]
    median_I = float(np.median(I_stable))
    tail_frac = np.mean(I_stable < median_I * 0.1)
    if tail_frac > 0.05:
        I_clean = I_stable[I_stable > median_I * 0.1]
        I_val = float(np.mean(I_clean))
        return I_val, I_val
    else:
        return float(np.percentile(I_stable, 5)), float(np.percentile(I_stable, 99.9))

# ===========================
# Dataset (与 compare-Y6-light-dark.py 一致)
# ===========================
class SpeechFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.X, self.y = data["X"], data["y"]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def collate_fn(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats, dim=0), torch.stack(labels, dim=0)

# ===========================
# 忆阻器模块 (与 compare-Y6-light-dark.py 一致)
# ===========================
class MemristorLinearI(nn.Module):
    def __init__(self, in_features, out_features, Imin, Imax, step=None, noise=0.0):
        super().__init__()
        self.Imin, self.Imax = Imin, Imax
        self.step = step if step is not None else (Imax - Imin) / 100
        self.noise = noise
        self.I = nn.Parameter(torch.empty(out_features, in_features).uniform_(Imin, Imax))
    def forward(self, x): return F.linear(x, self.I)

class MemristorNonlinear(nn.Module):
    def __init__(self, Imin, Imax, step=None, noise=0.0):
        super().__init__()
        self.Imin, self.Imax = Imin, Imax
        self.step = step if step is not None else (Imax - Imin) / 100
        self.noise = noise
        self.I = nn.Parameter(torch.empty(1).uniform_(Imin, Imax))
    def forward(self, x): return x.clamp(min=0.0) * self.I

# ===========================
# 模型 (与 compare-Y6-light-dark.py 一致)
# ===========================
class MemristorANN_LSTM_Attn(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, lstm_layers=1,
                 fc_hidden=128, num_classes=36, dropout=0.3, Imin=0.4, Imax=32.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=False)
        self.attn_weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attn_weight)
        self.fc1 = MemristorLinearI(hidden_dim, fc_hidden, Imin, Imax)
        self.layernorm = nn.LayerNorm(fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.nonlinear = MemristorNonlinear(Imin, Imax)
        self.fc2 = MemristorLinearI(fc_hidden, num_classes, Imin, Imax)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = torch.tanh(lstm_out @ self.attn_weight)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)
        x = self.nonlinear(self.fc1(pooled))
        x = self.layernorm(x)
        x = self.dropout(x)
        return self.fc2(x)

# ===========================
# 推理
# ===========================
def run_inference(checkpoint_path, y6_file, test_npz, batch_size, device="cpu"):
    Imin, Imax = load_y6_params(y6_file)
    dataset = SpeechFeatureDataset(test_npz)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    n_cls = len(np.unique(dataset.y))

    model = MemristorANN_LSTM_Attn(input_dim=64, hidden_dim=128, fc_hidden=128,
                                   num_classes=n_cls, Imin=Imin, Imax=Imax).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = (y_pred == y_true).mean()
    return y_true, y_pred, acc

# ===========================
# Main
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="which checkpoint to use (16/32/64)")
    parser.add_argument("--test_npz", type=str, default="/mnt/f/SLYY/final/test_features.npz")
    parser.add_argument("--ckpt_dir", type=str, default="/mnt/f/SLYY/ANN-M/checkpoints-AM-Uni")
    parser.add_argument("--light_file", type=str, default="/mnt/f/SLYY/ANN-M/Y6-data.xlsx")
    parser.add_argument("--dark_file", type=str, default="/mnt/f/SLYY/ANN-M/Y6-Dark.xlsx")
    parser.add_argument("--out_dir", type=str, default="/mnt/f/SLYY/ANN-M")
    args = parser.parse_args()

    bs = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Batch size: {bs}")

    light_ckpt = f"{args.ckpt_dir}/best_memristor_lstm_attn_bs{bs}_light.pth"
    dark_ckpt  = f"{args.ckpt_dir}/best_memristor_lstm_attn_bs{bs}_dark.pth"

    print(f"\nInference: Light ({light_ckpt})")
    yt_l, yp_l, acc_l = run_inference(light_ckpt, args.light_file, args.test_npz, bs, device)
    print(f"  Acc = {acc_l*100:.2f}%")

    print(f"Inference: Dark ({dark_ckpt})")
    yt_d, yp_d, acc_d = run_inference(dark_ckpt, args.dark_file, args.test_npz, bs, device)
    print(f"  Acc = {acc_d*100:.2f}%")

    # 保存预测 CSV
    for tag, yt, yp in [("light", yt_l, yp_l), ("dark", yt_d, yp_d)]:
        csv_path = f"{args.out_dir}/predictions-Y6-{tag}_bs{bs}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true", "pred"])
            for t, p in zip(yt, yp):
                writer.writerow([t, p])
        print(f"Saved: {csv_path}")

    print(f"\nSummary: Light={acc_l*100:.2f}%  Dark={acc_d*100:.2f}%  Delta={(acc_l-acc_d)*100:.2f}%")

if __name__ == "__main__":
    main()
