import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import random
import os
import csv

# ---------------- 数据集 ----------------
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.sequences = data["sequences"].astype(np.float32)
        self.labels = np.array(data["labels"])
        self.filenames = data["filenames"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.filenames[idx]

# ---------------- 模型 ----------------
class FrameCNN(nn.Module):
    def __init__(self, frame_len, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        feat = self.fc(x)
        return F.normalize(feat, dim=1)

# ---------------- triplet 生成 ----------------
def generate_triplets(labels):
    label_to_idx = defaultdict(list)
    for idx, l in enumerate(labels):
        label_to_idx[l].append(idx)
    triplets = []
    for idx, l in enumerate(labels):
        anchor_idx = idx
        positive_idx = random.choice(label_to_idx[l])
        negative_label = random.choice([x for x in label_to_idx.keys() if x != l])
        negative_idx = random.choice(label_to_idx[negative_label])
        triplets.append((anchor_idx, positive_idx, negative_idx))
    return triplets

# ---------------- 单次训练 ----------------
def train_one_batchsize(npz_file, batch_size, feature_dim=64, epochs=1000,
                        early_stop_patience=200, model_file="cnn_model.pth"):
    dataset = FrameDataset(npz_file)
    sequences = dataset.sequences
    labels = dataset.labels
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FrameCNN(frame_len=sequences.shape[1], feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=50,
                                                           verbose=False)
    best_loss = float('inf')
    epochs_no_improve = 0
    logs = []

    for epoch in range(epochs):
        model.train()
        triplets = generate_triplets(labels)
        random.shuffle(triplets)
        total_loss = 0

        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i+batch_size]
            if len(batch_triplets) == 0:
                continue
            a_idx, p_idx, n_idx = zip(*batch_triplets)
            a = torch.tensor(sequences[list(a_idx)], dtype=torch.float32).unsqueeze(1).to(device)
            p = torch.tensor(sequences[list(p_idx)], dtype=torch.float32).unsqueeze(1).to(device)
            n = torch.tensor(sequences[list(n_idx)], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            feat_a, feat_p, feat_n = model(a), model(p), model(n)
            loss = triplet_loss(feat_a, feat_p, feat_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_triplets)

        avg_loss = total_loss / len(triplets)
        logs.append({"epoch": epoch+1, "avg_loss": avg_loss})
        scheduler.step(avg_loss)

        # ----- 实时输出 -----
        print(f"[Batch {batch_size}] Epoch {epoch+1}/{epochs}, Avg Loss = {avg_loss:.6f}, Best Loss = {best_loss:.6f}")

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_loss, logs

# ---------------- 自动搜索 batch_size ----------------
def search_best_batch(npz_file, candidate_batches=[64,128,256],
                      feature_dim=64, epochs=1000, early_stop_patience=200,
                      save_dir="train_logs"):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    best_batch = None
    best_logs = None
    best_model_file = None

    for bs in candidate_batches:
        print(f"\n=== Testing batch_size={bs} ===")
        model_file = os.path.join(save_dir, f"best_model_bs{bs}.pth")
        try:
            loss, logs = train_one_batchsize(npz_file, batch_size=bs, feature_dim=feature_dim,
                                             epochs=epochs, early_stop_patience=early_stop_patience,
                                             model_file=model_file)
            print(f"Batch {bs}, Best Loss={loss:.6f}")

            # 保存 CSV
            log_file = os.path.join(save_dir, f"train_log_bs{bs}.csv")
            with open(log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "avg_loss"])
                writer.writeheader()
                for entry in logs:
                    writer.writerow(entry)

            if loss < best_loss:
                best_loss = loss
                best_batch = bs
                best_logs = logs
                best_model_file = model_file

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch_size={bs}, skipping...")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"\n>>> Best batch_size={best_batch}, loss={best_loss:.6f}")
    print(f"Best model saved at {best_model_file}")
    return best_batch, best_model_file, best_logs

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    best_batch, best_model_file, best_logs = search_best_batch(
        npz_file="train-cnn_seq.npz",
        candidate_batches=[64,128,256],
        feature_dim=64,
        epochs=2000,
        early_stop_patience=100,
        save_dir="cnn_train_logs"
    )
