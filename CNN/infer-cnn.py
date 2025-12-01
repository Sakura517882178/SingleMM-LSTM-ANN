# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os

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

def load_dataset(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    return data["sequences"].astype(np.float32), data["filenames"]

def extract_features(model, sequences, filenames, device, output_dir="feature_csv", batch_size=64):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    features_dict = defaultdict(list)

    for i in range(0, len(sequences), batch_size):
        batch_seq = torch.tensor(sequences[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            batch_feat = model(batch_seq)
        batch_feat = batch_feat.cpu().numpy()
        batch_files = filenames[i:i+batch_size]
        for f, feat in zip(batch_files, batch_feat):
            file_id = "_".join(f.split("_")[:-1])
            features_dict[file_id].append(feat)

    for fid, feats in features_dict.items():
        feat_matrix = np.stack(feats, axis=0)
        out_file = os.path.join(output_dir, f"{fid}.csv")
        np.savetxt(out_file, feat_matrix, delimiter=",")
        print(f"Saved {out_file}, shape={feat_matrix.shape}")

if __name__ == "__main__":
    npz_list = ["train_seq.npz", "valid_seq.npz", "test_seq.npz"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64

    sequences_dummy, _ = load_dataset("train_seq.npz")
    model = FrameCNN(frame_len=sequences_dummy.shape[1], feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load("./cnn_train_logs/best_model_bs256.pth", map_location=device))

    for npz_file in npz_list:
        seqs, files = load_dataset(npz_file)
        extract_features(model, seqs, files, device, output_dir=f"feature_csv_{os.path.splitext(npz_file)[0]}")
