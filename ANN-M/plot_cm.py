"""
混淆矩阵脚本：读取推理 CSV，画 Light/Dark 对比混淆矩阵
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse

def plot_cm(y_true, y_pred, n_cls, title, acc, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_cls))
    cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='RdPu', vmin=0, vmax=1, aspect='auto')

    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_ylabel('True label', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)

    # 网格线
    for i in range(n_cls + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.3)
        ax.axvline(i - 0.5, color='white', linewidth=0.3)

    # 刻度每6个标一个
    step = max(1, n_cls // 6)
    ticks = np.arange(0, n_cls, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=11)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Normalized accuracy', fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light_csv", type=str, default="/mnt/f/SLYY/ANN-M/predictions-Y6-light_bs64.csv")
    parser.add_argument("--dark_csv", type=str, default="/mnt/f/SLYY/ANN-M/predictions-Y6-dark_bs64.csv")
    parser.add_argument("--out_dir", type=str, default="/mnt/f/SLYY/ANN-M")
    parser.add_argument("--n_classes", type=int, default=36)
    args = parser.parse_args()

    df_l = pd.read_csv(args.light_csv)
    df_d = pd.read_csv(args.dark_csv)
    yt_l, yp_l = df_l["true"].values, df_l["pred"].values
    yt_d, yp_d = df_d["true"].values, df_d["pred"].values

    n_cls = args.n_classes
    acc_l = (yp_l == yt_l).mean()
    acc_d = (yp_d == yt_d).mean()
    print(f"Light Acc = {acc_l*100:.2f}%")
    print(f"Dark Acc  = {acc_d*100:.2f}%")

    # 单独保存
    plot_cm(yt_l, yp_l, n_cls,
            "Y6-Light (memristive plasticity)", acc_l,
            f"{args.out_dir}/confusion_matrix_light.png")
    plot_cm(yt_d, yp_d, n_cls,
            "Y6-Dark (no plasticity, fixed conductance)", acc_d,
            f"{args.out_dir}/confusion_matrix_dark.png")

    # 并排对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

    for ax, yt, yp, title, acc in [
        (ax1, yt_l, yp_l,
         f"a) Y6-Light (memristive plasticity)\nAcc = {acc_l*100:.2f}%", acc_l),
        (ax2, yt_d, yp_d,
         f"b) Y6-Dark (no plasticity)\nAcc = {acc_d*100:.2f}%", acc_d)]:
        cm = confusion_matrix(yt, yp, labels=range(n_cls))
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1, keepdims=True))
        im = ax.imshow(cm_norm, cmap='RdPu', vmin=0, vmax=1, aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)

        for i in range(n_cls + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.3)
            ax.axvline(i - 0.5, color='white', linewidth=0.3)

        step = max(1, n_cls // 6)
        ticks = np.arange(0, n_cls, step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Normalized accuracy', fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/confusion_matrix_comparison.png", dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {args.out_dir}/confusion_matrix_comparison.png")
    print("Done.")

if __name__ == "__main__":
    main()
