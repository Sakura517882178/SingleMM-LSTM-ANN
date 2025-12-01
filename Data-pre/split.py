# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

INPUT_FILE = "speech_dataset_aligned.npz"
OUT_TRAIN = "train-cnn.npz"
OUT_VALID = "valid-cnn.npz"
OUT_TEST = "test-cnn.npz"

# 划分比例
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 少样本类别阈值
MIN_SAMPLES_FOR_STRATIFY = 20

def safe_concat(arr1, arr2):
    if arr1.size == 0:
        return arr2
    if arr2.size == 0:
        return arr1
    return np.concatenate([arr1, arr2])

def split_dataset():
    print(f"Loading {INPUT_FILE} ...")
    data = np.load(INPUT_FILE, allow_pickle=True)
    waveforms = data["waveforms"]
    filenames = data["filenames"]
    labels = data["labels"]

    label_counts = Counter(labels)
    print("Label counts:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    rare_labels = {l for l, c in label_counts.items() if c < MIN_SAMPLES_FOR_STRATIFY}
    common_labels = {l for l, c in label_counts.items() if c >= MIN_SAMPLES_FOR_STRATIFY}

    print(f"\nRare labels (<{MIN_SAMPLES_FOR_STRATIFY} samples): {rare_labels}")

    rare_indices = [i for i, lbl in enumerate(labels) if lbl in rare_labels]
    common_indices = [i for i, lbl in enumerate(labels) if lbl in common_labels]

    # -------- 大类别 --------
    X_common = waveforms[common_indices]
    y_common = labels[common_indices]
    f_common = filenames[common_indices]

    X_train_c, X_temp_c, y_train_c, y_temp_c, f_train_c, f_temp_c = train_test_split(
        X_common, y_common, f_common,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=y_common
    )

    test_size = TEST_RATIO / (VALID_RATIO + TEST_RATIO)
    X_valid_c, X_test_c, y_valid_c, y_test_c, f_valid_c, f_test_c = train_test_split(
        X_temp_c, y_temp_c, f_temp_c,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_temp_c
    )

    # -------- 稀有类别 --------
    X_rare = waveforms[rare_indices]
    y_rare = labels[rare_indices]
    f_rare = filenames[rare_indices]

    if len(X_rare) > 0:
        n_samples = len(X_rare)
        # 至少留一个训练样本
        if n_samples > 1:
            test_size_rare = min(1 - TRAIN_RATIO, (n_samples - 1) / n_samples)
        else:
            test_size_rare = 0.0

        X_train_r, X_temp_r, y_train_r, y_temp_r, f_train_r, f_temp_r = train_test_split(
            X_rare, y_rare, f_rare,
            test_size=test_size_rare,
            random_state=RANDOM_SEED,
            stratify=None
        )

        # 再切验证/测试
        if len(X_temp_r) > 1:
            test_size_sub = TEST_RATIO / (VALID_RATIO + TEST_RATIO)
            X_valid_r, X_test_r, y_valid_r, y_test_r, f_valid_r, f_test_r = train_test_split(
                X_temp_r, y_temp_r, f_temp_r,
                test_size=test_size_sub,
                random_state=RANDOM_SEED,
                stratify=None
            )
        else:
            # 只剩 1 个样本，全放验证集
            X_valid_r, y_valid_r, f_valid_r = X_temp_r, y_temp_r, f_temp_r
            X_test_r, y_test_r, f_test_r = np.array([]), np.array([]), np.array([])
    else:
        X_train_r = X_valid_r = X_test_r = np.array([])
        y_train_r = y_valid_r = y_test_r = np.array([])
        f_train_r = f_valid_r = f_test_r = np.array([])

    # -------- 合并 --------
    X_train = safe_concat(X_train_c, X_train_r)
    y_train = safe_concat(y_train_c, y_train_r)
    f_train = safe_concat(f_train_c, f_train_r)

    X_valid = safe_concat(X_valid_c, X_valid_r)
    y_valid = safe_concat(y_valid_c, y_valid_r)
    f_valid = safe_concat(f_valid_c, f_valid_r)

    X_test = safe_concat(X_test_c, X_test_r)
    y_test = safe_concat(y_test_c, y_test_r)
    f_test = safe_concat(f_test_c, f_test_r)

    # -------- 保存 --------
    np.savez_compressed(OUT_TRAIN, waveforms=X_train, labels=y_train, filenames=f_train)
    np.savez_compressed(OUT_VALID, waveforms=X_valid, labels=y_valid, filenames=f_valid)
    np.savez_compressed(OUT_TEST,  waveforms=X_test,  labels=y_test,  filenames=f_test)

    print(f"Train set: {len(X_train)} samples -> {OUT_TRAIN}")
    print(f"Valid set: {len(X_valid)} samples -> {OUT_VALID}")
    print(f"Test set:  {len(X_test)} samples -> {OUT_TEST}")

if __name__ == "__main__":
    split_dataset()
