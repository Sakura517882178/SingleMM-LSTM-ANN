# -*- coding: utf-8 -*-
import os
import glob
import numpy as np

PART_DIR = "./speech_dataset_parts"
ALIGNED_NPZ = "speech_dataset_aligned.npz"
TARGET_SR = 16000
MAX_LEN_SEC = 2
FIXED_LEN = TARGET_SR * MAX_LEN_SEC   # 32000

def pad_or_trim(audio, fixed_len=FIXED_LEN):
    """对齐长度：不足补0，超长截断"""
    if len(audio) < fixed_len:
        padded = np.zeros(fixed_len, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded
    else:
        return audio[:fixed_len].astype(np.float32)

def align_waveforms():
    part_files = sorted(glob.glob(os.path.join(PART_DIR, "speech_dataset_part*.npz")))
    if not part_files:
        print("No parts found!")
        return

    all_waveforms, all_filenames, all_labels = [], [], []

    for i, part in enumerate(part_files):
        print(f"Loading {part} ...")
        data = np.load(part, allow_pickle=True)
        waveforms = [pad_or_trim(w) for w in data["waveforms"]]
        filenames = list(data["filenames"])
        labels = list(data["labels"])

        all_waveforms.extend(waveforms)
        all_filenames.extend(filenames)
        all_labels.extend(labels)

    all_waveforms = np.stack(all_waveforms).astype(np.float32)  # 变成 (N, FIXED_LEN)
    all_filenames = np.array(all_filenames, dtype=object)
    all_labels = np.array(all_labels, dtype=object)

    print(f"Aligned dataset shape: {all_waveforms.shape}")

    np.savez_compressed(ALIGNED_NPZ,
                        waveforms=all_waveforms,
                        filenames=all_filenames,
                        labels=all_labels)
    print(f"Saved aligned dataset -> {ALIGNED_NPZ}")

if __name__ == "__main__":
    align_waveforms()
