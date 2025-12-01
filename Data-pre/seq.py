# -*- coding: utf-8 -*-
import numpy as np
import os

FRAME_MS = 40
TARGET_SR = 16000

def waveform_to_sequences(waveform, frame_ms=FRAME_MS, sr=TARGET_SR):
    frame_len = int(frame_ms / 1000 * sr)
    num_frames = len(waveform) // frame_len
    waveform = waveform[:num_frames * frame_len]
    frames = np.reshape(waveform, (num_frames, frame_len))
    frames = frames / (np.max(np.abs(frames)) + 1e-8)
    return frames

def process_split(input_file, output_file_prefix):
    data = np.load(input_file, allow_pickle=True)
    waveforms = data["waveforms"]
    labels = data["labels"]
    filenames = data["filenames"]

    sequences_list = []
    labels_list = []
    filenames_list = []

    for wf, lbl, fname in zip(waveforms, labels, filenames):
        frames = waveform_to_sequences(wf)
        num_frames = frames.shape[0]
        for i in range(num_frames):
            sequences_list.append(frames[i])
            labels_list.append(lbl)
            # 生成序列唯一标识
            filenames_list.append(f"{fname}_seq{i}")

    sequences_array = np.array(sequences_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=object)
    filenames_array = np.array(filenames_list, dtype=object)

    out_file = f"{output_file_prefix}_seq.npz"
    np.savez_compressed(out_file,
                        sequences=sequences_array,
                        labels=labels_array,
                        filenames=filenames_array)
    print(f"Processed {len(sequences_array)} sequences -> {out_file}")
    print(f"Each sequence shape: {sequences_array.shape[1:]}")

if __name__ == "__main__":
    # 可以依次处理 train/valid/test
    for split in ["train", "valid", "test"]:
        input_file = f"{split}.npz"
        output_file_prefix = split
        process_split(input_file, output_file_prefix)
