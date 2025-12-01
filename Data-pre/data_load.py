# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample as sp_resample
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DIR = "./RAW"
OUT_DIR = "./speech_dataset_parts"
OUT_CSV = "speech_dataset.csv"
TARGET_SR = 16000
MAX_LEN_SEC = 2
NUM_WORKERS = 64     # 你的超算，可以开满核
BATCH_SIZE = 50000    # 每个分片保存多少条数据

os.makedirs(OUT_DIR, exist_ok=True)

def load_wav_file(path, target_sr=TARGET_SR, max_len_sec=MAX_LEN_SEC):
    try:
        sr, audio = wavfile.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if target_sr is not None and sr != target_sr:
            num = int(len(audio) * target_sr / sr)
            if num <= 0:
                return None, None
            audio = sp_resample(audio, num)
            sr = target_sr

        if max_len_sec is not None:
            max_len = int(max_len_sec * sr)
            if len(audio) > max_len:
                audio = audio[:max_len]

        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs

        return audio, sr
    except:
        return None, None

def process_file(wav_file, label):
    audio, sr = load_wav_file(wav_file)
    if audio is None:
        return None
    filename = os.path.basename(wav_file)
    return audio, filename, label

def build_dataset_parallel():
    all_files = []
    for label in sorted(os.listdir(RAW_DIR)):
        label_path = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_path):
            continue
        files = sorted(glob.glob(os.path.join(label_path, "*.wav")))
        all_files.extend([(f, label) for f in files])

    waveforms, filenames, labels = [], [], []
    first_csv = True
    part_id = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, f, label): f for f, label in all_files}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            audio, filename, label = result
            print(f"Processed: {filename}")
            waveforms.append(audio)
            filenames.append(filename)
            labels.append(label)

            # 增量写 CSV（只包含 filename 和 label）
            df_csv = pd.DataFrame([[filename, label]], columns=["filename", "label"])
            if first_csv:
                df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", mode="w")
                first_csv = False
            else:
                df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", mode="a", header=False)

            # 达到 BATCH_SIZE -> 保存一个 NPZ 分片
            if len(waveforms) >= BATCH_SIZE:
                out_path = os.path.join(OUT_DIR, f"speech_dataset_part{part_id}.npz")
                np.savez_compressed(out_path,
                                    waveforms=np.array(waveforms, dtype=object),
                                    filenames=np.array(filenames, dtype=object),
                                    labels=np.array(labels, dtype=object))
                print(f"Saved {out_path} with {len(waveforms)} samples")
                part_id += 1
                waveforms, filenames, labels = [], [], []

    # 保存剩余未保存的数据
    if waveforms:
        out_path = os.path.join(OUT_DIR, f"speech_dataset_part{part_id}.npz")
        np.savez_compressed(out_path,
                            waveforms=np.array(waveforms, dtype=object),
                            filenames=np.array(filenames, dtype=object),
                            labels=np.array(labels, dtype=object))
        print(f"Saved {out_path} with {len(waveforms)} samples")

    print(f"Finished. NPZ parts -> {OUT_DIR}, CSV -> {OUT_CSV}")

if __name__ == "__main__":
    build_dataset_parallel()
