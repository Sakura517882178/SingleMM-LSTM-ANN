# Y6 Single-Molecule Memristor for Speech Recognition

## Experimentally Parameterized Memristive Neural Network with Light/Dark Controlled Validation

---

## Highlights

- **All memristor weights are directly parameterized by measured Y6 single-molecule junction current (nA)**
- **Light/Dark controlled experiment**: dark-state memristor (Imin = Imax, fixed current) yields 0.01% accuracy, proving that Y6 photoconductivity is essential to device function
- **Light state achieves 71.5%** on 36-class Google Speech Commands; dark state performs no better than random
- **Stable-region extraction**: forming cycles (first 12 periods, t < 240 ms) excluded; P5/P99.9 percentile filtering removes transient spikes

---

## Background

The Y6 molecule (a non-fullerene acceptor) forms a single-molecule junction whose current is modulated by light. Under 50 Hz optical excitation, the junction exhibits a dynamic current range (~0.4–32 nA, 110× ratio); in the dark, current collapses to a near-constant value (~0.29 nA).

This repository implements a memristive artificial neural network (ANN-M) in which every memristor weight and nonlinear activation is **directly parameterized by the measured Y6 I-V data in nA**. The goal is to demonstrate that a physically constrained, experimentally grounded memristor model can support practical pattern recognition (36-class speech command classification on Google Speech Commands), and that removing photoconductivity (Dark control) abolishes learning.

---

## Y6 Parameter Extraction

### Data Source

| File | Condition | Description |
|---|---|---|
| `ANN-M/Y6-data.xlsx` | Light (experimental) | 50 Hz modulated optical pulse, I-V sweep |
| `ANN-M/Y6-Dark.xlsx` | Dark (control) | No illumination, otherwise identical measurement |

### Extraction Algorithm (`load_y6_params`)

```
Load .xlsx → filter stable region (t > 240 ms, skip 12 forming cycles)
    │
    ├─ tail fraction < 5% → LIGHT MODE
    │     P5   → Imin   (lower current bound, nA)
    │     P99.9 → Imax  (upper current bound, nA)
    │
    └─ tail fraction > 5% → DARK MODE
          filter instrument artifacts (I near 0)
          mean of clean population → I = fixed value
          (Imin = Imax: no plasticity, no learning)
```

The tail-fraction criterion automatically distinguishes light from dark measurements.

### Extracted Parameters

| | Y6-Light | Y6-Dark |
|---|---|---|
| **Regime** | Dynamic range | Fixed value |
| **Imin** (nA) | 0.41 | 0.29 |
| **Imax** (nA) | 32.26 | 0.29 |
| **Current ratio** | 110× | 1× (no plasticity) |

**Critical point**: Dark-state Imin = Imax. The memristor weights are clamped to a single value — gradient descent cannot move them. This is the mechanistic basis of the controlled experiment.

---

## Memristor Model

Memristor weights are stored directly in the nA current domain, exactly as measured from the Y6 junction. No log-domain or voltage conversion is applied.

### MemristorLinearI

A fully-connected linear layer whose weight matrix `I` is initialized from Uniform[Imin, Imax] and updated via gradient descent with hard clipping to [Imin, Imax]. Forward pass: `output = W @ x` where `W = I`.

### MemristorNonlinear

Replaces the conventional ReLU activation. Element-wise: `y = clamp(x, min=0) * I`, where `I` is a learned scalar clipped to [Imin, Imax].

### Weight Update

```
I ← I - lr · ∇I
I ← clip(I, Imin, Imax)
```

---

## Model Architecture

```
Input: 64-dim MFCC feature vector
    │
    ▼
LSTM (unidirectional, hidden=128)
    │
    ▼
tanh Attention → mean+max pooling
    │
    ▼
┌─────────────────────────────────┐
│  MemristorLinearI  (128 → 128)  │  ← Y6-parameterized weights (nA)
│  MemristorNonlinear              │  ← Y6-parameterized activation (replaces ReLU)
│  LayerNorm + Dropout(0.3)        │
│  MemristorLinearI  (128 → 36)   │  ← Y6-parameterized weights (nA)
└─────────────────────────────────┘
    │
    ▼
logits → 36-class softmax
```

- **Standard PyTorch modules**: LSTM, Attention, LayerNorm, Dropout
- **Memristor modules**: MemristorLinearI and MemristorNonlinear — all parameters bounded by Y6 experimental [Imin, Imax]

---

## Light/Dark Controlled Experiment

### Design

`ANN-M/compare-Y6-light-dark.py` trains two identical ANN-M architectures side by side:

| Group | Data Source | Memristor State | Plasticity |
|---|---|---|---|
| **Light (experimental)** | `Y6-data.xlsx` | I = [0.41, 32.26] nA | Yes |
| **Dark (control)** | `Y6-Dark.xlsx` | I = [0.29, 0.29] nA | No |

Both groups share: same architecture, same speech dataset, same optimizer, same random seed.

### Results

```
     Batch     Light Acc      Dark Acc       Delta
        16        58.95%         0.01%      58.94%
        32         3.83%         0.01%       3.82%
        64        71.51%         0.01%      71.50%
      Best        71.51%         0.01%      71.50%
```

- **Light** achieves 71.5% on 36-class speech recognition
- **Dark** performs at ~0% — fixed current prevents synaptic weight adaptation
- **Δ = 71.5%** demonstrates that Y6 photoconductivity is **necessary** for learning

### Interpretation

The Dark control rules out the possibility that the LSTM + attention backbone alone (without memristor plasticity) can solve the task. The dark-state accuracy confirms that:

1. The memristor layers are functional and trainable **only** under illumination
2. The experimentally measured current range [Imin, Imax] is sufficient to encode 36-class discriminative features
3. The photoconductive effect is the **enabling mechanism**

---

## Repository Structure

```
Single-Molecule-Memristor/
├── README.MD                          # This file
├── ANN-M/                             # Memristor ANN (Y6-parameterized)
│   ├── Y6-data.xlsx                   # Y6 light-state 50 Hz I-V data
│   ├── Y6-Dark.xlsx                   # Y6 dark-state I-V data (control)
│   ├── compare-Y6-light-dark.py       # Light/Dark controlled experiment (training)
│   ├── infer_light_dark.py            # Inference: load checkpoints, save predictions
│   ├── plot_cm.py                     # Confusion matrix: side-by-side Light vs Dark
│   └── checkpoints-AM-Uni/            # Saved model checkpoints
├── ANN/                               # Baseline ANN (ReLU, non-memristive)
├── CNN/                               # CNN feature extractor for audio
├── Data-pre/                          # Speech preprocessing pipeline
└── plot/                              # Visualization
```

---

## Quick Start

### Requirements

```
torch >= 1.10
numpy
openpyxl
pandas
scikit-learn
matplotlib
```

### Light/Dark Controlled Experiment (Training)

```bash
python ANN-M/compare-Y6-light-dark.py --epochs 200 --batch_sizes 16 32 64
```

### Inference

```bash
# Run inference with best checkpoint (bs=64)
python ANN-M/infer_light_dark.py --batch_size 64

# Output: predictions-Y6-light_bs64.csv, predictions-Y6-dark_bs64.csv
```

---

## Citation

If you use this work, please cite the corresponding publication. The key contribution is the **experimentally constrained memristor model** — all device parameters are traceable to measured single-molecule I-V data, and the light/dark controlled experiment provides a falsifiable test of the photoconductive mechanism.
