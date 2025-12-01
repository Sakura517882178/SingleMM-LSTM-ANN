# **Single-Molecule Memristor Realizing Synaptic Plasticity for Neuromorphic Applications**

This repository introduces a neuromorphic computing system in which a **single-molecule memristor replaces the conventional ReLU activation function** in neural networks.  
By incorporating synaptic plasticity properties directly into activation dynamics, the network demonstrates biologically inspired nonlinear processing capabilities, evaluated using the **Google Speech Commands public dataset**.

---

## 🔥 Core Concept

| Traditional ANN | Proposed ANN-M |
|---|---|
| Digital ReLU activation | **Memristor-based activation** |
| Numerical nonlinearity | Physical conductance modulation |

We provide both the **baseline ANN model** and the **memristor-enhanced ANN-M model**, enabling direct performance comparison.

---

## 📁 Repository Structure
Single-Molecule-Memristor/
│── CNN/ # CNN feature extractor for speech command audio
│── ANN/ # Baseline ANN classifier with ReLU activation
│── ANN-M/ # Modified ANN using memristive activation neurons
│── Data-pre/ # Speech dataset preprocessing / MFCC extraction
│── plot/ # Confusion matrix + plotting scripts
│── README.md # Project documentation


---

## 🧠 Workflow Summary

1. **Data-pre — Speech preprocessing**
   - Google Speech Commands waveforms are loaded and normalized
   - MFCC / log-mel spectrogram features generated
   - Data stored for training / inference

2. **CNN — Feature extraction**
   - CNN encodes temporal-frequency representations
   - Feature vectors fed into classification network

3. **ANN vs ANN-M — Model comparison**
   - ANN: classical ReLU neurons
   - ANN-M: **memristor activation modules** for synaptic-like response

4. **plot — Visualization**
   - Confusion matrix generation
   - Classification accuracy visualization

---

## 📊 Dataset

**Google Speech Commands Dataset (Google AI)**  
Short utterances — e.g. *yes, no, stop, go, left, right,…*  
Used to benchmark classification accuracy under biological-inspired activation.

---

## 🚀 Usage

```bash
git clone https://github.com/your-repo/Single-Molecule-Memristor.git
cd Single-Molecule-Memristor

# Step 1 - Data preprocessing
python Data-pre/data_load.py
python Data-pre/merged.py
python Data-pre/split.py
python Data-pre/seq.py

# Step 2 - CNN Feature extraction
python CNN/cnn-train.py
python CNN/infer-cnn.py

# Step 3 - Train baseline model
python ANN/training-ANN.py
python ANN/finetune-ANN.py

# Step 4 - Train memristor-based ANN-M model
python ANN/training-ANN-M.py
python ANN/finetune-ANN-M.py

# Step 5 - Generate confusion matrix / plots
python plot/CMX.py

