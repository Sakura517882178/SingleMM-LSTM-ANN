# ANN-M: Y6 单分子忆阻器语音识别

## 项目目标

用 Y6 单分子结的实测 I-V 电导数据参数化忆阻器 ANN，实现语音识别，并通过光照/暗态对照实验证明 Y6 光电导效应是器件工作的关键。

## 文件结构

```
├── Y6-data.xlsx              # Y6 光态 50Hz 脉冲 I-V 数据 (实验组)
├── Y6-Dark.xlsx              # Y6 暗态 I-V 数据 (对照组)
├── training-ANN-M.py         # 单组训练脚本 (支持 --y6_data 指定数据文件)
├── finetune-ANN-M.py         # 微调脚本 (加载预训练权重，可选冻结非忆阻器层)
├── infer-ANN-M.py            # 推理脚本
├── compare-Y6-light-dark.py  # 一键对照实验: 同时训练 Light+Dark 并对比
├── CMX.py                    # 混淆矩阵计算
├── run_pipeline.sh           # 流水线脚本
└── job-training.sh           # 作业提交脚本
```

## 核心架构

```
Y6 I-V 数据 (.xlsx)
    │
    ▼
load_y6_params()          ← 自动检测光/暗态，提取 Gmin/Gmax/V
    │
    ▼
MemristorANN_LSTM_Attn    ← LSTM → Attention → Memristor FC → Memristor Nonlinear → FC
    │
    ▼
训练/推理 → 语音识别准确率
```

## 忆阻器模型 (log 域)

忆阻器电流: **I = 10^G × V**，其中 G 存储在 log(G/G₀) 域。

- `MemristorLinearI`: 线性层权重，G 初始化为 [Gmin, Gmax] 均匀分布
- `MemristorNonlinear`: 替代 ReLU，G 初始化为 [Gmin, Gmax] 均匀分布
- `update_memristor()`: 梯度下降更新 G，clip 到 [Gmin, Gmax]，支持 step 限制和噪声

## Y6 参数提取逻辑 (`load_y6_params`)

```
加载 .xlsx → 取稳定区 (t > 240ms, 跳过前12个50Hz forming周期)
    │
    ├─ tail < 5% (Light):  P5 → Gmin,  P99.9 → Gmax  (动态范围)
    └─ tail > 5% (Dark):   G > -5 平台均值 → G = 固定值 (Gmin=Gmax, 无塑性)
```

### 当前参数

| | Y6-Light | Y6-Dark |
|---|---|---|
| 模式 | 动态范围 | 固定值 |
| Gmin | -4.14 | -4.58 |
| Gmax | -2.91 | -4.58 |
| V | 0.35 V | 0.35 V |

## 模型架构

```
LSTM (单向, 128维) → tanh Attention → 池化
    → MemristorLinearI (128→128) → MemristorNonlinear → LayerNorm → Dropout
    → MemristorLinearI (128→36) → logits
```

- LSTM + Attention: 标准 PyTorch 模块 (非忆阻器)
- FC 层 + 非线性: 忆阻器模块，参数由 Y6 实验数据约束

## 训练流程

1. 从 .npz 加载语音特征 (64维, 36类)
2. 批次扫描 [16, 32, 64]，每个 batch size 独立训练
3. AdamW 优化器 + ReduceLROnPlateau + EarlyStopping (patience=30)
4. 两步更新: 先 `model.update_memristors()` 更新忆阻器权重，再 `optimizer.step()` 更新非忆阻器参数
5. 保存最佳模型到 `checkpoints-AM-Uni/`

## 对照实验

```bash
# 一次投递，同时训练两种模型
python compare-Y6-light-dark.py --epochs 200 --batch_sizes 16 32 64
```

输出对比表格:

```
         Batch    Light Acc     Dark Acc      Delta
            16       75.xx%        xx.xx%     xx.xx%
            32        xx.xx%        xx.xx%     xx.xx%
            64        xx.xx%        xx.xx%     xx.xx%
          Best       75.xx%        2.78%       7x.xx%
```

Dark 因 Gmin=Gmax（固定电导），忆阻器无塑性，准确率应接近随机水平 (~2.8%, 1/36)。

## 对审稿人的回复要点

1. **Gmin/Gmax/V 全部来自 Y6 实验数据** — 不再是硬编码
2. **log 域建模** — I = 10^G × V，G 在实测电导范围内
3. **对照组** — Y6-Dark 固定电导，无塑性，无法学习，证明光电导效应是关键
4. **稳定区提取** — 跳过 forming 阶段 (t<240ms)，P5/P99.9 去跳变
