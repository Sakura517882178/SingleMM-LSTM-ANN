# ANN-M: Y6 单分子忆阻器语音识别

## 项目目标

用 Y6 单分子结的实测 I-V 电导数据参数化忆阻器 ANN，实现语音识别，并通过光照/暗态对照实验证明 Y6 光电导效应是器件工作的关键。

## 文件结构

```
├── Y6-data.xlsx              # Y6 光态 50Hz 脉冲 I-V 数据 (实验组)
├── Y6-Dark.xlsx              # Y6 暗态 I-V 数据 (对照组)
├── compare-Y6-light-dark.py  # 一键对照实验: 同时训练 Light+Dark 并对比
├── infer_light_dark.py       # 推理脚本: 加载 Light/Dark 权重，保存预测 CSV
├── plot_cm.py                # 混淆矩阵画图: 读取预测 CSV，生成 Light/Dark 对比图
├── run_pipeline.sh           # 流水线脚本
└── job-training.sh           # 作业提交脚本
```

## 核心架构

```
Y6 I-V 数据 (.xlsx)
    │
    ▼
load_y6_params()          ← 自动检测光/暗态，提取 Imin/Imax (nA)
    │
    ▼
MemristorANN_LSTM_Attn    ← LSTM → Attention → Memristor FC → Memristor Nonlinear → FC
    │
    ▼
训练/推理 → 语音识别准确率
```

## 忆阻器模型 (nA 电流域)

忆阻器权重直接使用 Y6 实测电流 (nA)，不经过 log 域/电压转换。

- `MemristorLinearI`: 线性层，权重 I 初始化为 [Imin, Imax] 均匀分布 (nA)
- `MemristorNonlinear`: 替代 ReLU，权重 I 初始化为 [Imin, Imax] 均匀分布 (nA)
- `update_memristor()`: 梯度下降更新 I，clip 到 [Imin, Imax]，支持 step 限制和噪声

## Y6 参数提取逻辑 (`load_y6_params`)

```
加载 .xlsx → 取稳定区 (t > 240ms, 跳过前12个50Hz forming周期)
    │
    ├─ tail < 5% (Light):  P5 → Imin,  P99.9 → Imax  (动态范围)
    └─ tail > 5% (Dark):   滤除仪器掉坑后取均值 → I = 固定值 (Imin=Imax, 无塑性)
```

### 当前参数

| | Y6-Light | Y6-Dark |
|---|---|---|
| 模式 | 动态范围 | 固定值 |
| Imin | 0.41 nA | 0.29 nA |
| Imax | 32.26 nA | 0.29 nA |
| 光暗电流比 | — | 110× |

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
     Batch     Light Acc      Dark Acc       Delta
        16        58.95%         0.01%      58.94%
        32         3.83%         0.01%       3.82%
        64        71.51%         0.01%      71.50%
      Best        71.51%         0.01%      71.50%
```

Dark 因 Imin=Imax（固定电流），忆阻器无塑性，准确率接近 0%。
