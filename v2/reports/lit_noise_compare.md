# 抗噪实验：可对比文献清单（已回原文核对）

> 核对方式：DSMC-ECA 与 DDGF-Net 均回原文（PMC 全文 / OpenAlex 元数据 + 摘要）逐条核对，
> 不是转引二手数字。核对日期：2026-10-15。
> 本文基准：protocol v5，验证集，匹配加噪 AWGN，3 通道，窗长 2048 / 步长 1024。

---

## 一、DSMC-ECA（Sensors 2026, 26(4):1196）★ 目前最可比的一篇

- Liu X., Li W., Li H., Zhu Y., Agarwal R. K. *Lightweight Gearbox Fault Diagnosis Under
  High Noise Based on Improved Multi-Scale Depthwise Separable Convolution and Efficient
  Channel Attention.* Sensors, 2026, 26(4): 1196. DOI: 10.3390/s26041196
- 开放全文：PMC12944274（已下载核对 Table 5 / 6 / 7 / 8）

### 协议（原文第 4.2 节）

| 维度 | DSMC-ECA | 本文 v5 | 是否一致 |
|---|---|---|---|
| 数据集 | SEU | SEU | ✅ |
| 工况 | 20 Hz–0 V | 20-0 | ✅ |
| 噪声类型 | 加性高斯白噪声（匹配：噪声加在原始信号上，训练与验证同 SNR） | 加性高斯白噪声（匹配） | ✅ |
| SNR 覆盖 | −6 / −4 / −2 / 0 / 2 / 4 / 6 / 无噪 | −10 / −8 / −6 / −4 / −2 / 0 / 4 / 无噪 | 部分重叠 |
| 通道 | 行星齿轮箱 **x 轴单通道** | 行星齿轮箱 **x/y/z 三通道** | ❌ |
| 窗长 | 1024 点，**无重叠** | 2048 点，步长 1024（50% 重叠） | ❌ |
| 类别 | **10 类联合**（5 轴承 + 5 齿轮） | **5 类**（齿轮、轴承各自建模） | ❌ |
| 划分 | 60/20/20（每类 300/100/100，随机） | 时间有序 70/15/15 | ❌ |
| 报的数字 | 测试集 | 验证集 | ❌ |
| 复杂度 | 0.204 M 参数 / 10.037 MFLOPs | 见第三节 | — |

### 原文 Table 6（SEU 数据集，完整抄录）

| 方法 | −6 | −4 | −2 | 0 | 2 | 4 | 6 | 无噪 | MFLOPs | 参数(M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSMC-ECA（其法） | **86.84** | 93.97 | 97.50 | 98.96 | 99.66 | 99.68 | 99.80 | 99.98 | 10.037 | 0.204 |
| MK-ResCNN | 86.24 | 93.70 | 97.84 | 98.70 | 99.44 | 99.54 | 99.58 | 99.72 | 83.893 | 2.117 |
| ResNet18 | 80.16 | 91.74 | 96.88 | 98.68 | 99.38 | 99.58 | 99.66 | 99.70 | 175.920 | 3.854 |
| MobileNet | 76.04 | 90.96 | 96.12 | 98.66 | 99.20 | 99.36 | 99.56 | 99.56 | 333.620 | 3.186 |
| Liconvformer | 74.84 | 87.08 | 95.06 | 98.32 | 99.14 | 99.52 | 99.56 | 99.58 | 14.520 | 0.323 |
| CLFormer | 68.76 | 74.46 | 89.62 | 94.12 | 97.64 | 98.68 | 99.30 | 99.78 | 0.005 | 0.133 |
| MCSAT | 57.24 | 70.24 | 80.14 | 88.86 | 94.58 | 97.00 | 96.06 | 97.40 | 8.497 | 0.044 |
| MobileNet-V2 | 54.50 | 81.34 | 91.58 | 97.84 | 99.04 | 99.52 | 99.52 | 99.64 | 96.955 | 2.192 |

**原文 Table 7（−6 dB，10 类平均）**：DSMC-ECA 平均召回 86.12 / 平均 F1 86.18。
**原文 Table 8（−6 dB 消融）**：完整模型 86.84；−SMDC 76.92；−SMC 84.04；−ECA 85.40。

---

## 二、DDGF-Net（Machines 2026, 14(9):1013）★ 已下全文逐条核对

- Ma R., Wang X., Cheng J., Xie T., Li S., Wang C. *DDGF-Net: A Novel Dual-Domain
  Generative-Discriminative Fusion Network for Gearbox Fault Diagnosis Under Strong Noise
  Conditions.* Machines, 2026, 14(9): 1013. DOI: 10.3390/machines14091013（2026-09-05）
- 全文核对来源：MDPI 附件直链 `mdpi-res.com/d_attachment/machines/machines-14-01013/article_deploy/machines-14-01013.pdf`
  （正文 30 页，已本地解析成文本后逐节核对；以下数字均注明表号/页位）

### 协议（原文 4.1 节 + Table 2 + Table 3）

| 维度 | DDGF-Net | 本文 v5 |
|---|---|---|
| 数据集 / 工况 | SEU 齿轮箱 5 类 / 20 Hz–0 V | SEU 齿轮 / 20-0 |
| 样本 | 每类 2000 段（1600 训练 / 400 测试） | 每类 715 训练 / 152 验证 |
| 窗 | 1024 点，**无重叠滑窗** | 2048 点，步长 1024（50% 重叠） |
| 划分 | **随机 8:2 分层 + 5 折分层 CV（seed 42）**，报 5 折均值 | **时间有序 70/15/15** |
| 加噪 | **训练时就加**（每 batch 随机抽 SNR ∈ [−10,10] dB 在线增强），"训练与测试的 SNR 保持一致的分布" | 训练与验证同 SNR（匹配） |
| 报的数字 | **测试集**，5 折均值 ± 标准差 | **验证集** |
| 显著性 | 5 折配对 t 检验（−10/−6/0 dB 打了 ***/**） | 多种子 + 统计检验 |
| 参数量 | **15.29 M（61.2 MB FP32）**，其中 CVAE 去噪器 8.49 M（**占 55.5%**） | CNN 0.0277 M / GCN 0.0549 M |

> 原文 Table 2 注：*"Partitioning is performed at the segment level (1024-point
> non-overlapping segments) **because each condition contains a single continuous recording**."*
> —— 作者自己承认每类只有一段连续录音，却仍按**段级随机**划分训练/测试。

### 原文 Table 4（SEU，5 类，20-0，匹配加噪，5 折均值 ± std；只有这 5 个 SNR 点）

| 方法 | −10 dB | −6 dB | 0 dB | 6 dB | 10 dB | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| MLP（8 手工特征） | 23.9±3.9 | 21.3±3.4 | 25.7±4.3 | 23.0±1.5 | 23.6±3.0 | 0.003 M |
| WDCNN | 45.4±0.8 | 64.8±1.0 | 90.4±0.8 | 98.1±0.5 | 98.9±0.2 | 0.048 M |
| ResNet1D | 56.3±1.0 | 82.7±1.1 | 99.3±0.2 | 100.0±0.0 | 100.0±0.0 | 3.85 M |
| BiLSTM-Attn | 36.9±1.1 | 53.9±2.2 | 81.1±4.3 | 92.8±5.8 | 94.7±5.6 | 0.159 M |
| DRSN | 56.5±1.0 | 83.1±0.9 | 99.3±0.1 | 100.0±0.0 | 100.0±0.0 | — |
| STFT2D | 46.5±0.6 | 69.5±1.3 | 94.7±0.4 | 99.8±0.1 | 99.9±0.1 | — |
| ViT-1D | 36.1±0.7 | 52.0±1.3 | 75.1±1.0 | 84.9±0.9 | 88.0±0.7 | — |
| DConformer | 42.2±0.5 | 61.6±1.1 | 87.6±2.0 | 97.1±0.7 | 98.2±0.7 | — |
| **DDGF-Net** | **71.2±0.5\*\*\*** | **93.5±0.8\*\*\*** | 99.9±0.1\*\* | 100.0±0.0 | 100.0±0.0 | **15.29 M** |
| **本文（齿轮 20-0）** | **84.39** | **86.10** | 95.64 | — | — | **0.0277 M** |

> 原文自注：*"Significance tests are not reported for SNR = 6 dB and 10 dB, where multiple
> methods reached the 100% accuracy ceiling."* —— 高 SNR 段任务已饱和，无区分度。

### −8 dB = 84.3 的出处（引用必须写清）

**84.3 不在 Table 4 里**。它出自原文 4.2 节正文 / 图 10 鲁棒性曲线：
*"DDGF-Net achieves 84.3% accuracy at SNR = −8 dB, while other approaches require
substantially cleaner signal conditions to reach comparable performance."*
同段还给出：DRSN / ResNet 要到 −4 dB 才到 92.5% / 92.4%，WDCNN / STFT2D 要到 −2 dB 才 84.6% / 88.8%。
→ 引用时写"正文（图 10）"，不能写"Table 4"。

### 它的方法为什么强：不是"表示更好"，是多挂了一个大去噪网络

原文 3.1 节：CVAE 以带噪信号为输入、**以干净信号为监督目标**训练
（"This supervised denoising formulation is enabled by the paired noisy-clean samples
constructed via AWGN addition"），做的是"带噪 → 干净"的信号重建，再送进 DRSN 主干。
它 15.29 M 参数里 **8.49 M（55.5%）就是这个去噪器**。

## 三、本文 v5 位置（齿轮 20-0，CNN/GCN 均值，验证集）

| SNR | 本文 | DSMC-ECA 系（10 类） | DDGF-Net 系（5 类） | 判断 |
|---|---:|---|---|---|
| −6 | **86.10** | 其法 86.84；**7 个基线 54.50–86.24** | 其法 93.5；DRSN 基线 83.1 | 高于 DDGF-Net 的通用基线，接近 DSMC-ECA 全部 |
| −8 | **84.98** | — | 84.3 | 略优 |
| −10 | **84.39** | — | 71.2 | 明显优 |
| −4 | 90.42 | 93.97（其法）/ 93.70（MK-ResCNN） | 92.4–92.5 | 落后 2–3 点 |
| 0 | 95.64 | 98.96（其法） | ~100 | 落后 |

**要点**：SEU 20-0 已发表的 −6 dB 数字横跨 **54.5% ~ 93.5%**；
"本文弱"的印象来自只拿 DDGF-Net 的单点（93.5）当参照。

---

## 四、正确写法

1. 可直接引用的：DSMC-ECA Table 6（8 行，含复杂度）、DDGF-Net 正文数值。
2. 必须写明的口径差异：类别数（10 vs 5）、通道数（1 vs 3）、窗长与重叠（1024 无重叠 vs 2048/1024）、
   划分（随机 60/20/20 或 8:2 vs 时间有序 70/15/15）、评估集（测试集 vs 验证集）。
3. 结论句只能写"量级相当 / 落在同一区间"，**不能写"优于"** —— 除非按对方设定复现。

---

## 五、要让"量级对照"变成"同协议对比"

文献共同设定 = **SEU 20-0 + 1024 点无重叠窗 + 匹配 AWGN**（DSMC-ECA 与 DDGF-Net 都是）。
按这个设定跑一版本文方法（协议变体 `compat_seu_1024_v1`），即可产出能直接进对方表格的数字：

```yaml
window: {length: 1024, stride: 1024, overlap: 0}
split: {kind: random_stratified, train: 0.8, test: 0.2, seed: 42}
noise: {type: awgn, matched: true, snr: [-6, -4, -2, 0]}
```

---

## 附录：本次核对发现的数据错误（必须改）

| 项 | 表里现值 | v5 正确值 | 说明 |
|---|---|---|---|
| CNN 参数量 | 25,093 | **27,685** | 25,093 是 v2（输入 21 通道 = 3 轴 × 7 滞后）的值；v5 是 30 通道 |
| GCN 参数量 | 18,549 | **54,941** | 18,549 属更早的 GCN 实现，v5（relational, steps=10, hidden=32）为 54,941 |
| FLOPs | 0.384 / 0.558 M | **待重算** | 来源不明，需用统一工具（hook 计数或 ptflops）重算后才能进对比表 |

复算命令（参数量，精确）：

```bash
python -c "import sys;sys.path.insert(0,'src');from config.protocol import load_protocol,build_train_omega;from models.cnn import build_model;p=load_protocol('configs/protocols/main_70_15_15_v5.yaml');cfg=build_train_omega(p,'cnn');m=build_model(5,cfg,in_channels=30,image_size=6);print(sum(x.numel() for x in m.parameters()))"
```

---

## 六、别人"实验看起来更好"的四个可核查原因（一手证据）

| # | 原因 | 证据（原文） | 我们的对应做法 |
|---|---|---|---|
| 1 | **段级随机划分 = 相邻窗泄漏** | DDGF-Net Table 2 注："each condition contains a single continuous recording"，却仍按段级随机 8:2；DSMC-ECA 只写"non-overlapping sliding window"，未声明时间不相交 | 时间有序 70/15/15，训练段与测试段在时间上完全分离 |
| 2 | **匹配加噪（训练就加噪）** | DDGF-Net："noisy samples with randomly selected SNR within [−10,10] dB were dynamically generated for each training batch"；DSMC-ECA："noise is added to the original vibration signals" | 同样匹配加噪（每档 SNR 单独训练一个模型） |
| 3 | **高 SNR 段已饱和** | DDGF-Net 自注："multiple methods reached the 100% accuracy ceiling"（6/10 dB）；DSMC-ECA 干净侧 99.98% | 我们干净侧也是 99.7，同样饱和 |
| 4 | **方法自带大去噪网络** | DDGF-Net 15.29 M 参数中 8.49 M 是 CVAE 去噪器（55.5%），用"带噪→干净"配对监督训练 | 无前置去噪、无额外模块（前置滤波已实测为负收益） |

### 拉平口径后的真实差距（−6 dB，SEU 20-0，匹配 AWGN）

| 层级 | 方法 | −6 dB | 参数量 |
|---|---|---:|---:|
| 通用基线（DDGF-Net 自跑） | WDCNN / BiLSTM / ViT-1D / DConformer / STFT2D / ResNet1D / DRSN | 52.0 – 83.1 | 0.05 – 3.85 M |
| 轻量新方法（DSMC-ECA 自跑，**10 类**） | MK-ResCNN / ResNet18 / MobileNet ... | 54.5 – 86.24 | 0.13 – 3.85 M |
| 强噪声专法 | DSMC-ECA（10 类） | 86.84 | 0.204 M |
| 强噪声专法 + 监督去噪 | DDGF-Net | 93.5 | 15.29 M |
| **本文** | 符号状态转移 + CNN/GCN（5 类，验证集） | **86.10** | **0.0277 M** |

**读法**：−6 dB 处本文高于 DDGF-Net 论文里**全部 8 个基线**（最高 DRSN 83.1），
低于两个"自带去噪模块的强噪声专法"；而在 −10 dB 处本文 84.39 高于 DDGF-Net 的 71.2 达 13.2 点。
