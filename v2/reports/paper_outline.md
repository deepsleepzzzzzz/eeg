# 论文正文骨架（v1，2026-10-15）

> 配套文件：`positioning_v2.md`（主张与边界）、`table_*.md`（各表）、`fig_*.png`（各图）
> 所有数字都可追溯到 `artifacts/` 下的原始产物。

---

## 标题（候选）

1. *Ultra-Lightweight Gearbox and Bearing Fault Diagnosis via Multi-Lag Symbolic State-Transition Tensors*
2. *Rethinking Noise-Robustness Benchmarks: A 27-kParameter Symbolic Transition Network for Fault Diagnosis*

**摘要只写两句主张**（见 positioning_v2.md 第一节），不放抗噪优势。

---

## 1. Introduction

**段 1** 故障诊断现状：深度模型（CNN / RNN / Transformer）在足够数据下有效，但部署代价高。

**段 2** 抗噪路线三条：判别式（DRSN 软阈值）、生成式（CVAE / 扩散）、多域融合。
**共同代价**：额外模块与大容量——DDGF-Net 15.29 M 参数，其中 CVAE 降噪器占 8.49 M（55.5%）。

**段 3** 表示层路线：RP / GAF / GADF / MTF 把 1D 信号转成 2D 图像。
**但 MTF 的图尺寸是 `N×N`，随窗长增长**（2048 点 → 2048²）。

**段 4（本文）** 提出**多滞后联合状态转移概率张量**：`[30, 6, 6] = 1080` 个元素，**与窗长无关**。
四条贡献（见 `positioning_v2.md` 第三节）。

⚠️ **本段必须把与 MTF 的区别写清**（否则整篇被否）——见 `positioning_v2.md` 第七节。

---

## 2. Method

### 2.1 符号化
6 档等频分位，边界**只用训练集**拟合（`fit_on_train_only`）。
给出为什么用等频而非等宽：等宽在非高斯振动幅值下会退化成 1~2 档（项目曾踩此坑）。

### 2.2 多滞后联合转移概率张量

对通道 `c`、滞后 `k`：

```
P_k^(c)(i, j) = P(s_t = i, s_{t+k} = j)
```

堆叠成 `[C × K, B, B] = [30, 6, 6]`。

**与 MTF 的差异**（表格，见 positioning_v2.md 第七节）。

### 2.3 分类器
CNN：两层 3×3 卷积（32/64）+ BN + 2×2 池化 + GAP，27,685 参数。
GCN：把 6 个状态视为节点、滞后视为关系类型（relational），两层，隐层 32，54,941 参数。

**强调**：两者都是极小容量。§5 用 LDA 复核证明判别力不来自分类器。

---

## 3. 评估协议

| 项 | 设置 |
|---|---|
| 划分 | **时间有序** 70/15/15，训练/验证/测试区间互不重叠 |
| 测试集 | **封存，全流程只评估一次** |
| 窗 | 2048 点 / 步长 1024（50% 重叠），窗不跨段 |
| 归一化 | 训练集固定尺度 z-score |
| 加噪 | AWGN，训练与验证**同 SNR**（匹配），加在切窗后、符号化前 |
| **噪声参考功率口径** | 逐窗（与文献同口径）**与** 全局（严格口径）**两种都报** ← 本文主张之一 |
| 指标 | accuracy / macro F1 / balanced accuracy |
| 重复 | 主实验 5 个 seed；抗噪 3 个 seed；报均值 ± 标准差 |

⚠️ 必须写明：**每类仅含一次采集记录** → "同源记录内评估"。

---

## 4. Results

### 4.1 主结果（`table2_main_v5.md`）
4 工况 × 2 模型 × 5 seed；干净侧 99.61 ~ 100.00%。

### 4.2 与文献同口径对比（`table_lit_same_protocol.md`）
逐窗口径下齿轮 20-0：−10 dB 本文 84.39 vs DDGF-Net 71.2。

### 4.3 同协议自跑基线（`table_baseline_per_window.md`）
**这一节回答"同级水平"**——所有基线由本项目管线自跑，排除口径混淆。

### 4.4 效率（`table_efficiency.md`）
CNN 27,685 参数 / 954,496 FLOPs；DSMC-ECA 204,000 / 10,037,000（**1/7.4 与 1/10.5**）；
DDGF-Net 15,290,000（**1/552**）。

### 4.5 抗噪性能（`table_noise_protocol.md` + `fig_noise_protocol.png`）
**两种口径并排报**，见 §5.3。

---

## 5. 机理分析（本文最独特的一章）

### 5.1 判别力归因：LDA 复核（`lda_check.csv`）
同一份缓存，LDA（线性，1080 维）与 CNN/GCN 相差 **0 ~ 4.4 点**（两种口径、4 工况）。
→ **判别力主要来自表示，而不是分类器容量**——这解释了为什么 2.8 万参数够用。

### 5.2 信息在转移结构而非幅值
转移矩阵逐行归一化后准确率几乎不变（≤1.5 点）。

### 5.3 噪声参考功率口径的效应（**本文独有发现**）
- 逐窗参考使加噪窗能量恒等于干净窗能量的常数倍；
- **零学习的纯能量分类器**在逐窗口径下齿轮 20-0 −10 dB 达 84.0%，与本文模型 84.4% 相同；
- 换全局参考后同一模型降至 70.5%（**−13.9 点**），−16 dB 降 40.9 点；轴承 20-0 −10 dB 降 64.7 点；
- 该口径是文献通用写法

**写法三条**（见 positioning_v2.md）：不做指控 / 用数据说话 / 放 Discussion 不放摘要。

### 5.4 符号翻转率（`table_symbol_flip.md`）
翻转率**只由 SNR 与量化档数决定，与任务无关**（4 工况在每档上最大差 0.7%）。

---

## 6. 局限（5 条，一条都不能少）

见 positioning_v2.md 第五节。

---

## 7. 结论

一句话：**以 2.8 万参数、无前置降噪的符号状态转移表示，在中高信噪比达到主流方法同级水平，
并指出抗噪文献普遍采用的噪声参考口径存在非物理捷径。**

---

## 图表清单

| 编号 | 文件 | 内容 |
|---|---|---|
| 表 1 | `table2_main_v5.md` | 主结果（acc / macro F1 / balanced acc） |
| 表 2 | `table_lit_same_protocol.md` | 与文献同口径对比 |
| 表 3 | `table_baseline_per_window.md` / `_global.md` | 同协议自跑基线 |
| 表 4 | `table_efficiency.md` | 参数 / FLOPs / 推理时间 |
| 表 5 | `table_noise_protocol.md` | 两种噪声口径对照 |
| 表 6 | `table_symbol_flip.md` | 符号翻转率 |
| 表 7 | `lda_check.csv` | LDA 复核 |
| 图 1 | `fig_transition.png` | 各类平均转移矩阵 |
| 图 2 | `fig_noise_protocol.png` | 双口径对照（含纯能量线） |
| 图 3 | `fig_confusion.png` | 混淆矩阵 |
| 图 4 | `fig_efficiency_scatter.png` | 参数量–精度 |
