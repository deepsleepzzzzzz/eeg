# SEU 同源可比文献清单

> 筛选口径（老师定的）：**只有使用同一个数据库（东南大学 SEU / DDS 数据集）的论文才有可比性**；数据库不同的（CWRU、HUST、风电等）"完全不一样"，只能当背景。
> 整理日期：2026-10-01

---

## 〇、先对齐我们自己的设置（对比的基准）

| 维度 | 本文 |
|---|---|
| 数据库 | SEU DDS 齿轮箱（Southeast University Drivetrain Dynamics Simulator） |
| 采样率 | 5120 Hz |
| 任务 | **轴承 5 类** 与 **齿轮 5 类**（分开建模） |
| 工况 | 20 Hz–0 V（20-0）与 30 Hz–2 V（30-2），**两种工况各自独立建模** |
| 任务性质 | **同工况内分类**（非跨工况） |
| 划分 | 时间有序划分，测试集冻结后只评一次 |
| 指标 | 准确率（accuracy） |
| 结果 | 轴承 99.5% ~ 100%；齿轮 97.7% ~ 100%；5 折时间序列 CV 最低 96.9% |

> 判断"能不能比"的关键：**是否同一个数据库 + 是否同类任务（同工况内分类 vs 跨工况） + 是否同指标**。

---

## 一、直接可比（同一个 SEU 数据库 + 同工况内分类 + accuracy）

以下文献都用了 SEU 数据集、做的都是**同工况内的分类**、报的也都是**准确率**，属于最接近可比的。

| # | 文献 | 出处/年份 | 任务（SEU 部分） | 划分方式 | 报告条件 | 准确率 |
|---|---|---|---|---|---|---|
| 1 | 改进 Transformer（Guan 等） | ACM AAIA'23 | 齿轮 5 类 | 80/20 | SNR = 10 dB | **99.4%** |
| 2 | ICEEMDAN-MPE-AWT + SE-ResNeXt50（Gao 等） | Applied Sciences 2024 | 齿轮 5 类 | 随机抽样 | 先去噪，SNR −4~6 dB | 97.5% ~ 100% |
| 3 | FE-MCFormer | arXiv 2025 | 齿轮 5 类，2 工况 | 8:2（4000/1000） | 加噪 SNR −10~−2 dB | −2 dB 时 **100%**；−10 dB 仍 95.51% |
| 4 | MTF-TLSSA-DarkNet-GRU-MSA | Frontiers in Mech. Eng. 2026 | 齿轮 5 类，2 工况 | 每类 1000 训练 + 1000 测试 | 原始信号 | **99.17%** |
| 5 | 同步电机故障诊断（MSRC+CBAM） | IIETA | **轴承 + 齿轮**，G1/G2 两工况 | 论文未明确 | 原始信号 | 轴承 99.4~99.6%；齿轮 99.4~99.8% |
| 6 | LECA-EfficientNetV2 | Scientific Reports 2023 | **轴承 + 齿轮**（20-0） | 4:1 | 原始信号 | 轴承 99.38%；齿轮 99.75% |
| 7 | SCBM-Net | Scientific Reports 2025 | 轴承 **10 类**（两工况合并） | 70:30 | 原始信号 | 98.33%；平均 >99% |
| 8 | SAGAN + 改进 ResNet | Scientific Reports 2025 | 齿轮 5 类，2 工况 | 8:1:1（重叠采样） | 加噪 | SNR6 时 98.42%；SNR0 时 95% |
| 9 | 多尺度轻量 CNN（MGE-ResNet） | Processes 2025 | 轴承 **10 类** + CWRU | 每类 100 样本，训练:验证 3:1 | 原始信号 | >99.4% |
| 10 | SVMD 熵 + 机器学习 | Algorithms 2023 | 齿轮 5 类 | 论文未明确 | 原始信号 | 传统机器学习（相对较低） |
| 11 | 多模态深度融合（BiLSTM+CNN+Transformer） | ACM 2025 | SEU 轴承 / 齿轮 | 论文未明确 | 原始信号 | 轴承 99.96%；齿轮 ~99% |
| 12 | TL-RN-ELM（ResNet+ELM+迁移） | 中文期刊 | 齿轮（迁移学习） | 论文未明确 | 原始信号 | 平均 98.79% |

**观察**：真正"最贴近我们"的是 **#5（同步电机论文）和 #6（LECA-EfficientNetV2）** —— 它们同时做了**轴承 + 齿轮**两个任务，和我们一样；其余多数**只做齿轮或只做轴承**。

---

## 二、同 SEU 数据库，但任务不同（跨工况 / 跨机器，精度不能直接比）

这些也用了 SEU，但做的是**跨工况迁移**（一个工况训练、另一个工况测试）或**跨机器**，属于**更难的另一种任务**，只能作为"领域现状"背景，**不能拿它的分数和我们的同工况分类分数直接比**。

| 文献 | 出处 | 任务 | 准确率 |
|---|---|---|---|
| ADAMFFN | Machines 2025 | SEU 齿轮**跨工况** C1→C2 / C2→C1 | 平均 94.54% |
| WDATL | Wiley 2021 | SEU 轴承+齿轮 **0→1 迁移** | 平均 76.17%（CNN 基线仅 33.65%） |
| 无监督域适应（UDA） | Lubricants 2023 | SEU 轴承**跨工况** | 最高约 98.2%（BDA_KNN） |
| MSWCTD | Sensors 2025 | SEU 轴承 **3 类跨机器** | 100% / 97.67% |

> 意义：**跨工况是另一个难度级**（多数方法只有 50%~95%）。这正好佐证我们报告里"不具备跨工况泛化能力"的边界声明是符合领域现状的。

---

## 三、非 SEU 数据库（老师口径：完全不一样，仅作背景）

| 文献 | 数据库 | 结果 |
|---|---|---|
| MSCNN-LSTM-CBAM-SE | HUST 齿轮箱 | 97.22% ~ 99.85% |
| SFDA 跨工况迁移 | 风电齿轮箱 | 52.55% ~ 67.20% |
| 高阶矩 LHOM 特征 | IIT/PHM/XJTU 三个 benchmark | 约 81.67% ~ 92.60% |

---

## 四、结论：我们该从哪些角度找优势

1. **水平相当**：直接可比的同源文献集中在 **97% ~ 100%**，我们 **97.7% ~ 100%** 落在同一水平，不虚高也不落后。
2. **任务更全（最硬的角度）**：#1~#4、#7~#12 多数**只做齿轮或只做轴承**；只有 #5、#6 像我们一样**轴承 + 齿轮都做**。→ "别人没做的我们也做了"。
3. **评估更严格（方法论角度）**：多数文献用**随机划分 / 重叠采样 / 8:2 随机**，相邻窗口极易同时落进训练与测试，分数偏乐观；我们用**时间有序划分 + 时间序列交叉验证**，同分数含金量更高。
4. **表示更轻量（效率角度）**：我们用 6 状态的转移矩阵，对比 Transformer / ResNeXt / 多模态大网络，参数与计算量小得多——**这正是老师说的"快、实时性强、能耗低"的切入点**。
5. **诚实边界**：齿轮单项个别文献（#1 的 99.4%）略高于我们（97.7~98%），要如实承认，不硬凹；优势落在**轴承任务 + 评估严谨性 + 效率**上。

---

## 附：已下载归档的原文

存放在 `doc/references/SEU同源文献/`：

| 文件 | 对应表中编号 | 说明 |
|---|---|---|
| FE-MCFormer_arXiv2025_SEU齿轮箱_强噪声.pdf | #3 | 开放获取，已下载 |
| LECA-EfficientNetV2_SciRep2023_SEU轴承与齿轮.pdf | #6 | 开放获取，已下载（轴承+齿轮，最可比） |
| SCBM-Net_SciRep2025_SEU轴承10类_98.33.pdf | #7 | 开放获取，已下载 |
| SAGAN-IResNet_SciRep2025_SEU齿轮箱.pdf | #8 | 开放获取，已下载 |

> 其余文献（MDPI 的 Machines / Lubricants / Processes / Algorithms / Sensors，以及 Frontiers、ACM、IIETA）在本次网络环境下被反爬拦截未能自动下载，可按下方出处手动获取。

## 附：本次检索到的同源文献出处

1. Keming Guan, Bing Du, Zhe Wu, Jinfeng Li, Yan Zhang. *Fault Diagnosis of Gearbox Based on Improved Transformer.* AAIA 2023 (ACM). DOI: 10.1145/3603273.3636498
2. Hongfeng Gao, Tiexin Xu, Renlong Li, Chaozhi Cai. *Gearbox Fault Diagnosis Based on ICEEMDAN-MPE-AWT and SE-ResNeXt50 Transfer Learning Model.* Applied Sciences, 2024, 14(6): 2565.
3. *FE-MCFormer: An interpretable fault diagnosis framework for rotating machinery under strong noise.* arXiv:2505.06285.
4. *Design of a transfer learning fault identification model based on a Markov field and an improved DarkNet.* Frontiers in Mechanical Engineering, 2026.
5. *Deep Learning-Based Fault Diagnosis of Electromechanical Systems in Synchronous Motors.* IIETA.
6. *Gearbox fault diagnosis method based on lightweight channel attention mechanism and transfer learning.* Scientific Reports, 2023, 13: 22724.
7. *SCBM-Net: a multimodal feature fusion-based dual-channel method for bearing fault diagnosis.* Scientific Reports, 2025.
8. *Bearing fault diagnosis method based on SAGAN and improved ResNet.* Scientific Reports, 2025.
9. *Bearing Fault Diagnosis Based on Multiscale Lightweight Convolutional Neural Network.* Processes, 2025, 13(4): 1239.
10. Lijun Zhang 等. *Fault-Diagnosis Method for Rotating Machinery Based on SVMD Entropy and Machine Learning.* Algorithms, 2023, 16(6): 304.
11. *Research on Equipment Fault Diagnosis Method Based on Multimodal Deep Fusion.* ACM CIISAI 2025.
12. *基于 ResNet-ELM 和迁移学习的风机齿轮箱故障诊断方法.* 中文期刊.

---

## 补充检索（2026-10-01）：报告了"效率指标"的 SEU 同源文献

> 目的：导师把"**快 / 计算效率高 / 实时性强 / 能耗低**"列为最大切入点。以下文献**在 SEU 数据集上报告了参数量或 FLOPs**，是与本文做"效率同表对比"的直接素材。全部为 SEU 同库，纳入直接可比。

| # | 文献 | 出处/年份 | 任务（SEU） | 划分 | 交叉验证 | 结果 | 复杂度指标 |
|---|---|---|---|---|---|---|---|
| 13 | DSMC-ECA（多尺度深度可分离卷积 + ECA） | Sensors 2026, 26(4):1196 | 齿轮 5 类（20-0） | 60/20/20 | ✗ | −6 dB 时 86.84% | **0.204 M 参数 / 10.037 MFLOPs** |
| 14 | ADGCC-Net（物理引导灰度图 + ShuffleNetV2） | Processes 2025, 13(11):3600 | 轴承 5 类 | 0.3:0.2 | ✗ | **99.91%** | **0.0822 MB（≈82 KB）** |
| 15 | CS-LAMRNet（压缩感知 + 多尺度残差 + 轻量注意力） | Mathematics 2025, 13(9):1393 | 齿轮 5 类 | 每类 800/200 | ✗ | SNR≥12 dB 时 100% | 未报具体参数量 |
| 16 | 1D CNN-LSTM | Wiley 2025, je/1670810 | 轴承+齿轮 10 类 | 6600/70/250 | **✓ 10 折** | 报告完整 | 未报 |
| 17 | TL + 注意力轻量模型 | 2025 | 齿轮+轴承（跨转速） | 迁移 | — | 100% | 模型尺寸 −91.97%、FLOPs −61.86% |

**对照本文**：CNN **25,093 参数 / 0.384 MFLOPs / 104 KB**；GCN **18,549 参数 / 0.558 MFLOPs / 79 KB**。
- 对比 #13（DSMC-ECA）：参数少约 **8~10 倍**，算量少约 **20~26 倍**。
- 对比 #14（ADGCC-Net）：模型大小**同一量级**（79~104 KB vs 82 KB）——它是唯一在"体积"上与本文相当的轻量方法，但它**未做交叉验证**，且输入依赖灰度图转换。
- 对比 #9（MGE-ResNet，1.99 GFLOPs）：算量少约 **3 个数量级**。

> 出处登记：
> 13. Xiubin Liu, Wei Li, Hui Li, Yi Zhu, R. K. Agarwal. *Lightweight Gearbox Fault Diagnosis Under High Noise Based on Improved Multi-Scale Depthwise Separable Convolution and Efficient Channel Attention.* Sensors, 2026, 26(4): 1196. DOI: 10.3390/s26041196
> 14. Youlin Zhang, Shidong Li, Furong Li. *ADGCC-Net: A Lightweight Model for Rolling Bearing Fault Diagnosis.* Processes, 2025, 13(11): 3600. DOI: 10.3390/pr13113600
> 15. *Gearbox Fault Diagnosis Based on Compressed Sensing and Multi-Scale Residual Network with Lightweight Attention Mechanism.* Mathematics, 2025, 13(9): 1393. DOI: 10.3390/math13091393
> 16. *Fault Diagnosis of Rotating Machines Based on Combination of One-Dimensional Convolutional Neural Network and Long Short-Term Memory in Variable Working Conditions.* Wiley, 2025. DOI: 10.1155/je/1670810
> 17. *Intelligent Fault Diagnosis for Rotating Machinery via Transfer Learning and Attention Mechanisms: A Lightweight and Adaptive Approach.* 2025.
