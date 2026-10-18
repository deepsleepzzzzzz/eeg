# SEU 同源文献新增检索结果

> 检索日期：2026-10-01
> 检索范围：SEU（东南大学 DDS）齿轮箱 / 轴承数据集，重点为**单工况 5 分类**任务
> 所有数值均回原文（PDF / 出版社页面）核对过，来源见表末
> 本文基准（resplit 密封测试，seed 42）：**轴承 99.85~100%｜齿轮 97.69~99.85%｜总 97.69~100%**

---

## 〇、一条必须先纠正的事（关于 FE-MCFormer）

上一篇文档里我写"FE-MCFormer 不是 SEU、须撤下"。**这个结论只对当前版本成立，需要修正**：

| 版本 | 时间 | 实验案例 | 是否含 SEU |
|---|---|---|---|
| v1 | 2025-05-07 | — | — |
| **v2** | 2025-12-11 | Case 1 = PU 轴承；**Case 2 = SEU 齿轮箱**；Case 3 = 离心压缩机 | **含 SEU** |
| **v3（当前版）** | 2026-07-21 | Case 1 = 滚动轴承（PU）；Case 2 = 离心压缩机 | **不含 SEU** |

- v2 原文：*"The Southeast University gearbox dataset (SEU) is employed in this case study… 5 fault types in two operating conditions (20 Hz–0 V and 30 Hz–2 V)… total number of samples is 5000, of which 4000 samples are selected for model training, and the rest is used for model testing."*
- v3（本地 `_lit_txt/FEMCFormer.txt` 就是这个版本）已删除 SEU，改为轴承 + 压缩机；全文 "Southeast" 出现 0 次。

**结论**：v2 那张 SEU 强噪声对照表（MSCNN-LSTM 28.16% / WDCNN 66.16% / ResNet50 78.54% / DenseNet 74.07% / MA1DCNN 62.29% / Li-convformer 75.02%，SNR = −10 dB）**确实存在**，但**已被作者自己在 v3 中删除**。

**处置建议**：不要放进主表。若一定要用，必须写明版本——"引自 arXiv:2505.06285**v2**（该案例在 v3 中已被作者移除）"。风险高，导师一查当前版本会找不到。**建议降级或直接弃用。**

---

## 一、A 级：SEU 齿轮箱 · 单工况 5 分类 · 有基线数值（可直接进主表，注明设置）

> 这一档全部满足"同库 + 同任务（5 分类）+ 同工况可分"，是本轮检索的主要收获。

### 1. DEFT-CNN（IEEE TIM 2021）★ 最推荐

- Zhu C., Chen Z., Zhao R., Wang J., Yan R. *Decoupled Feature-Temporal CNN: Explaining Deep Learning-Based Machine Health Monitoring.* **IEEE Transactions on Instrumentation and Measurement**, 2021, 70: 1–13. DOI: 10.1109/TIM.2021.3084310
- 任务：SEU 齿轮箱，5 类（Chipped / Miss / Root / Surface / Normal），**20 Hz-0 V 与 30 Hz-2 V 两个工况分别评估**
- 设置：1024 Hz 采样，512 s 窗，原始序列长 5120 切 20 个 256 点子窗 → 每样本 20×9 特征矩阵；**随机 80/20 划分**；X/Y/Z 三向振动

| 方法 | 20 Hz-0 V | 30 Hz-2 V |
|---|---|---|
| KNN | 88.37% | 88.24% |
| **SVM** | **88.49%** | **87.61%** |
| Sparse Filtering | 98.56% | 98.29% |
| LFGRU | 94.23% | 95.89% |
| DEFT-CNN | 98.40% | 98.12% |
| DEFT-CNN + 空间注意力 | 99.61% | 99.41% |

- **价值**：IEEE TIM 是仪器仪表领域顶刊，**同任务、单工况、分列报数**，且 KNN/SVM 只有 88% 档。比之前所有条目都干净。
- ⚠️ 注明：随机 80/20 划分，无交叉验证。

### 2. CS-LAMRNet（Mathematics 2025, MDPI）

- *Gearbox Fault Diagnosis Based on Compressed Sensing and Multi-Scale Residual Network with Lightweight Attention Mechanism.* **Mathematics**, 2025, 13(9): 1393.
- 任务：SEU 齿轮箱，5 类（CT 点蚀 / MT 齿根 / RF 断齿 / SF 磨损 / NO 正常），取第一工况（1200 r/min, 0 N·m）x 轴信号
- 设置：1000 点滑窗，GADF 转 224×224 图，每类 **800 训练 / 200 测试**（共 5000 图），8:2 随机划分

| 方法 | 平均准确率 | 方差 |
|---|---|---|
| **VGG11** | **90.58%** | 0.8563 |
| MobileNet V3 | 96.50% | 0.1464 |
| ResNet18 | 96.60% | 0.1326 |
| GADF-CNN | 97.70% | 0.0976 |
| MTF-ResNet | 97.80% | 0.0876 |
| ConvNeXt-T | 99.10% | 0.0064 |
| DRSN-CW | 99.60% | 0.0045 |
| CS-LAMRNet（本文） | 100% | 0 |

- **价值**：VGG11 90.58%、MobileNet V3 96.50% 是很好的低分基线；且**报了方差**（多次重复）。
- ⚠️ 注明：8:2 随机划分，无交叉验证。

### 3. WPD-attention-LSTM（Wiley）

- Wu N., Li Y., Li X., Yuan K., Jiang W. *An Investigation on Attention Mechanism–Based Long Short-Term Memory for Gearbox Fault Diagnosis.* Wiley, DOI: 10.1155/vib/8680245
- 任务：SEU 齿轮箱，5 类（缺齿 / 齿根裂纹 / 表面点蚀 / 正常），多工况
- 设置：**5240 样本**，每样本 1000 点，**随机 7:3 划分**，每实验**重复 5 次**

| 方法 | SEU 平均准确率 | 5 次明细 |
|---|---|---|
| **ELM** | **83.9%** | 78.3 / 83.0 / 83.9 / 91.1 / 83.2 |
| **1DCNN** | **83.6%** | 86.1 / 86.5 / 86.0 / 77.1 / 82.4 |
| **SVM** | **89.7%** | 89.7 / 90.1 / 89.3 / 90.1 / 89.2 |
| WPD-attention-LSTM（本文） | 98.2% | 98.3 / 98.1 / 98.5 / 98.1 / 98.0 |

- **价值**：这是**目前找到的最低档之一**（ELM 83.9 / 1DCNN 83.6）。且与本文方法并排列出 5 次重复，形式上是同设置自跑。
- ⚠️ **风险**：原文中 1DCNN 后带引用号 **[33]**，暗示可能转引；SVM/ELM 未标引用号但也不敢确定是自跑。**引用前必须核实 4.5 节与文献 [33]**，否则不能写成"同设置对比"。

### 4. MSCNN-ViT（IEEE AiDAS 2025）⚠️ 数值待核

- Duan L., Gaol F.L., Arifin Y., Soeparno H., ChunaKiat O. *Gearbox Fault Diagnosis Method Based On MSCNN-ViT.* 2025 6th Int. Conf. on Artificial Intelligence and Data Sciences (AiDAS). DOI: 10.1109/AiDAS67696.2025.11213541
- 任务：SEU 齿轮箱，5 类，**每类 300 样本**；本文方法 99.83%
- 原文报"相对提升"：SVM +21.32%、2D-CNN +10.12%、ResNet-50 +7.53%、DenseNet-121 +6.70%、ViT-Base +5.61%

| 方法 | 反推准确率（= 99.83 − 提升量） |
|---|---|
| **SVM** | **≈ 78.51%** |
| 2D-CNN | ≈ 89.71% |
| ResNet-50 | ≈ 92.30% |
| DenseNet-121 | ≈ 93.13% |
| ViT-Base | ≈ 94.22% |

- ⚠️ **上表是由"提升量"反推的，不是原文直接给的数值**，而且反推精度取决于原文表述。**引用前必须拿到原文表格核实**（IEEE 付费墙）。核实前不要写进正式文档。
- 另注意：该文摘要末尾把数据集写成 "Southeast University **bearing** dataset"，与正文的 gearbox 不一致，说明写作粗糙。

### 5. Kibrete 1D CNN-LSTM（Wiley J. Engineering 2025）

- Kibrete F., Woldemichael D.E., Gebremedhen H.S. *Fault Diagnosis of Rotating Machines Based on Combination of One-Dimensional Convolutional Neural Network and Long Short-Term Memory in Variable Working Conditions.* **Journal of Engineering** (Wiley), 2025. DOI: 10.1155/je/1670810
- 任务：SEU 齿轮箱（**注意：核心评估用的是含复合故障的 A/B/C/D 组合集，即 10 类变工况**）
- 设置：每样本 **2048 点**；6600 训练 / 70 验证 / 250 测试；**10 折交叉验证仅用于网格搜索调参**，最终用独立测试集报数

| 方法 | SEU 齿轮箱 | CWRU 轴承 |
|---|---|---|
| CNN | 94.12% | 95.12% |
| LSTM | 94.90% | 96.35% |
| 1D CNN | 97.67% | 98.91% |
| 1D CNN-LSTM（本文） | 99.15% | 99.57% |

- **价值**：原文明确"所有对比模型在同一条件下训练与测试，以保证公平评价"——**这是 A 类自跑基线**。
- ⚠️ **两个坑**：① Table 3 报的 SEU 齿轮箱任务，按正文描述更可能是**含复合故障的组合集**，类别数与你的单工况 5 类不同，**必须回原文确认**；② 10 折 CV 是调参手段，**不是性能评估**，不能写"该文做了交叉验证"。

### 6. HAWAN-PIR（MDPI Automation 2025）

- *Hierarchical Adaptive Wavelet-Guided Adversarial Network with Physics-Informed Regularization…* **Automation**, 2025, 6(2): 14.
- 任务：SEU 齿轮箱，Dataset A（20 Hz-0 V）/ Dataset B（30 Hz-2 V）/ A+B；1-D CNN，10 次试验

| 数据集 | 平均准确率 | 标准差 |
|---|---|---|
| A（20 Hz-0 V） | 98.67% | 0.27% |
| B（30 Hz-2 V） | 97.34% | 0.18% |
| A/B 组合 | 96.73% | 2.25% |

- **价值**：**单工况分列 + 报标准差 + 10 次重复**，设置规范。但注意它主题是类不平衡数据生成。

### 7. 其他（SEU 齿轮箱，数值待补）

| 文献 | 出处 | 已知信息 |
|---|---|---|
| DRSN-BiLSTM 风电齿轮箱 | ICoPESA 2025 | SEU 齿轮箱 5 类准确率 **99.5%**；比 DRSN / BiLSTM / DCNN 高 4.8~12.5% |
| 小样本轻量化 MCNN+ViT | 制造业自动化 2025, 47(7) | SEU 齿轮箱，多尺度 CNN + ViT，数值待补 |
| 决策融合 + 迁移学习 | 济南大学学报（自然科学版）2025, 39(3): 379-388 | CWRU → SEU 齿轮/轴承，准确率 **100%** |
| 模态融合深度聚类 | 电子与信息学报 (JEIT) 2025, 47(1): 244 | SEU 齿轮 **99.16%**、轴承 **98.63%**（无监督聚类） |

---

## 二、B 级：SEU 同库但任务不同（只能当量级参照）

| 文献 | 出处 | 任务 | 可用数值 |
|---|---|---|---|
| **MGE-ResNet** | Processes 2025, 13(4): 1239 | SEU 轴承 **10 类**（两工况合并） | SVM **81.46±4.28%**；TICNN **88.50±8.14%**；SE-ResNet152 95.65；Improved AlexNet 97.94；MGE-ResNet 99.44 |
| **ADAMFFN** | Machines 2026, 14(9): 960 | SEU 齿轮**跨工况**（C1→C2 / C2→C1），5 类 | 平均 **94.54%**；最佳单支 87.93%；最佳双支 91.45%；模型 6.69 MB、+4 ms 延迟 |
| **ACM 多模态深度融合** | CIISAI 2025, DOI 10.1145/3773365.3773555 | SEU 轴承 / 齿轮 | 轴承：BiLSTM 99.04 / CNN **89.50** / Transformer **63.56**、本文 99.96；齿轮：BiLSTM 98.95 / CNN 97.27 / Transformer **71.08**、本文 99.47 |
| **1D-CNN-LSTM 工业轴承** | ICIEAM 2025, DOI 10.1109/ICIEAM65163.2025.11028381 | SEU 轴承，变工况 | 摘要未给具体数值 |
| **迁移学习 + 注意力** | CORE 收录 | SEU 齿轮 + 轴承，跨转速迁移 | 100%（迁移任务，与你的任务难度不同级） |

---

## 三、C 级：任务完全对不上（不建议引）

Kibrete 的变工况组合集（10 类）、SCBM-Net（轴承 10 类）、SAGAN（少样本）、IIETA MSRC+CBAM（60/30 划分）、HUST / 风电 / CWRU 等跨库条目。

---

## 四、出处登记

1. Zhu C., Chen Z., Zhao R., Wang J., Yan R. Decoupled Feature-Temporal CNN: Explaining Deep Learning-Based Machine Health Monitoring. *IEEE Trans. Instrum. Meas.*, 2021, 70: 1–13. DOI 10.1109/TIM.2021.3084310
2. Gearbox Fault Diagnosis Based on Compressed Sensing and Multi-Scale Residual Network with Lightweight Attention Mechanism. *Mathematics*, 2025, 13(9): 1393.
3. Wu N., Li Y., Li X., Yuan K., Jiang W. An Investigation on Attention Mechanism–Based Long Short-Term Memory for Gearbox Fault Diagnosis. Wiley. DOI 10.1155/vib/8680245
4. Duan L., et al. Gearbox Fault Diagnosis Method Based On MSCNN-ViT. *2025 6th Int. Conf. on AI and Data Sciences (AiDAS)*. DOI 10.1109/AiDAS67696.2025.11213541
5. Kibrete F., Woldemichael D.E., Gebremedhen H.S. Fault Diagnosis of Rotating Machines Based on Combination of 1D CNN and LSTM in Variable Working Conditions. *Journal of Engineering*, 2025. DOI 10.1155/je/1670810
6. Hierarchical Adaptive Wavelet-Guided Adversarial Network with Physics-Informed Regularization… *Automation*, 2025, 6(2): 14.
7. Bearing Fault Diagnosis Based on Multiscale Lightweight Convolutional Neural Network. *Processes*, 2025, 13(4): 1239.
8. Adaptive Domain-Aligned Multi-Modal Feature Fusion Network for Cross-Speed Fault Diagnosis of Planetary Gearboxes. *Machines*, 2026, 14(9): 960.
9. Research on Equipment Fault Diagnosis Method Based on Multimodal Deep Fusion. *ACM CIISAI 2025*. DOI 10.1145/3773365.3773555
10. Muratbakeev E., Novak D., Kozhubaev Y. Investigation of Industrial Bearing Fault Diagnosis Based on 1D-CNN-LSTM. *ICIEAM 2025*: 1059–1067. DOI 10.1109/ICIEAM65163.2025.11028381
11. 【版本敏感】Yuan Y., Jiang X., et al. FE-MCFormer. arXiv:2505.06285 — **v2 含 SEU 齿轮箱案例，v3 已移除**（v3 改为 PU 轴承 + 离心压缩机）
12. 刘婷婷, 等. 基于决策融合方法和迁移学习的齿轮箱故障诊断. 济南大学学报（自然科学版）, 2025, 39(3): 379-388.
13. 伍章俊, 许仁礼, 方刚, 邵海东. 一种面向旋转机械多传感器故障诊断的模态融合深度聚类方法. 电子与信息学报, 2025, 47(1): 244. DOI 10.11999/JEIT240648
14. 小样本条件下轻量化齿轮箱特征提取与故障诊断方法. 制造业自动化, 2025, 47(7).

---

## 五、下一步建议

**要回原文核实的三件事（未核前不要写进正式文档）：**

1. **WPD-attention-LSTM 的 SVM / ELM / 1DCNN 是自跑还是转引**——看 4.5 节与文献 [33]。这决定它能不能算 A 类。
2. **Kibrete 的 Table 3 是 5 类单工况还是 10 类组合集**。
3. **MSCNN-ViT 的 SVM 78.51% 反推值**——需原文表核实。

**要处理的矛盾：**

- FE-MCFormer：从"必须撤下"改为"版本敏感、建议弃用"，并在旧文档中同步改正。
