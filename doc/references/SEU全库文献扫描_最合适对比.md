# SEU 数据集文献全库扫描 · 最合适的对比条目

> 检索日期：2026-10-01
> 检索范围：MDPI / IEEE Xplore / Wiley / Springer / IOP / Nature / ACM / 中文核心期刊
> 筛选口径：**必须使用 SEU（东南大学 DDS 驱动系统动态模拟器）数据集**；优先"单工况 5 分类"（与本文任务一致）
> 本文基准（resplit 密封测试，seed 42）：**轴承 99.85~100%｜齿轮 97.69~99.85%**
> 标记说明：**【自跑】** = 对照模型由原文作者在相同设置下自行训练评测（Type A，导师要的）；**【转引】** = 引用他人论文数值

---

## 一、结论先行：最合适的是这 8 篇

按"同库 + 同任务（单工况 5 分类）+ 划分明确 + 有基线数值"四项筛完，**最值得用的 8 篇**：

| 优先级 | 文献 | 为什么最合适 | 可用数值 |
|---|---|---|---|
| ★★★ | **DDGF-Net**（Machines 2026） | **唯一做了 5 折交叉验证**的同任务文献；8 个基线全部自跑 | 见 §2.1 |
| ★★★ | **DEFT-CNN**（IEEE TIM 2021） | 仪器仪表顶刊；两工况**分列报数**；KNN/SVM 仅 88% | 见 §2.2 |
| ★★★ | **AMMMP 2024** | 20-0 单工况、800/200、结构最贴合 | 见 §2.3 |
| ★★ | **CBAM-ResNeXt50**（Sensors 2023） | 5 类两工况；AlexNet 仅 93% | 见 §2.4 |
| ★★ | **CWT+CNN-DOA-LSSVM**（Information 2026） | 20-0、5 类、7:3，划分交代清楚 | 见 §2.5 |
| ★★ | **CS-LAMRNet**（Mathematics 2025） | VGG11 90.58%，且**报了方差** | 见 §2.6 |
| ★★ | **ConvFormer-SENet**（IEEE 2025） | 20-0、5 类、1023 样本/类、7:3 | 见 §2.7 |
| ★★ | **Dual-Channel 多模态融合**（Machines 2025） | **轴承侧** 5 类单工况，自跑基线 + 报标准差 | 见 §3.1 |

另外三篇"低分但需加条件"的（SVMD 熵、IAMCNN、EBRB-EU）见 §4；只作量级参照的见 §5。

---

## 二、齿轮箱 · 单工况 5 分类（主战场）

### 2.1 DDGF-Net ★★★ —— 唯一有交叉验证的

**Ma R., Wang X., Cheng J., Xie T., Li S., Wang C. *DDGF-Net: A Novel Dual-Domain Generative-Discriminative Fusion Network for Gearbox Fault Diagnosis Under Strong Noise Conditions.* Machines, 2026, 14(9): 1013.**
DOI: 10.3390/machines14091013

- **任务**：SEU 齿轮箱 5 类（正常 / 齿面崩裂 / 断齿 / 齿根裂纹 / 齿面磨损），**恒定工况 20 Hz-0 V**，采样 5120 Hz
- **设置**：样本长 1024 点、无重叠滑窗；**8:2 分层划分（seed 42）+ 5 折分层交叉验证**；加性高斯白噪声，训练与测试同 SNR

| 方法 | −10 dB | −8 dB | −6 dB | −4 dB | 0 dB | 6 dB | 10 dB |
|---|---|---|---|---|---|---|---|
| MLP（8 手工特征） | ~21–26（全区间） | — | — | — | — | — | — |
| ViT-1D | **36.1** | — | — | — | — | 84.9 | 88.0 |
| BiLSTM-Attention | **36.9** | — | — | — | — | — | — |
| ResNet-1D | **56.3** | — | — | 92.4 | — | — | — |
| DRSN | **56.5** | — | 83.1 | 92.5 | — | — | — |
| WDCNN / STFT-2D | — | WDCNN 需 −2 dB 才达 84.6% | — | — | — | — | — |
| DConformer | — | — | — | — | — | — | — |
| **DDGF-Net（本文）** | **71.2** | **84.3** | **93.5** | — | ~100（CI 99.8–100.0） | — | — |

> 原文明确："All baseline models were evaluated under identical conditions: the same data split (seed 42, 8:2 train/test, five-fold stratified cross-validation), the same SNR levels, and the same accuracy metric." —— **这是 A 类自跑基线，且带 5 折 CV，正对导师要求。**

⚠️ **注明**：原文 Table 4 未公开完整数值矩阵，上表中 WDCNN / STFT-2D / DConformer 在部分 SNR 下无明确数字；引用时只写正文明确给出的那些。

### 2.2 DEFT-CNN ★★★ —— 顶刊，两工况分列

**Zhu C., Chen Z., Zhao R., Wang J., Yan R. *Decoupled Feature-Temporal CNN: Explaining Deep Learning-Based Machine Health Monitoring.* IEEE Transactions on Instrumentation and Measurement, 2021, 70: 1–13.**
DOI: 10.1109/TIM.2021.3084310

- **任务**：SEU 齿轮箱 5 类；**20 Hz-0 V 与 30 Hz-2 V 两个工况分别评估**
- **设置**：采样 1024 Hz；512 s 窗、原始长 5120 切 20 个 256 点子窗 → 每样本 20×9 特征矩阵；**随机 80/20 划分**；X/Y/Z 三向振动

| 方法 | 20 Hz-0 V | 30 Hz-2 V |
|---|---|---|
| KNN | **88.37%** | **88.24%** |
| SVM | **88.49%** | **87.61%** |
| LFGRU | 94.23% | 95.89% |
| Sparse Filtering | 98.56% | 98.29% |
| DEFT-CNN | 98.40% | 98.12% |
| DEFT-CNN + 空间注意力 | 99.61% | 99.41% |

**【自跑】**（原文"we compared our method with benchmark methods…"），无 CV。**这是目前档次最高、最干净的一条**——IEEE TIM 顶刊，KNN/SVM 只有 88%，与本文齿轮 97.69~99.85% 差 9~11 个点。

**✅ 2026-10-01 回原文（Table V）复核：KNN 88.37 / 88.24、SVM 88.49 / 87.61 无误。**

⚠️ **引用时只能列 KNN / SVM 两行，绝不能把该文的 DEFT-CNN 那两行也列进去**——它自己的方法在 30 Hz-2 V 上是 **99.41%**（DEFT-CNN 98.12%、+空间注意力 99.41%），**高于本文齿轮 30-2 的 98.15 / 97.69%**；20 Hz-0 V 上 99.61% 与本文 99.54 / 99.85% 基本持平。列进去等于自曝短板。

补充事实（原文核实）：
- **输入模态与本文相近**：原始 5120 点切 20 个 256 点子窗，每子窗提 **9 个手工特征**（时域 rms / p2p / skew；频域谱峭度 / 谱偏度 / 谱功率；时频 db1 / db2 / db3 小波能量）→ 每样本 **20×9 特征矩阵**。与本文 21×6×6 符号转移矩阵同属"压缩表示"，**比跟原始信号方法比更具可比性**。
- **齿轮箱样本总数原文未报告**（只报了 Paderborn 轴承的 13 860），故"同数据量"一项**无法核对**。
- 原文写采样率 1024 Hz、512 s 窗（SEU DDS 常见记载为 5.12 kHz，疑为原文笔误）；随机 8:2 划分的表述在"Implementation Details"中统一给出，字面主语写的是 bearing，引用时宜写"随机划分"而不写死比例。
- 类别数：正文"randomly selected **five sets of data from five different classes**"+ 图 11–15 五张显著图 → **5 类**（Normal / Chipped / Root / Miss / Surface）。注：原文网络描述里另有一句"four output nodes"，与 5 类表述不一致，属原文自身瑕疵。

### 2.3 AMMMP 2024 ★★★ —— 结构最贴合

**《Fault Diagnosis of Gearbox Based on CNN and LSTM with Attention Mechanism》. Highlights in Science, Engineering and Technology, 2024, 93: 291–292.**

- **任务**：SEU 齿轮箱、**仅 20 Hz-0 V 单工况**、5 类（NC / Chipped / Miss / Root / Surface）
- **设置**：每类 800 训练 / 200 测试；**随机 8:2**；30 epoch；Adam；10 次重复

| 方法 | 准确率 |
|---|---|
| LSTM | **90.75%** |
| CNN | **92.90%** |
| CNN-LSTM | **94.82%** |
| CNN-AM（对照模型） | 96.28% |
| **该文提出的方法** | **98.5%** |
| **本文（SEU 齿轮）** | **97.69~99.85%** |

**✅ 2026-10-01 回原文（Table 3）复核：LSTM 90.75 / CNN 92.9 / CNN-LSTM 94.82 / CNN-AM 96.28 / 该文方法 98.5，五项无误。**
**更正**：此前本文档把 **CNN-AM 96.28% 误记为该文自己提出的方法**。实际 CNN-AM 与 LSTM / CNN / CNN-LSTM 同属**对照模型**，该文自己提出的方法报 **98.5%**（Table 3 第五行）。

原文核实的设置：SEU DDS 齿轮箱、**仅 20 Hz-0 V 单工况**、5 类（NC / Chipped / Miss / Root / Surface，每类 1024 点）、**每类 800 训练 + 200 测试（合计 4000 / 1000）**、30 epoch、网格搜索定参。

⚠️ **三条必须写明的条件**：
1. **无交叉验证**——原文"Ten experiments are conducted simultaneously for each of the four models and the average of the experimental results is taken"，是**10 次重复取均值**，**不是 10 折 CV**（旧文档曾把两者混写，引用前务必统一口径）；
2. 该文**自己报的最好结果是 98.5%**，高于本文齿轮下限 97.69%。→ **只能写"优于其列出的基线"，不能写"优于该文献"。**
3. 出处是 Darcy & Roy Press 的 *Highlights in Science, Engineering and Technology*（AMMMP 2024, Vol.93），**层级偏低、非主流索引**，别当主要论据。

### 2.4 CBAM-ResNeXt50 ★★

**Wang Z., Tang Y., Dong Y., et al. *Optimization of Gearbox Fault Detection Method Based on Deep Residual Neural Network Algorithm.* Sensors, 2023, 23(17): 7573.**
DOI: 10.3390/s23177573

- **任务**：SEU 齿轮箱 5 类，两个工况分别实验；数据来源 Gitee（zhengkun110/Mechanical-datasets）
- **设置**：混淆矩阵评价；训练时间约 22.6 / 23.1 s；本文方法测试准确率 **100% / 99.875%**

| 方法 | 平均精度（工况1 / 工况2） | 平均召回（工况1 / 工况2） |
|---|---|---|
| CBAM-ResNeXt50（本文） | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| DenseNet121 | 0.9741 / 0.9778 | 0.9738 / 0.9776 |
| ResNeXt50 | 0.9747 / 0.9864 | 0.9738 / 0.9863 |
| ResNet50 | 0.9699 / 0.9655 | 0.9687 / 0.9650 |
| **AlexNet** | **0.9322 / 0.9152** | **0.9325 / 0.9113** |

**【自跑】**。⚠️ 原文报的是 **precision / recall**，不是 accuracy，引用时要写对指标名。

### 2.5 CWT + CNN-DOA-LSSVM ★★

**《Fault Classification Method for Rotating Machinery Based on Hybrid Model of CWT and CNN-DOA-LSSVM》. Information, 2026, 17(6): 580.** DOI: 10.3390/info17060580

- **任务**：SEU 齿轮箱、**20 Hz-0 V**、行星齿轮箱 **Z 向**、5 类（health / chipped / missing / root / surface）
- **设置**：丢弃前 1000 点；窗长 2048、步长 1000；**每类 300 样本 = 1500 总**；**7:3 划分**；CWT 转 64×64

| 方法 | 准确率 |
|---|---|
| CNN | **98.22%** |
| CNN-DOA-LSSVM（本文） | 99.78% |

**【自跑】**，无 CV。差距小，必须写清设置。

### 2.6 CS-LAMRNet ★★ —— 报方差

**《Gearbox Fault Diagnosis Based on Compressed Sensing and Multi-Scale Residual Network with Lightweight Attention Mechanism》. Mathematics, 2025, 13(9): 1393.** DOI: 10.3390/math13091393

- **任务**：SEU 齿轮箱 5 类（CT 点蚀 / MT 齿根 / RF 断齿 / SF 磨损 / NO 正常），第一工况（1200 r/min, 0 N·m）x 轴
- **设置**：1000 点滑窗 → GADF 224×224 图；每类 800 训练 / 200 测试；**8:2 随机**

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

**【自跑】**，无 CV 但**报了方差**（说明多次重复），是规范性较好的一条。

### 2.7 ConvFormer-SENet ★★

**《Hierarchical Convolution-Transformer Framework for Gear Fault Diagnosis Under Severe Noise》. IEEE Xplore, 2025, 文献号 11037744.**

- **任务**：SEU 齿轮箱 5 类（healthy / root crack / missing tooth / broken tooth / surface wear），**行星齿轮箱 x+y、1200 r/min、0 V**
- **设置**：窗长 1024、滑窗；**每类 1023 样本**；**7:3 划分**；150 epoch；本文 100% 准确率

| 方法 | 准确率 |
|---|---|
| ResNet50 | **96.8%** |
| CNN-CBAM | **97%** |
| UniFormer | **98%** |
| ConvFormer-SENet（本文） | 100% |

**【自跑】**。差距小（1~3 个点），必须写清设置。

### 2.8 其余同任务条目（可用，条件较多）

| 文献 | 出处 | 设置 | 可用数值 | 备注 |
|---|---|---|---|---|
| **WPD-attention-LSTM** | Wiley, DOI 10.1155/vib/8680245 | SEU 齿轮 5 类 · 5240 样本 · 7:3 · 重复 5 次 | **ELM 83.9 / 1DCNN 83.6 / SVM 89.7** | ⚠️ 1DCNN 后带引用号 [33]，**可能转引**，引用前须核 |
| **MSCNN-ViT** | IEEE AiDAS 2025, DOI 10.1109/AiDAS67696.2025.11213541 | SEU 齿轮 5 类 · 每类 300 · 本文 99.83% | SVM ≈78.51 / 2D-CNN ≈89.71 / ResNet-50 ≈92.30 / DenseNet-121 ≈93.13 / ViT-Base ≈94.22 | ⚠️ **由"提升量"反推，非原表数值**，须核 |
| **MSSLWF** | Springer LNEE 2025, DOI 10.1007/978-981-96-5527-4_28 | SEU 齿轮 5 类 · 两工况 · 8:2 · 8000/2000 · 5 次 | X 向 99.92 / Y 向 99.11 / 非自学习融合 99.75 | 本文 99.97，差距极小 |
| **GSWOA-KELM** | Lubricants, 2024, 12(1): 10 | SEU 齿轮 20-0 · 5 类 · 每类 100（2048 点）· 7:3 | **时域特征 86.67 / 频域特征 85.33**（融合后 100） | 单域特征的低分，可作"特征工程对比" |
| **RTSMRDE + SSA-SVM** | Machines, 2023, 11(6): 646 | SEU 齿轮 20-0 · Y 向 · 5 类 × 50 子样本 · 10 训 / 40 测 | SSA-SVM 100%（样本量仅 250） | 样本量太小，只能作参照 |
| **Swin Transformer + MTF** | Eng. Res. Express, 2025, 7: 015225. DOI 10.1088/2631-8695/ada71f | DDS 齿轮 | 本文 **99.69%**；优于 CWT / GAF / CNN / ViT | ⚠️ 对照数值待补 |
| **轻量化 CNN 系统** | IEEE 2025, 文献号 11508850 | SEU 齿轮 | 99.92% | 会议短文，无对比表 |
| **DSMC-ECA** | Sensors, 2026, 26(4): 1196 | SEU 齿轮 · 强噪声 · 0.204 M 参数 | −6 dB 下 **86.84%** | 轻量化方向，与本文效率节呼应 |
| **He et al. 迁移学习** | Nondestructive Testing and Evaluation, 2025. DOI 10.1080/10589759.2025.2495802 | SEU 变工况齿轮箱 | 平均 **96.12%** | 变工况任务，难度不同级 |

---

## 三、轴承 · 单工况 5 分类

### 3.1 Dual-Channel 并行多模态融合 ★★

**Li W., Cai H., Yang X., Xue Y., Ye J., Hu X. *Dual-Channel Parallel Multimodal Feature Fusion for Bearing Fault Diagnosis.* Machines, 2025, 13(10): 950.**
DOI: 10.3390/machines13100950

- **任务**：SEU **轴承**、**30 Hz-2 V（1800 rpm, 7.32 N·m）单工况**、5 类（ball / compound / health / inner / outer）
- **设置**：采样 5120 Hz；样本长 1024、重叠 512；**每类 300 = 1500 总**；**6:2:2 划分**；AWGN；每点 5 次实验取均值

| 方法 | −6 dB | −4 dB | 0 dB | 4 dB | 6 dB | 无噪声 |
|---|---|---|---|---|---|---|
| MLP（BP） | **59.67** | 61.67 | **70.00** | 76.00 | 77.33 | — |
| LSTM | 74.33 | 75.00 | 78.33 | 82.47 | 83.93 | — |
| CNN | 72.67 | 73.27 | 78.13 | 81.47 | 82.73 | — |
| GRU | 74.93 | 76.93 | 82.07 | 83.27 | 84.47 | — |
| **本文** | **85.87** | 86.73 | **91.67** | 93.27 | 94.67 | 97.33 |

**【自跑】**，无 CV，**报了标准差**。这是轴承侧最干净的一条：单工况 5 类、划分明确、基线自跑。

### 3.2 ADSRN（小样本）

**Jiang Y., Lu M., Dong Z., et al. *Adaptive Deeping Siamese Residual Network: A Novel Model for Few-Shot Bearing Fault Diagnosis.* Machines, 2025, 13(3): 193.**
DOI: 10.3390/machines13030193

- **任务**：SEU 轴承 5 类（Ball / Comb / Inner / Outer / Normal），x+y 双通道，**每类训练样本 10 / 15 / 20 / 30** 的小样本设置
- **数值（30-shot）**：ResNet **75.50±2.32**、WDCNN **81.06±4.55**、CNN-LSTM 96.35±2.41、ADSRN 98.85±0.23

⚠️ **只能用于小样本场景对比**，与本文全样本设置不同级。

### 3.3 其余轴承条目

| 文献 | 出处 | 任务 | 可用数值 |
|---|---|---|---|
| **CAM 交叉注意力** | Electronics, 2025, 14(5): 886 | SEU 轴承 **10 类**（20_0 与 30_2 混合） | CNN **90.76** / MCNN-LSTM 98.36 / BAN 99.49 / CAM 99.98 |
| **MGE-ResNet** | Processes, 2025, 13(4): 1239 | SEU 轴承 **10 类** | SVM **81.46±4.28** / TICNN **88.50±8.14** / MGE-ResNet 99.44 |
| **CNN-Informer-DA** | Meas. Sci. Technol., 2025, 37(2). DOI 10.1088/1361-6501/ae2d7e | SEU 轴承 | **98.44%** |
| **MSWCTD** | Sensors, 2025, 25(10): 3141 | SEU 轴承 **3 类**跨机器迁移 | 100% / 97.67% |
| **模态融合深度聚类** | 电子与信息学报, 2025, 47(1): 244 | SEU 齿轮 / 轴承（无监督） | 齿轮 **99.16%** / 轴承 **98.63%** |

---

## 四、低分但必须加条件的三条

| 文献 | 可用低分 | 必须写明 |
|---|---|---|
| **SVMD 熵**（Algorithms, 2023, 16(6): 304） | 齿轮 **89.41~96.02%**；轴承 30-2 **94.12~97.25%** | 原文明确"**无训练集，全部数据用于测试**"——样本内拟合值，非泛化准确率 |
| **IAMCNN**（华东理工大学学报, 2024, 50(6): 920-928） | **BP 82.03 / CNN 85.46 / LSTM 90.29 / ResNet 95.35** | 只能取**表 1**（表 2 是行星齿轮箱）；无 CV |
| **EBRB-EU**（Sci. Rep., 2026. DOI 10.1038/s41598-026-44629-8） | **IBRB 82.0 / BRF 81.7 / ABRB 82.6 / CSL 83.2 / CWSVM 84.0 / HBRB 84.1 / BPNN 86.2**（本文 92.8） | **跨工况（20→30）+ 类别不平衡**，任务难度与单工况 5 类不同级 |

**EBRB-EU 是全库定位最接近"80 多分"的一篇**——但它换了工况也换了采样不平衡设置，只能作"更困难场景下仍有方法落在 82~86%"的旁证。

---

## 五、同库 · 任务不同（只能作量级参照）

Kibrete 1D CNN-LSTM（Wiley J. Eng. 2025, DOI 10.1155/je/1670810，变工况组合集 10 类，CNN 94.12 / LSTM 94.90 / 1D CNN 97.67）、ACM 多模态融合 2025（ACM CIISAI，SEU 轴承 CNN 89.50 / Transformer 63.56；SEU 齿轮 CNN 97.27 / Transformer 71.08）、ADAMFFN（Machines 2026，齿轮跨工况 94.54%）、SAGAN-IResNet（Sci. Rep. 2025，8:1:1 划分 + CWT 时频图）、IIETA MSRC+CBAM（60%/30% 划分）。

---

## 六、必须撤下 / 慎用

- **FE-MCFormer（arXiv:2505.06285）**：**v2 版含 SEU 齿轮箱案例**（5 类两工况、5000 样本、4000/1000），**v3 当前版已整段删除**（改为 PU 轴承 + 离心压缩机）。它那张强噪声表（MSCNN-LSTM 28.16 / WDCNN 66.16 / ResNet50 78.54）**确实存在过但已被作者删除**。→ **弃用；若一定要用，必须写明"引自 v2"**。
- 各种 CSDN / 公众号"博客"里的东南大学数据集结果（如 WSET-CNN-BKA-LSSVM、GADF-CNN-SSA-XGBoost）：**非正式发表文献，不能引**。

---

## 七、出处登记（完整）

1. Ma R., et al. DDGF-Net. *Machines*, 2026, 14(9): 1013. DOI 10.3390/machines14091013
2. Zhu C., et al. Decoupled Feature-Temporal CNN. *IEEE Trans. Instrum. Meas.*, 2021, 70: 1–13. DOI 10.1109/TIM.2021.3084310
3. Fault Diagnosis of Gearbox Based on CNN and LSTM with Attention Mechanism. *Highlights in Science, Engineering and Technology*, 2024, 93: 291–292
4. Wang Z., et al. Optimization of Gearbox Fault Detection Method Based on Deep Residual Neural Network Algorithm. *Sensors*, 2023, 23(17): 7573. DOI 10.3390/s23177573
5. Fault Classification Method for Rotating Machinery Based on Hybrid Model of CWT and CNN-DOA-LSSVM. *Information*, 2026, 17(6): 580. DOI 10.3390/info17060580
6. Gearbox Fault Diagnosis Based on Compressed Sensing and Multi-Scale Residual Network with Lightweight Attention Mechanism. *Mathematics*, 2025, 13(9): 1393. DOI 10.3390/math13091393
7. Hierarchical Convolution-Transformer Framework for Gear Fault Diagnosis Under Severe Noise. *IEEE Xplore*, 2025, 文献号 11037744
8. Li W., et al. Dual-Channel Parallel Multimodal Feature Fusion for Bearing Fault Diagnosis. *Machines*, 2025, 13(10): 950. DOI 10.3390/machines13100950
9. Jiang Y., et al. Adaptive Deeping Siamese Residual Network. *Machines*, 2025, 13(3): 193. DOI 10.3390/machines13030193
10. Wu N., et al. An Investigation on Attention Mechanism–Based LSTM for Gearbox Fault Diagnosis. Wiley. DOI 10.1155/vib/8680245
11. Duan L., et al. Gearbox Fault Diagnosis Method Based On MSCNN-ViT. *IEEE AiDAS 2025*. DOI 10.1109/AiDAS67696.2025.11213541
12. Gearbox Fault Diagnosis Method Based on Model Transfer and Multi-sensor Self-learning Weighted Fusion. *Springer LNEE*, 2025. DOI 10.1007/978-981-96-5527-4_28
13. Time-Frequency Fusion Features-Based GSWOA-KELM Model for Gear Fault Diagnosis. *Lubricants*, 2024, 12(1): 10. DOI 10.3390/lubricants12010010
14. Gearbox Fault Diagnosis Based on Refined Time-Shift Multiscale Reverse Dispersion Entropy and Optimised SVM. *Machines*, 2023, 11(6): 646. DOI 10.3390/machines11060646
15. Liu J., et al. A gearbox fault diagnosis method based on Swin Transformer and Markov transform fields. *Eng. Res. Express*, 2025, 7: 015225. DOI 10.1088/2631-8695/ada71f
16. Lightweight Gearbox Fault Diagnosis Under High Noise…（DSMC-ECA）. *Sensors*, 2026, 26(4): 1196. DOI 10.3390/s26041196
17. He X., et al. A deep-transfer-learning fault diagnosis method for gearboxes. *Nondestructive Testing and Evaluation*, 2025. DOI 10.1080/10589759.2025.2495802
18. A Novel Framework Based on Complementary Views for Fault Diagnosis with Cross-Attention Mechanisms. *Electronics*, 2025, 14(5): 886. DOI 10.3390/electronics14050886
19. Bearing Fault Diagnosis Based on Multiscale Lightweight Convolutional Neural Network. *Processes*, 2025, 13(4): 1239. DOI 10.3390/pr13041239
20. Wang Y., et al. Rolling bearing health status diagnosis method based on bidirectional feature extraction by fusing CNN and Informer. *Meas. Sci. Technol.*, 2025, 37(2). DOI 10.1088/1361-6501/ae2d7e
21. A Novel Multistep Wavelet Convolutional Transfer Diagnostic Framework for Cross-Machine Bearing Fault Diagnosis. *Sensors*, 2025, 25(10): 3141. DOI 10.3390/s25103141
22. 伍章俊, 等. 一种面向旋转机械多传感器故障诊断的模态融合深度聚类方法. 电子与信息学报, 2025, 47(1): 244. DOI 10.11999/JEIT240648
23. Lijun Zhang, et al. Fault-Diagnosis Method for Rotating Machinery Based on SVMD Entropy and Machine Learning. *Algorithms*, 2023, 16(6): 304. DOI 10.3390/a16060304
24. 邵浙梁, 戚知宽, 周邵萍. 基于改进注意力机制的 CNN 的齿轮箱故障诊断. 华东理工大学学报（自然科学版）, 2024, 50(6): 920-928. DOI 10.14135/j.cnki.1006-3080.20231207002
25. A bearing fault diagnosis method for complex system based on improved extended belief rule base（EBRB-EU）. *Sci. Rep.*, 2026. DOI 10.1038/s41598-026-44629-8
26. Kibrete F., et al. Fault Diagnosis of Rotating Machines Based on Combination of 1D CNN and LSTM in Variable Working Conditions. *J. Engineering*, 2025. DOI 10.1155/je/1670810
27. Research on Equipment Fault Diagnosis Method Based on Multimodal Deep Fusion. *ACM CIISAI 2025*. DOI 10.1145/3773365.3773555
