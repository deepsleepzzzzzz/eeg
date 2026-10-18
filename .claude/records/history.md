# Changelog

## 2026-09-22

- 去掉交接压缩包。给老师看 `doc/handoff/seu_stage_report.docx`，图在 `doc/figures/seu/`。

## 2026-09-21

- 给导师的 Word 按概要、流程、细节与引用重写。见 `doc/handoff/seu_stage_report.docx`
- 四条线（CNN/GCN × 关/开注意力）在验证集扫学习率和 dropout。见 `doc/method.md`
- 验证集混淆矩阵和 t-SNE 接到 SEU 训练，论文图在 `doc/figures/seu/`。见 `doc/results.md`
- `feat/rotating-fault` 去掉脑电分类入口，只留旋转机械。脑电在标签 `eeg-cnn`、`eeg-gcn`。

- CNN 与 GCN 合进同一工程，用配置切换。脑电冻在标签 `eeg-cnn`、`eeg-gcn`。

## 2026-09-20

- GCN 种子 42 的 val 已记。43、44 还在跑，test 未开。见 `doc/results.md`
- GCN 不加图注意力和加图注意力的 test 已评，没有重训。见 `doc/results.md`
- CNN 在 CUDA 上优化参数后，不加注意力和自注意力的 test 已评。见 `doc/results.md`
- 训练设备 `cuda` 为可选项。见 `doc/config.md`

## 2026-09-19

- 新方案在验证集定下后都评一次 test。见 `doc/method.md`
- 后续 CNN、GCN 分两条路。注意力是开关，不和骨干叠成第三条路。见 `doc/method.md`
- CNN 自注意力开关已评 test。见 `doc/results.md`
- 统一节点编号已评 test。见 `doc/results.md`

- 6 节点按导堆叠已评 test，用已有权重，没有重训。见 `doc/results.md`

## 2026-09-18

- 正式三集 Transition：14 人 × 3 种子，test 封闭。val acc 均值 0.5033（CI 0.4857–0.5210）。见 `doc/results.md`
- 并入队友交接包 `eeg-master_handoff_with_results_20260918_231758.zip`：RP 三集实验代码、冻结配置、14 人独立测试结果。未覆盖本地状态转移文件。最终 RP test acc 0.8952（CI 0.8731–0.9173），见 `doc/results.md`
- 时间步消融：`[1]` 0.430，`[1,2]` 0.426，`[1,2,3]` 0.419。多时间步未涨点。见 `doc/results.md`

## 2026-08-31

- 表示逻辑对齐论文：AMI+FNN 选 m/τ、各导 z-score、ε 三种规则、节律滤波真正接入；MRP 多导改为 joint
- 锁定 m=3、τ=12（`embed_select=fixed`）。正式 1500 @512 ms：z-score+joint 下 τ=12 为 R 0.749 / M 0.552；同设置 τ=1 为 R 0.718 / M 0.475。工程默认（τ=1、无 z-score、M 每导一张）R 0.803 / M 0.734 仅对照。**z-score 保持开**（论文1），不关回去冲分；开了之后 ε/训练可能要另调。见 `doc/results.md`
- 去掉旋转机械故障诊断残余：删除 `config/{default,smoke,b_r,m_1k,m_1w}.yaml`、`data/dataset.py`；step1 不再接受 `source=rotating`
- 无参入口改为 `config/eeg_ws_r.yaml`

## 2026-08-30

- 初始化 EEG 工程：`.claude/`、配置驱动方法变体、阶段 R 优先（先 R 后 M）
- 论文摘录移至 `.claude/reference/papers_rp_eeg.md`
- 约定用旋转机械 raw_datas 做工程验证
- 已写：`config/default.yaml`、`data/`、`representation/phase_space.py` + `recurrence_plot.py`
- **暂停**：R 通路其余代码（models/scripts/跑通）未做；详见 `status.md`

### 正式测试：每类 1000

- `num_windows_per_class=1000`；`main_pipeline.py config/default.yaml`
- best_val_acc≈0.5674；checkpoint：`datas/checkpoints/best_rp_cnn.pt`

### 正式测试：每类 10000

- `num_windows_per_class=10000`；best_val_acc≈0.7859；early stop @16
- checkpoint 覆盖写入 `datas/checkpoints/best_rp_cnn.pt`

### 阶段 M：MRP + additive

- 实现 soft MRP + AdditiveAttention；配置 `m_1k.yaml` / `m_1w.yaml`
- 修复 raw `(N,1)` 被误判多通道的问题
- M-1k best_val_acc≈0.7461；M-1w best_val_acc≈0.8849

### 方案 B：时间切分 + 不重叠窗

- 重写 `data/dataset.py`、step1/step2；配置 `config/b_r.yaml`
- R 试跑：best_val≈0.798，**test_acc≈0.777**（train/val/test=36868/7898/7898）

### 切换到 ds002680 EEG（放弃振动默认）

- 数据迁入 `datas/ds002680/`；mne+pandas；`data/eeg_ds002680.py` 4 类 epoch
- 链路冒烟 R/M 通；全 run（2 被试）R test≈0.318，M test≈0.328

### 2-Fold LOSO（2026-08-31）

- `data.protocol=loso2`：按被试缓存，内部 8:2 早停，换人测试后平均
- 配置：`eeg_loso2_smoke_r.yaml` / `eeg_loso2_r.yaml`
- 正式 R：Acc_003=0.197，Acc_002=0.221，mean=0.209
- **决定**：放弃多人跨被试，聚焦个体内划分；报告见 `doc/results.md`
- 重定目标：数据 ds002680 单被试；方法 RP/MRP+CNN；目标个体 test 高于随机。见 `doc/method.md`
- 个体内 `data.subject` 可指定批次
- 纠正：200 窗只作通路，不作正式（旧 R 0.388 / M 0.577 作废）
- 配置可配 `max_samples` / `eval_samples` / `train.augment` / `model.conv_channels`；缓存指纹不匹配则重建
- **200 测通**：R report=0.279，M=0.262（139/61/61）
- **1500 @ 256 ms（已作废）**：R report=0.421，M=0.399
- 窗长扫描 R：256=0.421，400=0.565，**512=0.803**，800=0.778，1000=0.803；锁定 512
- **1500 @ 512 ms 正式**：R report=**0.803**（早停 47），M=**0.734**（早停 81）
- 见 `doc/results.md`
