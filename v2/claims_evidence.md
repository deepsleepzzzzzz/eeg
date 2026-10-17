# 主张 ↔ 证据矩阵（claims_evidence.md）

> 用途：论文里**每一句话主张**，都能在这里找到对应的实验目录、协议和自动校验。
> 写论文/答辩时先查这张表；表里没有的结论，不许写进论文。
> 2026-10-17：由 6 行 TBD 改写为实际台账（此前 TODO 行指向的旧协议名 `main_70_15_15` 已作废）。

**当前主协议**：`configs/protocols/main_70_15_15_v5.yaml`（`status: main`）。
下表的"协议"列写的是**产出该证据时实际用的协议**——不是当前主协议的，必须写明。

## 一、论文主张与证据

| # | 主张（论文里的话） | 论文位置 | 协议 | 实验 / 产物 | 自动校验 | 状态 |
|---|---|---|---|---|---|---|
| C1 | 表示极紧凑：1080 元素、与窗长无关；CNN 27,685 / GCN 54,941 参数 | §2.2、表 1 | v5 | `artifacts/main_70_15_15_v5/`（40 run） | `tests/test_paper_integrity.py` | DONE |
| C2 | 干净条件精度 99.68%–100%、跨 seed 标准差 ≤0.21 点 | §4.2、表 4 | v5 | `artifacts/main_70_15_15_v5/`（4 工况 × 2 模型 × 5 seed） | `tests/test_paper_integrity.py`、`src/audit_paper_numbers.py` C | DONE |
| C3 | 参数量/算量对比（1/139、1/552、0.34 ms） | §4.5、表 7、图 13 | v5 | `src/analysis/flops.py` → `reports/efficiency.json`、`reports/efficiency_all.json`；`artifacts/variant_scan_fill/` | `tests/test_paper_integrity.py`（表 7、图 13） | DONE |
| C4 | 判别力来自表示：LDA 与网络差 ≤0.16 点 | §4.3、表 5 | v5 | `src/analysis/lda_check.py` → `reports/lda_check.csv`；`artifacts/main_70_15_15_v5/` | `tests/test_paper_integrity.py`（表 5） | DONE |
| C5 | 滞后阶数由符号互信息衰减确定（k>10 后 <1/3） | §4.1、图 4、表 2 | v5 | `src/analysis/lag_information.py` → `reports/lag_information_*.csv` | `tests/test_paper_integrity.py`（表 2） | DONE |
| C6 | 档数由归一化互信息饱和曲线确定 | §4.1、图 6、表 3 | v5 | `src/reporting/make_lag_info_figure.py` → `reports/fig_bins_saturation.png`；`artifacts/variant_scan_rep_v5/`（bins 变体） | `tests/test_paper_integrity.py`（表 3） | DONE |
| C7 | 滞后取样方式（连续 vs 间隔）对照 | §4.1、图 5 | v5 | `artifacts/variant_scan_lagspace/`（18 run） | `src/audit_paper_numbers.py` A/B | DONE |
| C8 | 窗长由固定步长配对比较确定（1600/2048/3200） | §4.1 | v5 | `artifacts/variant_scan_win_fixed/`（30 run） | `registry_scans.csv`（协议列） | DONE |
| C9 | 模型/训练层开关（17 项）对结果无显著影响（±0.26 内） | §4.3 | v5 | `artifacts/variant_scan/`（model_layer 变体）、`configs/scan/model_layer.yaml` | `src/audit_paper_numbers.py` A/B | DONE |
| C10 | 与原始波形基线同协议对比：干净 100.00 vs 99.6–99.8 | §5.5(2)、表 6 | v5 | `artifacts/variant_scan/`（raw_cnn / wdcnn / resnet1d） | `tests/test_raw_baselines.py` | DONE |
| C11 | 抗噪曲线（逐窗参考口径，8~9 档 SNR） | §4.6、表 8 | v5 | `artifacts/variant_scan_noise_v5/`（60）+ `variant_scan_noise_restore/`（144）+ `variant_scan_fill/`（12） | `tests/test_paper_noise_tables.py`、`src/audit_paper_numbers.py` D | DONE |
| C12 | **口径效应**：同一模型只换参考功率口径，−10 dB 差 13.9 点、−16 dB 差 40.9 点 | §5、表 9、图 11 | v5 | `reports/table_noise_protocol.md`（由 `variant_scan*` 汇总）；`src/reporting/make_noise_protocol_report.py` | `tests/test_paper_noise_tables.py`、`tests/test_noise_reference.py` | DONE |
| C13 | 零学习窗能量分类器在逐窗口径下 84.0% ≈ 模型 84.4%（−10 dB，齿轮 20-0） | §5.2、§5.3、表 9 | v5 | `src/analysis/energy_only.py` → `reports/energy_only_per_window.csv`、`energy_only_global.csv` | `tests/test_paper_noise_tables.py::test_table9_energy_classifier_matches_energy_only_csv` | DONE |
| C14 | 加幅值特征（log-RMS）不能解决齿轮两类重叠 → 信息损失在波形形状而非幅值 | §5.5(3) | v5 | `artifacts/variant_scan_energy/`（12 run） | `src/audit_paper_numbers.py` A/B | DONE |
| C15 | 跨工况（20-0 → 30-2）迁移结果 | §5.5(2) | v5 | `artifacts/cross_condition/`（15 run：cnn/gcn/raw_cnn/wdcnn/resnet1d） | `registry_scans.csv`、`src/audit_paper_numbers.py` A/B | DONE |
| C16 | 归一化方式选型（逐窗 z-score → 训练集固定 μσ） | §3.2 | **v4**（选型对照，在 v5 定案前跑） | `artifacts/variant_scan_normtest2/`（33 run，协议字段为 v4） | `registry_scans.csv`；`tests/test_artifacts_ledger.py`（必须在台账里带标注） | DONE |
| C17 | 前置滤波是负收益 → v5 声明 `prefilter: none` | §3.2 | **v4**（同上） | `artifacts/variant_scan_prefilter/`（24 run，协议字段为 v4） | `registry_scans.csv`；`tests/test_artifacts_ledger.py` | DONE |
| C18 | 加噪的参考功率口径必须显式声明，否则结果不可比 | §5.1 | v5 | `configs/protocols/*.yaml` 的 `noise.reference` | `tests/test_noise_reference.py`、`tests/test_noise_reference_declared.py` | DONE |

## 二、尚未有证据 → 不许写的主张

| 主张 | 缺什么 | 现状 |
|---|---|---|
| "独立验证 / 跨设备泛化" | 另一台试验台或另一批记录的数据 | `artifacts/external_new_record/` **空**；`configs/protocols/external_new_record.yaml` 只有协议骨架 |
| "与文献在随机 80/20 口径下可比" | 同口径随机划分的自跑结果 | `artifacts/compat_80_20_random/` **空** |
| "结构测度 / 冻结 1-NN 复核" | 该组实验 | `artifacts/supp_structure/` **空** |
| 任何"优于文献"的精度主张 | 同协议、同口径自跑基线 | 现只有齿轮 20-0 的 3 个原始波形基线；**不得跨口径比数** |

## 三、规矩

1. **口径必须写清**：引用任何数字，必须同时说明协议名（v5 / v4 选型对照 / 文献报告值）、口径（验证集还是测试集、逐窗还是全局参考功率）。
2. **测试集未开**：`artifacts/sealed_test/` 为空，论文所有数字都是**验证集**，正文必须写明"验证集"。
3. **同源记录内评估**：每个故障类别只有一次采集记录 → 只能写"同源记录内评估"，不得写"独立验证"。
4. **台账索引**：主实验 `registry.csv`（40 run）、扫描实验 `registry_scans.csv`（1329 run，自动生成）、目录可用性 `artifacts/README.md`。

## 四、本表自身的守卫

这张表会过期，所以用测试盯着：

| 测试 | 盯什么 |
|---|---|
| `tests/test_registry_scans.py` | `registry_scans.csv` 必须与 artifacts 磁盘一致；有残缺 run 直接失败 |
| `tests/test_artifacts_ledger.py` | 每个 artifacts 目录都要在台账里；非 v5 协议目录必须带标注 |
| `tests/test_paper_integrity.py`、`tests/test_paper_noise_tables.py` | 论文里的表 1–9 逐格核对（数字追不到源即失败） |