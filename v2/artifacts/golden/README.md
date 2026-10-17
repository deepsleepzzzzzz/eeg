# artifacts/golden —— 冻结的**回归测试**基准（不是论文结果！）

Frozen old results for golden regression only. Do not overwrite. **NOT paper results.**

## 用途

只被 `tests/test_golden_regression.py` 使用：从每个 run 存下来的 `test_predictions.csv`
重新计算 accuracy / balanced accuracy / macro F1 / 混淆矩阵，必须与同目录 `metrics.json`
里存的值完全相等（atol 1e-12）。它检验的是**指标计算代码的行为没有变**，与模型好坏无关。

## 内容与来源（重要）

- 8 个 run = 4 个任务-工况组合 × {CNN, GCN}，均为 seed 42。
- 协议是**旧协议**：`protocol: seu_time_split`，窗长 1600 / 步长 800，输入张量 `[21, 6, 6]`
  （3 通道 × 7 滞后），每类验证 261 窗 —— 即当时的 **70/20/10** 划分，**不是当前主协议**。
- 各目录里的 `metrics.json` 带 `test_*` 字段，是因为旧协议当时确实开过一次测试集
  （`test_evaluated: true` / `test_acc_once` / `test_evaluated_after_freeze: true`）。
  这些测试集数字属于历史遗留。
- 对应的旧协议 yaml（`main_70_20_10.yaml`）**已不在** `configs/protocols/` 中，因此这批数据
  **无法重跑、也不该重跑**。

## 纪律

1. **禁止覆盖**本目录下任何 `metrics.json` / `test_predictions.csv`（回归基准必须冻结）。
2. **禁止把这里的任何数字写进论文**。论文的全部数字只能来自：
   - 主结果：`artifacts/main_70_15_15_v5/`（时间有序 70/15/15，窗 2048/1024，输入 `[30,6,6]`，每类验证 152 窗）
   - 图表：`reports/`（由 `src/reporting/*.py` 从上述 artifacts 生成）
3. 本目录的数字与当前主协议**不可比**（划分、窗长、滞后数、通道数全都不同）。

改写于 2026-10-16：原文只有一行英文，容易被误当成主结果引用；备份见
`backups/README.backup_before_golden_readme_20261016.md`。
