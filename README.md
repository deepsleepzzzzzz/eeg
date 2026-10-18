# 状态转移网络：旋转机械故障诊断

本分支在东南大学 DDS 上做轴承、齿轮五分类。信号做成多时间步有向加权转移矩阵，再用 CNN 或 GCN 分类。

脑电 Go-nogo 四分类不在本分支。要看那条线，检出标签 `eeg-cnn`、`eeg-gcn` 或 `rp-mrp-baseline`。

协议和命令：`doc/method.md`。数字：`doc/results.md`。配置键：`doc/config.md`。老师原话：`doc/teacher_improvement_directions.md`。

给老师：`doc/handoff/seu_stage_report.docx`。图在 `doc/figures/seu/`。源码在本仓库。

## v2（2026-10-17）

新版工程放在 `v2/`，自包含（`src/`、`configs/`、`tests/`、`scripts/`），不依赖本目录的旧实现，旧代码保持不动：

- 主协议：`v2/configs/protocols/main_70_15_15_v5.yaml`（`status: main`）
- 一键检查：`python v2/verify_all.py`（单元测试 / 端到端审计 / 数据溯源 / 论文数字溯源 / 干净性 / 变异测试）
- 论文正文：`v2/reports/论文初稿_v1.md`
- 原始数据不入库（见 `v2/.gitignore`）：按 `v2/data/README.md` 放好 CSV 后，运行 `python v2/src/data/build_dataset.py --protocol v2/configs/protocols/main_70_15_15_v5.yaml --task gear --conditions 20_0`