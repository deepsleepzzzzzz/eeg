# Tests —— 检查入口与覆盖范围

一条命令跑完（含审计与变异测试）：`python verify_all.py`（约 2.5 分钟）。
只跑单元测试：`python -m pytest tests -q`（约 1 分钟，pre-commit 钩子跑的就是这条）。
实际项数以运行输出为准；检查入口和变异测试均沿用启动它们的 Python 解释器。

## 缓存与交付回归

- `test_cache_integrity.py` —— 新克隆重建、配置/来源/数组/元数据失配、重建中断、训练前拒绝和测试集封闭；契约见 [数据模块](../src/data/README.md)。
- `test_review_portability.py` —— 官方清单缺失/错配、Git 忽略例外、解释器选择和历史登记路径迁移；清单来源见 [数据溯源](../data/DATA_PROVENANCE.md)。

## 论文专用的五个核心测试

- `test_split_invariants.py` —— 划分边界与窗数（参数一律从当前主协议读）
- `test_no_leakage.py` —— 训练/验证/测试之间没有跨段窗重叠
- `test_representation.py` —— 张量形状、行归一化、分位边界只用训练集拟合
- `test_metrics.py` —— accuracy / macro F1 / balanced accuracy 的定义
- `test_golden_regression.py` —— 存下来的预测能复现存下来的指标

## 论文数字与台账守卫

- `test_paper_integrity.py` —— 图表编号/引用/参考文献自洽；表 1–7 与图 13 逐格核对
- `test_paper_noise_tables.py` —— 表 8 / 表 9 逐格核对（表 9 纯能量列追到 `reports/energy_only_*.csv`）
- `test_paper_figures.py` —— 图 13 的参数量与来源
- `test_paper_figures.py::test_every_figure_has_an_explanation` —— **有图必有讲解**：
  每张图都要在正文里有实质解释（含解释性动词或 ≥40 字），"如图 N 所示"不算
- `test_figure_caption_consistency.py` —— **图文适配**：题注写的参数（通道/滞后/归一化/色标/
  种子数/验证集）必须与绘图函数写出的参数台账一致（2026-10-17 由此抓出图 12 题注漏写"验证集"）
- `test_paper_prose_numbers.py` —— 正文里不在表里的数字（消融 / t-SNE 漂移 / TV / 效率比 /
  跨工况 / 表 5 的两列网络数字）逐条复算；2026-10-17 由此抓出并修正 2 处表述
- `test_assets_integrity.py` —— **资产完整性**：参考文献 25 条逐条格式、16 张图 PNG 有效性、
  实验配置 protocol 字段、扫描/调参配置可解析、数据清单一致
- `test_paper_tests_are_effective.py` —— **对论文测试本身做变异测试**：注入 8 类已知错误，
  逐条验证对应测试必须失败、原文必须通过（2026-10-17 由此发现表 8 测试的"双重去头"bug：
  "干净"行此前从未被核对）
- `test_registry_scans.py` —— `registry_scans.csv` 与 artifacts 磁盘一致、无残缺 run
- `test_artifacts_ledger.py` —— 每个 artifacts 目录都在台账里；非 v5 协议目录必须有标注
- `test_provenance.py` / `test_audits.py` —— 数据溯源与审计脚本本身的守卫

## 防陈旧 / 隔离

- `test_no_stale_code.py` —— 审查代码不许写死旧协议名、旧窗长、假检查、绝对路径
- `test_no_old_project_refs.py` —— 新项目自包含，不许引用旧项目
- `test_verify_all_guard.py` —— `verify_all.py` 的结果格式化函数必须健壮
- `test_src_compiles.py` —— `src/` 全部可编译

## 协议驱动的加载器

测试里的窗口/步长/窗数/划分一律从 `tests/_protocol.py`（自动解析 `status: main`）取，
**不许写死 1600/800 或 916/195/195** 这类历史参数 —— 曾经因此让测试在协议升级后仍"全绿"。

## 不用 pytest 也能跑

```powershell
python tests/run_all.py
```
