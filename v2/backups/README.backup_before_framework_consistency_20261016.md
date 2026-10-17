# SEU Fault Diagnosis v2 — 项目框架说明

> 用户 2026-10-16 明确：**旧项目（`D:\EEG_SEU成果_20261005`）整个不要碰**——不读、不写、不复制、不删除。
> 本项目必须自包含（原始数据在本项目 `data/seu/`，与官方 blob sha 已核验一致）。
> 这条由 `tests/test_no_old_project_refs.py` 强制。

## 这是什么

用**符号状态转移概率张量**作为振动信号的输入表示，配极小容量 CNN/GCN 做齿轮箱与轴承的故障诊断。
论文正文在 `reports/论文初稿_v1.md`（工作版 `reports/paper_draft_v2.md`）。

## 当前主协议

- **主协议：`configs/protocols/main_70_15_15_v5.yaml`（`status: main`）** —— 时间有序 70/15/15、
  窗 2048/步 1024、滞后 1–10、6 档等频分位（边界只在训练集拟合）、训练集固定 μσ 归一化。
- 代码里**一律不写死协议名**：用 `status: main` 自动解析（硬规则第 2 条，`tests/test_no_stale_code.py` 强制）。
- 其它协议：`compat_80_20_random`（文献口径对照）、`noise_*`、`external_new_record`、`main_70_15_15(_v2..v4)`（**历史，勿引用**）。

## 目录职责

```
src/                    公共代码：所有实验共用，禁止写死协议参数
  config/               协议加载与透传（唯一真源）
  data/                 原始 CSV → 切窗 → 划分 → 表示缓存
  representation/       符号化与状态转移张量
  models/               CNN / GCN / 原始波形基线
  training/             训练循环与跑批
  tuning/               扫描脚本（scan_variants 等）
  analysis/             分析脚本（互信息 / LDA / 跨工况 / 能量对照 / FLOPs）
  reporting/            图表生成（论文表图全部自动生成）
  audit_*.py            审计：端到端 / 干净性 / 数据溯源 / 论文数字溯源 / 变异测试
  utils/                控制台编码、IO、交付物改前备份
configs/
  protocols/            **协议（唯一真源）**：划分/窗口/预处理/表示/训练超参
  scan/                 扫描变体定义（configs/scan/*.yaml），跑批脚本用它生成实验
  experiments/          单次实验配置（手动 `train.py --experiment` 用）；其 protocol 字段
                        必须等于当前主协议（由 tests/test_protocol_structure.py 校验）
  tuning/               调参与速度基准配置
data/seu/               原始数据（1.6 GB，20 个 CSV）+ 官方 _files.json 清单
artifacts/              实验结果，**按实验/扫描分批隔离**（详见 artifacts/README.md 台账）
  main_70_15_15_v5/     主结果 40 run（论文表 4 的唯一来源）
  variant_scan*/        各专项扫描（消融/噪声/基线/表示/滞后/能量/cross_condition…）
reports/                论文表图与报告（自动生成，勿手改）；论文正文也在这里
tests/                  84 项检查：协议/泄漏/边界/指标/图表自洽/数字溯源/防陈旧…
legacy/                 v1–v4 历史结果与登记表（**冻结，禁止引用**）
backups/                交付物改前备份 + 台账（背份规则见 AGENTS.md 第九节）
registry.csv            主实验总账（40 run）
registry_scans.csv      扫描实验总账（所有 variant_scan*/cross_condition run，自动生成）
claims_evidence.md      论文主张 ↔ 实验/产物 映射（写作与自查用）
verify_all.py           一键跑完全部检查（6 项）
```

## 规则（硬规则见 AGENTS.md）

1. **协议是唯一真源**：任何影响结果的参数都写在协议 yaml 里，代码不许靠默认值兜底。
2. **测试集封存**：模型冻结前不得读取；只开一次。
3. 训练集拟合参数 → 验证集调参/早停 → 测试集只评一次。
4. **论文表图全部由 artifacts 自动生成**，禁止手改数字。
5. 每完成一件事立刻 git 提交（不许攒）；交付物改动前先备份。
6. 旧项目不碰；不引用来路不明的数字。

## 常用命令

```powershell
# 全部检查（6 项：单元测试 / 端到端审计 / 数据溯源 / 论文数字溯源 / 干净性 / 变异测试）
python verify_all.py

# 建数据缓存（按主协议）
python src/data/build_dataset.py --protocol configs/protocols/main_70_15_15_v5.yaml --task gear --conditions 20_0

# 一次扫描（示例：滞后取样对照）
python src/tuning/scan_variants.py --variants configs/scan/lag_spacing_v5.yaml --task gear --condition 30_2 `
  --workers 6 --seeds 42,43,44 --models cnn,gcn --device cuda --out-name variant_scan_lagspace

# 出论文表图
python src/reporting/make_paper_figures.py ; python src/reporting/make_teacher_figures.py

# 交付物改前备份
python src/utils/backup_file.py <tag> reports/论文初稿_v1.md
```
