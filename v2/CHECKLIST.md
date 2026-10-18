# 检查清单（每次改动后照做）

> 目的：防止"检查本身失效"——本项目已经发生过三次：
> ① 符号化尺度 bug（齿轮 95%→100%）在旧审计下完全没被发现
> ② 审计脚本写死旧协议，导致"24 项全过"实际审的是 v2
> ③ 审计脚本没有非零退出码，"审计通过"这句话没被任何东西验证
> 所以现在用**变异测试**来证明审计本身还有效。

---

## 一条命令

```bash
python verify_all.py
```

跑 6 项，任何一项失败即以非零码退出（约 2.5 分钟）：

| 检查 | 内容 | 当前 |
|---|---|---|
| 单元测试 | `pytest tests` | 以运行输出为准 |
| 端到端审计 | `src/audit_main.py`（从原始 CSV 独立复算、泄漏、标签顺序、退化窗） | 44 项 |
| 数据溯源审计 | `src/audit_provenance.py`（齿轮/轴承不串、A~F 全通过） | 4/4 工况 |
| 论文数字溯源 | `src/audit_paper_numbers.py`（论文每个数字追到本项目 run，协议必须是 v5） | 全通过 |
| 干净性审计 | `src/audit_clean.py`（划分隔离、测试集未访问、归一化来源、seed 独立性） | 22 项 |
| **变异测试** | `src/audit_mutation.py`（往缓存注入 4 种错误，看审计能否抓到） | 4/4 全中 |

（单元测试项数每次新增测试会变；以 `verify_all.py` 输出的实际数字为准。）

## 自动执行

`.git/hooks/pre-commit` 已在每次 `git commit` 前强制跑 **pytest tests**。
未通过则拒绝提交。跳过：`git commit --no-verify`（**禁止**，见 AGENTS.md 硬规则第 5 条）。

变异测试因为会改动数据缓存，**不在钩子里**，请手动跑 `python verify_all.py`。

---

## 论文专用的五类测试（"五个测试"）

论文的每个数字都必须能被机器核对，对应下面五个文件（另有各自的守卫测试）：

| 测试文件 | 盯什么 |
|---|---|
| `tests/test_split_invariants.py` | 划分边界与窗数（一律从当前主协议读，不许写死） |
| `tests/test_no_leakage.py` | 训练/验证/测试之间没有跨段窗重叠 |
| `tests/test_representation.py` | 张量形状、行归一化、分位边界只用训练集拟合 |
| `tests/test_metrics.py` | accuracy / macro F1 / balanced accuracy 的定义 |
| `tests/test_golden_regression.py` | 存下来的预测能复现存下来的指标 |

配套的台账类守卫（2026-10-17 新增）：

| 测试文件 | 盯什么 |
|---|---|
| `tests/test_paper_integrity.py` | 图/表编号连续、都被正文引用、参考文献完整、表 1–7 与图 13 逐格核对 |
| `tests/test_paper_noise_tables.py` | 表 8 / 表 9 逐格核对（含表 9 纯能量列追到 `reports/energy_only_*.csv`） |
| `tests/test_registry_scans.py` | `registry_scans.csv` 与 artifacts 磁盘一致；没有残缺 run |
| `tests/test_artifacts_ledger.py` | 每个 artifacts 目录都在台账里；非 v5 协议目录必须有标注 |
| `tests/test_no_stale_code.py` | 审查代码本身不许写死旧协议/旧参数/假检查 |

---

## 什么时候跑什么

| 场景 | 必跑 |
|---|---|
| 改了代码（模型/训练/表示） | `python verify_all.py` |
| **改了协议 yaml** | `python verify_all.py` **+ 重建对应数据缓存** |
| **重建了数据缓存** | `python verify_all.py`（尤其变异测试） |
| 跑了新实验 | `python src/audit_main.py` + 更新 `registry.csv`（主实验）/ `registry_scans.csv`（扫描） + 论文表 |
| 论文定稿前 | `python verify_all.py` **+ 开测试集（只开一次）** |

---

## 三条硬规则

1. **协议是唯一真源。** 任何影响结果的参数都必须在 `configs/protocols/*.yaml` 里显式声明，不许靠代码默认值。
2. **测试集只开一次。** `artifacts/sealed_test/` 在模型冻结前不得读取。
3. **审计要能自证。** 每次改协议/模型后跑变异测试；若某种错误注入后审计不报警，说明审计有盲区，必须先补审计。

---

## 当前协议

`main_70_15_15_v5`（`status: main`）：三通道行星齿轮箱 xyz、窗长 2048/步长 1024、
滞后 1–10、6 档等频分位、**训练集固定尺度归一化**、时间有序 70/15/15、测试集封存。

历史版本：v1（常数 lr）→ v2（加学习率衰减）→ v3（8 通道，官方未背书，不采用）→ v4（表示调优）→ **v5**（训练集固定 μσ 归一化）。
**v1–v4 的结果已整体归档到 `legacy/results/`（不是 `artifacts/`），一律禁止引用**；
登记表在 `legacy/registry_main_v*_40runs.csv`，逐版缺陷见 `legacy/README.md`。

---

## 定期维护

```bash
python src/cleanup_caches.py           # 只列出，不删
python src/cleanup_caches.py --apply   # 真正删除
python src/reporting/make_registry_scans.py   # 刷新扫描实验总账
```

删除与当前主协议无关的历史数据缓存（`artifacts/var_*`，内容哈希命名）。
它们可随时重建，删掉不影响任何已有结果；保留与当前主协议 `split_hash`
一致的那些，避免重复重建。

## 为什么会有历史缓存混存

`artifacts/var_<hash>` 是按【协议内容哈希】命名的数据缓存。换协议（v2→v4→v5）
就会产生新的一批，旧的不会自动消失。同一个 SNR 可能同时存在 v2 / v4 / v5 三份。
**它们不会互相误读**（指纹不同就落在不同目录），但会占空间、也会让人分不清。

---

## 跑批的"完成判据"（踩过三次的坑）

**不要用"目录数"判断扫描是否跑完** —— `train.py` 一开始就创建目录，
**目录存在 ≠ 跑完**。已经因此报错过三次数字：

| 次数 | 场景 | 错误 |
|---|---|---|
| 1 | 加噪扫描 | 报 −6/−8 dB 的 GCN "跑失败"，其实还在跑 |
| 2 | 表示层扫描 | 报窗长变体的 GCN "失败"，其实还在跑 |
| 3 | v5 噪声曲线 | 报的 −6/−8/−10 dB 数字偏低 0.5~1.8 个点 |

**正确的完成判据（两个都要满足）：**
```powershell
(Get-Process -Name python -ErrorAction SilentlyContinue).Count   # 应该为 0
# 且日志里出现 "完成 N/M" 那一行
```

**兜底检查（三个）：**
```powershell
python src/audit_main.py                       # 报出只有 manifest.json 的残缺目录
python src/reporting/make_registry_scans.py    # 扫描总账：incomplete / corrupt 行必须是 0
python -m pytest tests/test_registry_scans.py -q
```
