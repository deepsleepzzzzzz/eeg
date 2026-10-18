# 项目规则（给 AI 编码代理看）

这是一个**科研项目**，不是软件产品。代码的价值在于**结果可信**，不在于跑得快。
下面几条是硬规则，每次会话都必须遵守。

## 一、动手前先读这三个文件

```
CHECKLIST.md                             检查清单 + 三条硬规则
configs/protocols/main_*.yaml            当前主协议（status: main 的那个）
legacy/                                  历史版本结果（v1..v4）
```

## 二、硬规则

1. **协议是唯一真源。** 任何影响结果的参数都必须在 `configs/protocols/*.yaml`
   里**显式声明**，不许靠代码里的默认值兜底。
   - 新加参数 = 同时改 yaml、`protocol.py` 的透传、以及协议自洽测试
   - 绝不在代码里写死窗口长度、滞后阶数、归一化方式、通道索引等

2. **不许写死协议文件名。** 需要用当前主协议时，**自动解析 `status: main`**：
   ```python
   for f in sorted((ROOT/"configs"/"protocols").glob("main_*.yaml")):
       if str(load_protocol(f).get("status")) == "main": return f
   ```
   历史上审计脚本写死过 v2，导致"24 项全过"实际审的是旧协议。

3. **测试集只开一次。** `artifacts/sealed_test/` 在模型冻结前**不得读取**。
   读它的唯一入口是 `train.py --evaluate-test --confirm-test`，且必须先得到用户明确同意。

4. **审计要能自证。** 审计脚本本身可能失效，必须用变异测试证明它还有效。
   每次改协议/模型/数据后都要跑，发现盲区就先补审计再继续。

5. **不许绕过检查。** 不使用 `git commit --no-verify`。
   若检查失败，修问题，不是绕过去。

## 三之前、提交纪律（2026-10-15 事故后新增）

**每完成一件独立的事就立刻提交一次 git，不要攒。**

2026-10-15 踩的坑：一天里写了基线模型、表示分派、FLOPs 计算、四张图表、
五个跑批脚本，全部没提交——**当天 324 次训练的跑批记录和全部新代码只存在于工作区**，
硬盘故障或一次误操作就全没了。

提交粒度参考：
- 新增/修改一个模块 → 提交一次
- 修一个 bug + 加对应测试 → 提交一次
- 跑完一批实验、生成一批图表 → 提交一次

**提交前必须 `python verify_all.py` 并全绿**（pre-commit 钩子也会跑 pytest，不许 --no-verify）。

## 三、改完代码的必做动作

```bash
python verify_all.py        # 或等价的 pytest tests（259 项，约 1 分钟）
```

**然后在回复里写明结果**（通过了几项 / 有没有失败）。没跑就不能说"改好了"。

改动涉及数据或协议时，还要：
```bash
python src/data/build_dataset.py --protocol <主协议> --task <t> --conditions <c> --force
python verify_all.py
```

## 四、发现问题的标准处理顺序

1. **先证明问题真实存在**（跑命令、看数据），不要凭猜测下结论
2. **找出根本原因**，不要只修表象
3. **修完加一条检查**，防止同类问题再出现
4. **验证这条检查有效**（故意注入错误，看它是否报警）
5. 在回复里说明：发现了什么、怎么修的、怎么证明修好了

## 五、长期授权（2026-10-15 用户明确下达，永久生效）

用户已下达长期授权：**跑批、改代码、建新协议变体、联网检索文献、安装依赖、
生成图表与报告，一律不必再问，做完直接报结果。**
不要每做一步就停下来确认，不要问"要不要继续"。

### 只有这四件事必须停下来问

1. **开测试集**（`artifacts/sealed_test/`，全项目只能开一次）
2. **删改 `legacy/` 里的历史结果**
3. **修改主线协议 `status: main` 的评估口径**（划分方式、窗口、通道数）
4. **任何会让已有结果作废的改动**

除了这四条，一律直接做，做完在回复里报结果（通过了几项 / 数字是多少 / 有没有失败）。

### 边界

- 新增**旁支/兼容协议**（如 `compat_*`、`scan/*`）不算改主协议，不用问。
- 跑批完成后按第七节的判据自检，再报结果；没跑完不许报。
- 联网检索文献、下载 PDF 属于常规操作，不用问。

## 六、这个项目已经踩过的坑（都是"静默使用错误配置"）

| 坑 | 表现 | 现在靠什么防 |
|---|---|---|
| 符号化尺度不一致 | 6 档只用 2 档，齿轮卡在 95% | `tests/test_bin_edges_scale.py`（3 项） |
| 审计写死旧协议 | "24 项全过"实际审的是 v2 | 审计自动解析 `status: main` |
| 审计没有退出码 | 钩子永远"通过" | 两个审计加 `SystemExit` |
| 扫描脚本写死 v2 | 加噪实验用了旧表示 | 扫描脚本自动解析 `status: main` |
| 假 GPU | 声称用 GPU 实际跑在 CPU | `device=cuda` 时加运行时断言 |
| 协议里的窗数抄错 | 写 916 实际 715 | 协议窗数一致性测试 |
| 审计对内容泄漏无感 | 验证集混入训练样本不报警 | 逐窗 SHA256 求交 |
| 审计对标签错乱无感 | 打乱标签不报警 | 标签顺序检查 |
| 审计偶发失败（原因未明） | 2026-10-15 20:57 出现一次 `PASS=40 FAIL=2`（两项表示一致性检查），此后 24 次连跑均通过，未能复现 | 见 `reports/incident_audit_flaky_20261015.md`；已加 CSV 短读守卫与失败偏差打印 |
| scan 覆盖旧结果 | `scan_variants.py` 把结果写进同一个 `artifacts/variant_scan/`，后续扫描就地覆盖：窗长消融 win1600/win3200（各 40 个文件）被噪声扫描吃掉，只能从 git 历史捞回 | `archive_foreign_variants()` 开跑前自动把「本次配置未定义的变体」移到 `artifacts/variant_scan_archived/` |
| 加噪参考功率口径 | `add_noise` 按每个窗自己的功率缩放噪声 → 加噪窗能量恒等于干净窗能量×常数 → "看窗能量"成了任何 SNR 下都有效的捷径。实测：齿轮 20-0 在 −10 dB 以下，CNN/GCN 与"纯能量+最近类中心"同分（84.4 vs 84.0），表示没提供任何额外信息 | `noise.reference: global`（所有窗共用一个参考功率）；`tests/test_noise_reference.py`（3 项） |

| artifacts 台账漏登记 | `variant_scan_energy` 等 4 个目录从未写进 `artifacts/README.md`；两名录用了 v4 协议却没标注 | `tests/test_artifacts_ledger.py`（2 项） |
| 扫描实验没有总账 | 1300+ 个 run 散在 15 个目录里，"跑完没有"只能靠人记 | `registry_scans.csv` + `src/reporting/make_registry_scans.py` + `tests/test_registry_scans.py`（3 项） |
| 测试本身写错 | 表 8 的测试先 `tab[1:]` 再 `_body()`，把"干净"行也当表头删掉——该行从未被核对 | `tests/test_paper_tests_are_effective.py`（对论文测试做变异测试，8 项） |
| 图文不适配 | 题注写"第 3 个滞后/1σ/行归一化"，图里画的却是别的参数；旧测试只查结构不查内容 | `tests/test_figure_caption_consistency.py`（3 项，题注 ↔ 绘图参数台账） |
| 论文里"形式检查"冒充真检查 | 表 9 的纯能量列只断言 `0 < x <= 100`，任何数字都能过 | `tests/test_paper_noise_tables.py::test_table9_energy_classifier_matches_energy_only_csv` |

## 七、跑批之后的完成判据（踩过三次）

**不要用目录数判断扫描跑完。** `train.py` 一开始就创建目录，目录存在 ≠ 跑完。
已经因此把未完成的结果当成最终结果报错过三次。

```powershell
# 正确判据：进程归零 + 日志出现"完成 N/M"
(Get-Process -Name python -ErrorAction SilentlyContinue).Count   # 必须为 0
```
兜底：`audit_main.py` 的"实验目录完整性"会报出只有 manifest.json 的残缺目录。

## 八、论文归属（2026-10-16 用户明确）

- （2026-10-17 补充）**唯一交付物是 `reports/论文初稿_v1.md`**；
  `reports/paper_draft_v2.md` 是早期工作版（正文措辞与个别数字已落后，例如仍写“不以精度领先为卖点”、
  跨工况只写 51.14% 而不区分 CNN/GCN）——**只能当历史参考，不得作为投稿底稿**。
- **论文正文 = `reports/论文初稿_v1.md`**（工作版 `reports/paper_draft_v2.md`，带逐段写作标注）。
  本项目即论文工程：正文、图、表全部在 `reports/`，不要再往外找。
- **旧项目 `D:\EEG_SEU成果_20261005` 整个不要再碰**（2026-10-16 用户明确：不要理旧项目，怕干扰）：
  不读、不写、不复制、不删除它的任何内容。新项目必须自包含（原始数据已在本项目 `data/seu/` 内并与官方
  sha 核验一致；全部实验在本项目 `artifacts/`、`legacy/`、`registry.csv` 里）。
  这条由 `tests/test_no_old_project_refs.py` 强制。
- （历史说明）旧项目 `doc\` 下的 Word 稿（含
  `论文初稿_多时间步状态转移网络与CNN_GCN故障分类.docx`）**已废弃**：不再作为交付物，
  也不要往里写。（该文件长期被 Word 占用，写入会被系统拒绝。）
- 嵌图现状：图 1（`fig_pipeline_style_a.png`）已插入 §2；图 2–图 5 仍只在正文引用，
  PNG 分别是 `fig_lag_mi.png` / `fig_bins_saturation.png` / `fig_confusion.png` / `fig_transition.png`。
- 写 `reports/*.md` 必须保持工作树 CRLF（`core.autocrlf=true`）：用**二进制写入 + 显式 `\r\n`**，
  不要用 `Path.write_text()` —— 它会把 `\n` 再转成 `\r\n`，产生 `\r\r\n`，整文件 diff 变脏（10-16 已踩过）。


## 九、交付物修改必须先备份，且必须进 git（2026-10-16 用户明确）

**任何对交付物（`reports/*.md`、`reports/*.png`、`configs/**/*.yaml`、论文正文与图表）的修改，
必须在写入之前先留一份物理备份，改完连同备份一起 git 提交。**

```powershell
# 1) 改之前先备份
python src/utils/backup_file.py <tag> reports/论文初稿_v1.md
# 2) 再改文件
# 3) 改完提交（备份一起进 git，不许只留在工作区）
git add -A ; git commit -m "paper: ..."
```

- 备份落在 `backups/`，命名 `<原名>.backup_before_<tag>_<日期>.<后缀>`；每次备份自动追加台账 `backups/README.md`。
- 补录历史版本：`python src/utils/backup_file.py <tag> --from-git <git-rev> <文件>`
  （2026-10-16 已用 `a52f3a1` 补录了 claimfix 之前的两个 md）。
- 代码文件由第三节「每完成一件事就提交一次」的 git 纪律保护；交付物改动频繁、影响面大，额外留物理副本便于肉眼对照。
- **绝不允许「改了但没备份、也没进 git」。** 回复里要能报出：改了哪个文件、备份在哪、commit 号是多少。


## 十、目录与脚本（2026-10-17 补）

- **跑批脚本在 `scripts/`**（`run_*.ps1`、`speed_wrap.py`），清单见 `scripts/README.md`。
  脚本一律用 `Split-Path $PSScriptRoot -Parent` / `Path(__file__).parents[1]` 解析项目根，**不许写绝对盘符**。
- **两张总账**：`registry.csv`（主实验，手工登记，历史绝对路径不回改）、
  `registry_scans.csv`（扫描实验，**自动生成**：`python src/reporting/make_registry_scans.py`）。
  跑完批忘了刷新会被 `tests/test_registry_scans.py` 拦下。
- **主张 ↔ 证据**：`claims_evidence.md`。写进论文的每句话都要能在这张表里找到目录与校验；
  表里标"未做"的（外部独立验证 / 随机 80/20 文献口径 / 结构测度）**不许写进论文**。
- **artifacts 台账**：`artifacts/README.md`。新目录必须登记；跑的不是当前主协议的目录必须带标注
  （`tests/test_artifacts_ledger.py` 强制）。
