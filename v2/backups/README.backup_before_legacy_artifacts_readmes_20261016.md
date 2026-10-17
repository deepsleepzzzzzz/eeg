# artifacts —— 实验产物目录台账

> 生成于 2026-10-16。**看这个文件就知道哪个目录能用、哪个不能用。**
> 规矩见 `AGENTS.md`；交付物改动要先备份（第九节）。

## 唯一可用作论文数字的目录

**`main_70_15_15_v5/`** —— 当前主协议（`configs/protocols/main_70_15_15_v5.yaml`，`status: main`）的主结果，
40 个 run（4 任务-工况 × {CNN, GCN} × 5 seed）。论文表 4（主结果）、表 5（LDA 复核）、表 8（抗噪）、
表 9（口径对照）的全部数字来自这里。**除此之外的任何目录，引用前必须逐个核对协议。**

## 分类

### A. 当前可引用（论文数字的来源）

| 目录 | 用途 |
|---|---|
| `main_70_15_15_v5/` | **当前主协议主结果**（论文表 4/5/8/9） |
| `variant_scan/` | 变体扫描汇总（滞后/窗长/档数/模型层开关消融，论文 §4.1） |
| `variant_scan_rep_v5/` | 表示层专项扫描（v5） |
| `variant_scan_noise_v5/` | 加噪扫描（v5，逐窗口径） |
| `variant_scan_mixraw/` | 混合表示（转移张量 + 原始点）扫描 —— 结论：不一致，未采用 |
| `variant_scan_normtest2/` | 归一化方式对照（逐窗 z-score vs 训练集固定尺度）→ v5 选后者 |
| `variant_scan_prefilter/` | 前置滤波对照 —— 结论：两种归一化下都是负收益，故 v5 声明 prefilter: none |
| `variant_scan_win_fixed/` | 窗长消融（固定步长 1024，窗长 1600/2048/3200） |
| `cross_condition/` | 跨工况迁移（20-0 → 30-2）→ 论文 §5.5 第（2）条的 51.14% |
| `tuning/` | 调参中间记录，不直接引用 |

### B. 历史协议结果（禁止引用，见 legacy/README.md 体检报告）

- `main_70_15_15/`
- `main_70_15_15_v2/`
- `main_70_15_15_v3/`
- `main_70_15_15_v4/`
- `variant_scan_round1/`
- `variant_scan_round2/`
- `variant_scan_archived/`
- `variant_scan_noise_v4/`
- `variant_scan_rep_v5_bak/`
- `legacy_bug_smoke/`

### C. 冻结：回归测试基准（禁止覆盖、禁止引用）

- `golden/`

### D. 封存：测试集张量

- `sealed_test/`

### E. 探针/临时（可由脚本重跑；不参与论文）

- `chprobe_add0_0123/`
- `chprobe_all_01234567/`
- `chprobe_c0_0123456/`
- `chprobe_cur_123/`
- `chprobe_par_123456/`
- `probe_bins_6/`
- `probe_bins_8/`
- `probe_bins_10/`
- `probe_bins_12/`
- `probe_channels_7/`
- `smoke_test/`
- `synthetic_audit/`

### F. 协议占位空目录（0 文件）

- `ablation_70_15_15/`
- `compat_80_20_random/`
- `external_new_record/`
- `noise_matched/`
- `supp_structure/`

### G. 变体扫描的**单次运行工作目录**（可删，可由脚本重跑）

共 62 个 `var_<12位十六进制>/` 目录，合计约 6.3 GB，是扫描时每个变体单独一份的中间产物。
论文数字不直接来自它们（汇总结果在 `variant_scan*/`）；确认不再重跑后可以清除。

```
  var_042816a6d8a3, var_049a5ae79d54, var_0825d9dbc90c, var_08ad23c31675, var_0c0887817483, var_0df67c724888, var_178ad6cba8e1, var_1da9e88f2bda, var_215d1d7bb18d, var_2671308bcba6, var_2732b640251f, var_2bd2943bed90 ...
```

## 特别提醒（踩过的坑）

1. **`golden/`** 是回归测试夹具（`tests/test_golden_regression.py` 用），里面是**旧 70/20/10 协议**、
   还带着当时的测试集数字。目录名带 `main_` 前缀，极易被误当主结果 —— 详见 `golden/README.md`。
2. **`sealed_test/`**：当前只有 `main_70_15_15/`（**v1 时代**的封存张量）。当前 v5 的封存测试集
   **尚未编码**（路径应为 `sealed_test/main_70_15_15_v5/`），也从未开过。
3. **`main_70_15_15`、`_v2`、`_v3`、`_v4`** 各有 40 个 run，但全部是**旧协议**：
   v1 结果不可用（符号化尺度 bug）、v2 的 GCN 有语义缺陷、v3 通道选择不被官方背书、v4 已被 v5 取代。
   逐版证据见 `legacy/README.md`。
4. **`legacy/` 里目前只放了两样**：v1–v4 的**登记表**（`registry_main_v*_40runs.csv`）和那份体检报告；
   **结果目录本身仍留在 `artifacts/`**（上面的 B 类）。若要彻底归档进 `legacy/`，需要同步改
   `src/audit_main.py` 的扫描范围（它会 `artifacts/**/manifest.json` 全扫）与 `legacy/README.md`，
   属于结构改动，需单独确认后再动。

改写于 2026-10-16（原文只有一行英文）。备份：`backups/README.backup_before_artifacts_readme_20261016.md`。
