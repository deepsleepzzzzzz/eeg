# artifacts —— 实验产物目录台账

> 更新于 2026-10-16：v1–v4 旧协议结果已归档进 `legacy/`（见 `legacy/README.md` 末节）。
> **看这个文件就知道哪个目录能用、哪个不能用。** 规矩见 `AGENTS.md`。

## 唯一可用作论文数字的目录

**`main_70_15_15_v5/`** —— 当前主协议（`configs/protocols/main_70_15_15_v5.yaml`，`status: main`），
40 个 run（4 任务-工况 × {CNN, GCN} × 5 seed）。论文表 4、表 5、表 8、表 9 的全部数字来自这里。
**其它任何目录，引用前必须逐个核对协议。**

## A. 当前可引用（论文数字来源）

| 目录 | 用途 |
|---|---|
| `main_70_15_15_v5/` | **当前主协议主结果**（论文表 4/5/8/9 的唯一来源） |
| `variant_scan/` | 变体扫描汇总（滞后/窗长/档数/模型层开关，论文 §4.1） |
| `variant_scan_rep_v5/` | 表示层专项扫描（v5） |
| `variant_scan_noise_v5/` | 加噪扫描（v5，逐窗口径） |
| `variant_scan_mixraw/` | 混合表示扫描 —— 结论：不一致，未采用 |
| `variant_scan_normtest2/` | 归一化对照 → v5 选 train_global_zscore |
| `variant_scan_prefilter/` | 前置滤波对照 —— 结论：负收益，v5 声明 prefilter: none |
| `variant_scan_win_fixed/` | 窗长消融（固定步长 1024） |
| `cross_condition/` | 跨工况迁移 20-0 → 30-2（论文 §5.5 第（2）条） |
| `tuning/` | 调参中间记录，不直接引用 |

另有一个**机制目录**：`variant_scan_archived/` —— 扫描脚本用来把"本次未定义的变体"挪进来的归档位（`scan_variants.py::archive_foreign_variants`），**不要手删**。

## B. 历史协议结果 —— 已归档，不在这里

v1–v4 的结果与缓存已整体移入 `legacy/`（2026-10-16）：

| 原位置 | 现位置 | 状态 |
|---|---|---|
| `artifacts/main_70_15_15/` | `legacy/results/main_70_15_15_v1/` | **结果不可用**（符号化尺度 bug） |
| `artifacts/main_70_15_15_v2/` | `legacy/results/main_70_15_15_v2/` | CNN 可用；GCN 有语义缺陷 |
| `artifacts/main_70_15_15_v3/` | `legacy/results/main_70_15_15_v3/` | 可用，通道选择不被官方背书 |
| `artifacts/main_70_15_15_v4/` | `legacy/results/main_70_15_15_v4/` | 可用，已被 v5 取代 |
| `artifacts/smoke_test/`、`legacy_bug_smoke/` | `legacy/results/…` | 早期冒烟/缺陷复现 |
| `artifacts/variant_scan_round1|round2|noise_v4|rep_v5_bak/` | `legacy/scans/…` | 早期扫描 |

**一律禁止引用。** 逐版缺陷证据见 `legacy/README.md`；登记表（`legacy/registry_main_v*_40runs.csv`）保持原样，未回改。

## C. 冻结：回归测试基准（禁止覆盖、禁止引用）

- `golden/` —— 详见 `golden/README.md`：旧 70/20/10 协议、含当时的测试集数字，只被 `tests/test_golden_regression.py` 用作指标计算回归。

## D. 封存：测试集张量

- `sealed_test/` —— **当前为空**。v1 时代的张量已归档到 `legacy/sealed_test_v1/main_70_15_15_v1/`；
  当前 v5 的封存测试集**尚未编码**（路径应为 `sealed_test/main_70_15_15_v5/`），也从未开过。

## E. 探针 / 临时（不参与论文，可重跑）

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
- `synthetic_audit/`

## F. 协议占位空目录（0 文件）

- `ablation_70_15_15/`
- `compat_80_20_random/`
- `external_new_record/`
- `noise_matched/`
- `supp_structure/`

### G. 变体扫描的单次运行缓存（可清）

共 62 个 `var_<12位十六进制>/`，约 6.3 GB：每个变体一份 `dataset/` 缓存，由 `src/tuning/scan_variants.py` 生成。
论文数字不直接来自它们（汇总在 `variant_scan*/`）。清理工具：

```powershell
python src/cleanup_caches.py            # dry-run，列出可删的
python src/cleanup_caches.py --apply    # 真删（只删 split_hash 与当前主协议不一致的）
```

## 特别提醒（踩过的坑）

1. **`golden/` 的数字不能引用**（旧协议 + 含测试集结果，目录名还带 `main_` 前缀）。
2. **`sealed_test/` 为空不等于"测试集开过了"**：说明 v5 还没建封存张量。
3. **旧协议结果已不在 artifacts/**：看到 `main_70_15_15_v1..v4` 请到 `legacy/results/` 找，且不得引用。
4. `reports/` 里的数字必须能追到 `main_70_15_15_v5/` 或相应 `variant_scan*/`；追不到就是有问题。

本文件历史：2026-10-16 由一行英文改为台账；同日 v1–v4 归档。备份见 `backups/`。
