# scripts —— 跑批与基准脚本

**这些 .ps1 只是"命令行封装的跑批入口"**，本身不含任何协议参数：协议/变体定义一律在
`configs/protocols/*.yaml` 与 `configs/scan/*.yaml`（硬规则第 1 条）。

| 脚本 | 作用 | 产物目录 |
|---|---|---|
| `run_baseline.ps1` | 同协议自跑基线（原始波形 1D-CNN / WDCNN / ResNet1D），4 工况 | `artifacts/variant_scan/` |
| `run_energy.ps1` | 窗能量对照（两种参考口径） | `reports/energy_*.log`、`reports/energy_only_*.csv` |
| `run_energy_stats.ps1` | 能量统计扫描（齿轮 2 工况） | `artifacts/variant_scan_energy/` |
| `run_fill_gear200.ps1` | 补齿轮 20-0 的加噪点 | `artifacts/variant_scan_fill/` |
| `run_lag_spacing.ps1` | 滞后取样对照（图 5） | `artifacts/variant_scan_lagspace/` |
| `run_mixraw.ps1` | 混合表示对照（结论：未采用） | `artifacts/variant_scan_mixraw/` |
| `run_noise_global.ps1` / `_low.ps1` / `_global_high.ps1` | 全局参考功率下的噪声曲线（高/低 SNR 分段） | `artifacts/variant_scan*/` |
| `run_noise_other3.ps1` | 补其余 3 个工况的噪声曲线 | `artifacts/variant_scan*/` |
| `run_noise_restore.ps1` | 恢复/补齐 4 工况噪声曲线 | `artifacts/variant_scan_noise_restore/` |
| `speed_wrap.py` | 速度基准包装（限 torch 线程后跑 `src/tuning/run_tuning.py`） | `artifacts/tuning/` |

## 规则

1. **脚本从自己的位置解析项目根**（`Split-Path $PSScriptRoot -Parent` / `Path(__file__).parents[1]`），
   不许写绝对盘符路径 —— 换了盘符/目录就静默读错地方。
2. 跑批完成后按 `CHECKLIST.md` 的完成判据确认（进程归零 + 日志出现"完成 N/M"），
   再跑 `python src/reporting/make_registry_scans.py` 更新 `registry_scans.csv`。