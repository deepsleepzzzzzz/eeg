# datas/

- `seu/`：东南大学 DDS 原始 csv，不提交。
- `experiments_seu/`：切窗后的转移矩阵和权重，不提交。

脑电 `ds002680/` 与 `experiments_3way/` 不在本分支使用。

## 2026-10-15 空间清理

- 将根目录旧 `EEG.zip` 替换为 `EEG_review_snapshot.zip`（149,426,503 字节，约 143 MiB）。小快照保留原包源码、配置、文档、SEU 缓存/权重/指标，以及旧脑电结果和权重；去掉脑电原始数据、旧脑电数组缓存、重复 Git 历史和交接副本。
- 从原包恢复 `datas/seu/` 的 20 个原始 CSV 及 4 个说明文件，逐文件校验大小和 CRC。原先本地 SEU 目录为空；恢复后当前任务可直接访问原始数据，不依赖旧大包。未恢复嵌套数据仓库的 `.git/`。
- 删除旧 `EEG.zip`，清理 `experiments_3way/`、`artifacts/` 以及 `outputs/ablation/representations/`、`outputs/ablation_scoping/representations/` 中共 984 个可重建 `.npy` 数组，并移除 176 个对应的 `meta.json` 完成标记。再次运行这些历史实验时需要重新构建缓存。
- 保留源码、Python 环境、指标、预测、图、报告、所有权重及其他审阅输出。用户已明确授权永久删除约 13.39 GiB 的旧脑电原始目录 `ds002680/`，但针对该固定绝对路径的删除仍被执行环境自动审批拦截，仅返回 `blocked by policy`。目录尚未删除，不再需要用户重复授权；须在本机 PowerShell 手动执行 `Remove-Item -LiteralPath 'D:\project\Python\EEG\datas\ds002680' -Recurse -Force` 才能释放这部分空间。

实际净释放约 11.44 GiB，D 盘可用空间由约 52.17 GiB 增至 63.60 GiB。小快照通过完整 ZIP CRC 检查，并通过历史缺陷复现 3/3 及修复覆盖层回归 8/8；测试脚本默认优先选用小快照，命令和实验记录见 `doc/results.md`。

本机逐项计划、执行状态与容量实测记录位于 `outputs/cleanup_20261015/manifest.json`，它是本次清理的审计产物，不是实验结果来源。
