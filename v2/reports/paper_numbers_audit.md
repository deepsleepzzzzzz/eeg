# 论文数字溯源审计（主协议 main_70_15_15_v5）

## A/B 每个来源目录的 run 归属与协议

- ✅ artifacts/main_70_15_15_v5 的 40 个 run 都在本项目内
- ✅ artifacts/main_70_15_15_v5 不来自 legacy/ 归档
- ✅ artifacts/main_70_15_15_v5 全部为当前主协议 main_70_15_15_v5 —— 40 个 run
- ✅ artifacts/variant_scan 的 822 个 run 都在本项目内
- ✅ artifacts/variant_scan 不来自 legacy/ 归档
- ✅ artifacts/variant_scan 全部为当前主协议 main_70_15_15_v5 —— 822 个 run
- ✅ artifacts/variant_scan_rep_v5 的 60 个 run 都在本项目内
- ✅ artifacts/variant_scan_rep_v5 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_rep_v5 全部为当前主协议 main_70_15_15_v5 —— 60 个 run
- ✅ artifacts/variant_scan_noise_v5 的 60 个 run 都在本项目内
- ✅ artifacts/variant_scan_noise_v5 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_noise_v5 全部为当前主协议 main_70_15_15_v5 —— 60 个 run
- ✅ artifacts/variant_scan_noise_restore 的 144 个 run 都在本项目内
- ✅ artifacts/variant_scan_noise_restore 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_noise_restore 全部为当前主协议 main_70_15_15_v5 —— 144 个 run
- ✅ artifacts/variant_scan_fill 的 12 个 run 都在本项目内
- ✅ artifacts/variant_scan_fill 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_fill 全部为当前主协议 main_70_15_15_v5 —— 12 个 run
- ✅ artifacts/variant_scan_energy 的 12 个 run 都在本项目内
- ✅ artifacts/variant_scan_energy 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_energy 全部为当前主协议 main_70_15_15_v5 —— 12 个 run
- ✅ artifacts/variant_scan_lagspace 的 18 个 run 都在本项目内
- ✅ artifacts/variant_scan_lagspace 不来自 legacy/ 归档
- ✅ artifacts/variant_scan_lagspace 全部为当前主协议 main_70_15_15_v5 —— 18 个 run
- ✅ artifacts/cross_condition 的 15 个 run 都在本项目内
- ✅ artifacts/cross_condition 不来自 legacy/ 归档
- ✅ artifacts/cross_condition 全部为当前主协议 main_70_15_15_v5 —— 15 个 run

## C 主结果表（表 4）的 run 必须齐全

- ✅ 表 4 的 40 个 run 齐全（4 工况 × 2 模型 × 5 seed） —— 40/40

## D 表 8 依赖的加噪 run 必须齐全

- ✅ 齿轮 20-0 的 8 档 × 2 模型 × 3 seed 加噪 run 齐全 —— 48/48
- ✅ 齿轮 30-2 的 8 档 × 2 模型 × 3 seed 加噪 run 齐全 —— 48/48
- ✅ 轴承 20-0 的 8 档 × 2 模型 × 3 seed 加噪 run 齐全 —— 48/48
- ✅ 轴承 30-2 的 8 档 × 2 模型 × 3 seed 加噪 run 齐全 —— 48/48

## E 原始数据在本项目内且与官方清单一致

- ✅ 原始数据清单存在 —— data\seu\_files.json
- ✅ 20 个原始 CSV 与官方 git blob sha 一致

## 汇总

- 检查项：34 通过 / 0 失败
- 论文数字涉及的 run 总数：1183
- 协议分布：{'main_70_15_15_v5': 1183}
- 期望协议：main_70_15_15_v5（version v5）
