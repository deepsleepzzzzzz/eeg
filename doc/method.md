# 方法

数字只写在 `doc/results.md`。配置键只写在 `doc/config.md`。老师原话整理在 `doc/teacher_improvement_directions.md`。

本分支只做旋转机械故障诊断。脑电四分类不在这里跑，代码和数字在标签 `eeg-cnn`、`eeg-gcn`、`rp-mrp-baseline`。

## 现在做什么

东南大学 DDS：轴承一套模型，齿轮一套模型。幅值符号化 → 多时间步有向加权转移矩阵 → CNN 或 GCN（`model.arch`）。五类 Softmax，交叉熵训练。

## 协议

- 数据：东南大学 Drivetrain Dynamic Simulator。原始文件在 `datas/seu/`
- 两套任务，互不拿去验对方：轴承五类（health / ball / inner / outer / comb），齿轮五类（Health / Chipped / Miss / Root / Surface）
- 工况 `20_0`、`30_2` 分训分测，再报两个准确率的平均
- 通道：行星齿轮箱 2、3、4（x/y/z）。不用转矩和其余振动
- 窗：1600 点，步长 800（50% 重叠）。先按时间 70/20/10 切开，再切窗
- 归一化：`normalize=zscore`，逐窗 z-score，消除类间幅值尺度差
- 分档：`bin_scope=train_global`，只用该任务该工况的训练集；`bin_edges_mode=quantile` 等频边界
- 转移：步长 `[1,2,3,4,5,6,7]`，概率权重，三通道堆叠，输入 `[21,6,6]`
- 训练种子默认 42；多种子验证用 `42,43,44`。按 `val_acc` 早停
- test 默认封闭。评测须同时加 `--evaluate-test --confirm-test`
- 四条对照：CNN 关注意力 / CNN 自注意力 / GCN 关图注意力 / GCN 开图注意力。表示和划分冻第一版。各自扫学习率 `{1e-4, 5e-4, 1e-3}` 和 dropout `{0.1, 0.2, 0.3}`。选参看两工况 val 平均。扫参目录不覆盖第一版权重。

## 表示升级（改进版）

第一版在轴承 20-0（幅值塌缩）和齿轮 30-2（Miss/Surface 混淆）上不足 0.95。诊断后把表示层升级为：逐窗 z-score + 等频分档 + 7 步转移 + 1600 点重叠窗。四线全组合验证集/测试集均 ≥0.95，详见 `doc/results.md` 的「调参改进版」。调参过程见 `scripts/experiments/tune_seu.py`。

## 命令

```bash
python scripts/experiments/build_seu_dataset.py --config config/seu_smoke_bearing_cnn.yaml --force
python scripts/experiments/train_seu.py --config config/seu_smoke_bearing_cnn.yaml

python scripts/experiments/build_seu_dataset.py --config config/seu_bearing_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_bearing_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_bearing_gcn.yaml

python scripts/experiments/build_seu_dataset.py --config config/seu_gear_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_gear_cnn.yaml
python scripts/experiments/train_seu.py --config config/seu_gear_gcn.yaml

# 已有权重出验证集混淆矩阵和 t-SNE，不打开测试集
python scripts/experiments/plot_seu_val.py

# 四条线扫 lr / dropout，只看 val
python scripts/experiments/sweep_seu.py
```

训练结束后也会把图写到 `doc/figures/seu/`。t-SNE 用分类头之前的向量。

## 缓存一致性与论文图命名

缓存校验由 `data/seu_cache.py` 统一维护。构建器复用前检查缓存格式版本、表示配置、窗口/划分配置、原始数据目录、原始 CSV 内容摘要及六个数组文件的摘要。配置不同、来源变化、缓存不完整或旧格式缺少校验信息时，报错并要求使用原构建命令加 `--force` 显式重建。更换分类器、注意力、学习率或训练种子仍可共用同一表示缓存。修改表示实现而影响缓存语义时，需要同时递增缓存格式版本。

训练入口同样检查缓存配置及所用数组，默认只校验 train/val，不读取 test 数组；显式开启测试评测才校验 test。原始文件变化由构建器检测，训练使用缓存中记录的数据快照。重建开始写数组前撤销旧完成标记，只有全部写完后才保存新元数据。

`split_sample_id_fingerprint` 由任务、工况、通道、类别顺序、原始文件摘要、窗长及各类别各划分的实际窗口起点生成。完整缓存身份另覆盖表示配置和数组摘要，并写入新训练的权重及指标；重新出图要求数据、权重、指标的缓存身份一致。旧缓存必须重建，缺少缓存身份的旧权重/指标须重新训练生成后才可通过新出图入口；不能给历史结果直接补上新身份以代替验证。

默认种子 42 的基线图保留 `任务_cnn_工况` / `任务_gcn_工况` 名称。其他方法使用完整方法标识，例如 `bearing_transition_self_cnn_20_0_confusion.png`、`bearing_transition_gcn_attn_20_0_confusion.png`；其他种子追加 `_seed43` 等后缀。同一方法、工况、种子再次出图会更新自己的图，不覆盖其他方法或种子的图。

修复回归和原包复现入口为 `scripts/experiments/test_package_review.py`，命令、证据和测试边界见 `doc/results.md` 的包内审查记录。
