# 配置键

当前主线配置是 `config/seu_bearing_cnn.yaml`、`config/seu_gear_cnn.yaml`。GCN 为同路径的 `*_gcn.yaml`。协议和命令见 `doc/method.md`。

脑电配置不在本分支，见标签 `eeg-cnn`、`eeg-gcn`。

| 段 | 项 | 含义 |
|----|----|------|
| method | representation | 本分支为 `transition` |
| | attention | `none` / `se` / `additive` / `self`。仅 CNN。`self` 在卷积前 |
| | normalize | 逐窗归一化方式；当前正式协议见 `doc/method.md` |
| | symbol_bins | 幅值状态/节点数量 |
| | bin_scope | `train_global`：边界只由训练集拟合 |
| | transition_steps | 如 `[1,2,3]` |
| | transition_weight | `probability` |
| | transition_multichannel | `stack` |
| model | arch | `cnn` 或 `gcn`，互斥 |
| | gcn_hidden / gcn_layers / gcn_pool / gcn_layout / gcn_steps / gcn_attention | 仅 GCN |
| | conv_channels | 仅 CNN |
| data | task | `bearing` 或 `gear` |
| | raw_root | 原始 SEU 目录，默认 `./datas/seu` |
| | output_root | 转移矩阵缓存 |
| | conditions | `["20_0","30_2"]`，须加引号，否则 YAML 会把 `20_0` 读成整数 |
| | epoch_samples / window_stride | 窗长与步长；当前正式值见 `doc/method.md` |
| | max_windows_per_class | 冒烟截断；`null` 为全量 |
| | train_ratio / val_ratio / test_ratio | 按时间切开，三个正比例之和必须为 1；当前正式值见 `doc/method.md` |
| train | device / require_gpu | `cuda` 可选；无 GPU 且 `require_gpu=false` 则走 CPU |
| train | lr / dropout | 扫参网格见 `doc/method.md`。第一版为 0.0005 / 0.2 |
| | seeds | 训练随机种子，与时间划分无关 |
