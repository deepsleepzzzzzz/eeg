# 当前状态

分支：`feat/rotating-fault`。只做东南大学旋转机械分类。脑电在标签 `eeg-cnn`、`eeg-gcn`、`rp-mrp-baseline`。

主线、协议、命令：`doc/method.md`。数字：`doc/results.md`。

## 待办

- [x] 轴承/齿轮两套流程已接，CNN/GCN 配置可切。冒烟通。见 `doc/method.md`
- [x] 正式第一版已训完（种子 42，test 封闭）。见 `doc/results.md`
- [x] 验证集混淆矩阵和 t-SNE 已接到训练；论文图在 `doc/figures/seu/`。见 `doc/results.md`
- [x] 四条线 val 扫 lr / dropout。见 `doc/results.md`
- [ ] 打开测试集
- [ ] 手选指标仍是对照行
