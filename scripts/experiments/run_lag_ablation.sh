#!/usr/bin/env bash
# 导师问题 1/2：延迟步数上限 & 是否需要间隔
# 固定 L=1600/stride=800，只变 transition_steps。val only，测试集封闭。
set -u
cd "C:/Users/lgt11/Desktop/EEG/EEG"
PY="C:/miniconda/envs/eeg/python.exe"

VARIANTS="zq_s1_w1600_over zq_s3_w1600_over zq_s5_w1600_over zq_s7_w1600_over \
zq_s9_w1600_over zq_s12_w1600_over zq_s15_w1600_over \
zq_s135_w1600_over zq_s1357_w1600_over zq_s13579_w1600_over"

for T in bearing gear; do
  for v in $VARIANTS; do
    echo "########## $T 30_2 $v ##########"
    "$PY" scripts/experiments/tune_seu.py --task "$T" --condition 30_2 \
      --archs cnn --variant "$v" --max-trials 1 --lr 5e-4 --dropout 0.2 || echo "FAILED $T $v"
  done
done
echo "ALL DONE"
