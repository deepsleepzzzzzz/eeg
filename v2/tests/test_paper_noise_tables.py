"""论文正确性测试（二）：表 8 / 表 9 的加噪数字逐格核对。

背景（本次审计发现的真问题）：表 8 有 4 个任务-工况 × 8 档 SNR = 32 个数字，但其中 3 个工况
（齿轮 20-0、轴承 20-0、轴承 30-2）的加噪 run 当年被变体扫描**覆盖丢失**（AGENTS.md 记录的
variant_scan 覆盖事故），只有齿轮 30-2 的 run 还在 artifacts/variant_scan_noise_v5/ 下。
因此这些数字此前无法被逐格核对。已用同一协议重跑补齐：
    configs/scan/noise_restore_v5.yaml -> artifacts/variant_scan_noise_restore/
（同一协议、同一 seed，重跑值与当年报告逐位一致 —— 已用齿轮 20-0 的 −6 dB / −10 dB 验证过。）

本测试在数据就绪后逐格核对；若重跑尚未完成，会 skip 并给出明确原因（长期 skip 即说明数据未恢复）。
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "reports" / "论文初稿_v1.md"
RESTORE = PROJECT_ROOT / "artifacts/variant_scan_noise_restore"
EXISTING = PROJECT_ROOT / "artifacts/variant_scan_noise_v5"
SEED = (42, 43, 44)
ROWS = [("干净", "base"), ("+10", "snr10"), ("+4", "snr4"), ("0", "snr0"),
        ("−4", "snrm4"), ("−6", "snrm6"), ("−8", "snrm8"), ("−10", "snrm10")]
COMBOS = [("齿轮 20-0", "gear", "20_0"), ("齿轮 30-2", "gear", "30_2"),
          ("轴承 20-0", "bearing", "20_0"), ("轴承 30-2", "bearing", "30_2")]


def _scan_dir(task: str, cond: str) -> Path | None:
    """优先用补齐后的目录；齿轮 30-2 的数据在 variant_scan_noise_v5 里本来就还在。"""
    for d in (RESTORE, EXISTING):
        if (d / "base" / f"{task}_cnn_{cond}_base_seed42" / "metrics.json").exists():
            return d
    return None


def _value(scan: Path, tag: str, task: str, cond: str) -> float | None:
    vals = []
    for mdl in ("cnn", "gcn"):
        for s in SEED:
            p = scan / tag / f"{task}_{mdl}_{cond}_{tag}_seed{s}" / "metrics.json"
            if p.exists():
                v = (json.loads(p.read_text(encoding="utf-8")).get("val_metrics") or {}).get("accuracy")
                if v is not None:
                    vals.append(100 * float(v))
    return st.mean(vals) if vals else None


def _paper_tables() -> dict:
    L = PAPER.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    out, i = {}, 0
    while i < len(L):
        m = re.match(r"\*\*表 (\d+[ab]?)\s+(.*?)\*\*", L[i].strip())
        if m:
            rows, j = [], i + 1
            while j < len(L):
                if L[j].lstrip().startswith("|"):
                    rows.append([c.strip() for c in L[j].strip().strip("|").split("|")]); j += 1
                elif rows and not L[j].strip():
                    break
                elif not rows:
                    j += 1
                else:
                    break
            if rows:
                out[m.group(1)] = rows
            i = j
        else:
            i += 1
    return out


def _body(rows):
    return [r for r in rows if not set("".join(r)) <= set("-: ")][1:]


def _f(cell: str) -> float:
    """'**-13.93**' / '−10' -> float。"""
    return float(cell.replace("**", "").replace("−", "-").strip())


def _have_all(scan: Path, tag: str, task: str, cond: str) -> bool:
    """6 个 run（2 模型 × 3 seed）必须齐全，否则均值不完整、不能用来核对。"""
    return all((scan / tag / f"{task}_{m}_{cond}_{tag}_seed{s}" / "metrics.json").exists()
               for m in ("cnn", "gcn") for s in SEED)


@pytest.mark.parametrize("label,task,cond", COMBOS)
def test_table8_noise_accuracy_is_reproducible(label, task, cond):
    scan = _scan_dir(task, cond)
    if scan is None:
        pytest.skip(f"{label} 的加噪 run 尚未补齐（重跑未完成）；完成后本测试会自动执行")
    tab = _paper_tables()["8"]
    row_map = {r[0]: r for r in _body(tab)}   # _body 自己会去掉表头；此前写成 tab[1:] 会把“干净”行也删掉
    for row_label, tag in ROWS:
        if row_label not in row_map:
            continue
        col = 1 + [c[0] for c in COMBOS].index(label)
        want = _f(row_map[row_label][col])
        if not _have_all(scan, tag, task, cond):
            pytest.skip(f"{label} / {row_label} 的 run 尚未补齐（{tag}）")
        got = _value(scan, tag, task, cond)
        assert abs(want - got) < 0.005, f"表 8 {label} {row_label}: 论文 {want} vs 复算 {got:.4f}"


def test_table9_protocol_contrast_is_consistent():
    """表 9 的逐窗列必须与表 8 一致；差值列必须等于算术差。"""
    tabs = _paper_tables()
    t8 = {r[0]: r for r in _body(tabs["8"][1:])}
    for r in _body(tabs["9"]):
        label, snr = r[0], r[1]
        if snr in t8:                      # −12 / −16 只在表 9 出现（表 8 只到 −10）
            col = 1 + [c[0] for c in COMBOS].index(label)
            want8 = _f(t8[snr][col])
            assert abs(_f(r[2]) - want8) < 0.005, f"表 9 逐窗列 {label} {snr}: {r[2]} vs 表 8 {want8}"
        per, glob = _f(r[2]), _f(r[3])
        assert abs(_f(r[4]) - (glob - per)) < 0.005, f"表 9 差值列 {label} {snr}"
        assert 0 < _f(r[5]) <= 100, f"表 9 纯能量列 {label} {snr}"


def test_table6_this_work_row_matches_table8():
    """表 6 的 This work 行必须与表 8 的齿轮 20-0 列一致（−10 / −6 / 0 dB）。

    本检查会抓到"同一批实验在两张表里写成不同数字"这类不一致。
    由于齿轮 20-0 的加噪 run 曾被覆盖丢失，须等补齐后才能判定谁对谁错，故先按数据齐备与否 skip。
    """
    scan = _scan_dir("gear", "20_0")
    if scan is None or not all(_have_all(scan, tag, "gear", "20_0") for _, tag in ROWS):
        pytest.skip("齿轮 20-0 的加噪 run 尚未补齐；补齐后本检查自动生效")
    tabs = _paper_tables()
    t6 = [r for r in _body(tabs["6"]) if "This work" in r[0]]
    assert len(t6) == 1
    this = t6[0]
    t8 = {r[0]: r for r in _body(tabs["8"][1:])}
    for col, row_label in ((3, "−10"), (4, "−6"), (5, "0")):
        want6 = _f(this[col])
        want8 = _f(t8[row_label][1])
        assert abs(want6 - want8) < 0.05, \
            f"表 6 与表 8 不一致（{row_label} dB）：表 6 {want6} vs 表 8 {want8}"


def test_table9_energy_classifier_matches_energy_only_csv():
    """表 9 最后一列（纯窗能量分类器，逐窗参考）必须能追到 reports/energy_only_per_window.csv。

    此前这一列只做形式检查（assert 0 < x <= 100）—— 任何数字都能过。它的真实来源是
    `src/analysis/energy_only.py`：3 通道对数均方 + 最近类中心，不含任何学习。
    这里逐格核对，堵住"论文里的数追不到本项目产物"这一类问题。
    """
    csv_path = PROJECT_ROOT / "reports/energy_only_per_window.csv"
    assert csv_path.exists(), (
        f"缺少 {csv_path}（先跑 python src/analysis/energy_only.py --reference per_window）")
    src = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            src[(r["task"], r["condition"], float(r["snr_db"]))] = 100 * float(r["acc_3ch"])

    checked = 0
    for r in _body(_paper_tables()["9"]):
        label, snr = r[0], _f(r[1])
        task, cond = ("gear", "20_0") if "齿轮" in label else ("bearing", "20_0")
        key = (task, cond, snr)
        assert key in src, f"energy_only CSV 里没有 {key}（表 9 的 {label} {snr} dB）"
        want, got = _f(r[5]), src[key]
        assert abs(want - got) < 0.06, (
            f"表 9 纯能量列 {label} {snr} dB: 论文 {want} vs CSV {got:.2f}")
        checked += 1
    assert checked >= 7, f"表 9 只核对了 {checked} 行，行数不对"
