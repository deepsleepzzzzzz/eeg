"""守卫：加噪参考功率口径（noise.reference）必须真实生效。

真实踩过的坑（2026-10-15）：
  add_noise 历史上按【每个窗自己的功率】缩放噪声，于是
      加噪窗能量 = 干净窗能量 × 常数
  这意味着"窗能量"在任何 SNR 下都保持类别可分 —— 一个与表示无关的捷径。
  实测后果：齿轮 20-0 在 -10 dB 以下，CNN/GCN 的准确率与"纯用窗口能量 +
  最近类中心"完全相同（84.4% vs 84.0%），即表示没有提供任何额外信息。

  修正：noise.reference=global 时，所有窗共用同一个参考功率（由训练集全部窗算得）。
  本测试锁死两种口径的差异，防止有人把 global 静默改回 per_window。
"""
import sys

import numpy as np

from _bootstrap import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.seu_dds import add_noise  # noqa: E402


def _windows():
    rng = np.random.default_rng(0)
    # 4 个窗，功率依次相差 10 倍 —— 模拟不同故障类别的振动强弱差异
    return [(rng.standard_normal((3, 2048)) * 10 ** (i / 3)).astype(np.float64) for i in range(4)]


def _pooled_energy(windows):
    return np.array([float(np.mean(np.asarray(w) ** 2)) for w in windows])


def test_per_window_reference_preserves_energy_ratio():
    """per_window 口径下，加噪后各窗能量比值 = 干净能量比值（这就是那个捷径）。"""
    ws = _windows()
    clean = _pooled_energy(ws)
    noisy = _pooled_energy(add_noise(ws, -10.0, 1))
    np.testing.assert_allclose(noisy / clean, noisy[0] / clean[0], rtol=0.05)


def test_global_reference_achieves_target_snr():
    """global 口径下，实测噪声功率与参考功率之比 = 10**(-snr/10)。"""
    ws = _windows()
    p_ref = float(np.mean(np.concatenate([w.ravel() for w in ws]) ** 2))
    for snr in (-4.0, -10.0):
        noisy = _pooled_energy(add_noise(ws, snr, 1, p_ref=p_ref))
        clean = _pooled_energy(ws)
        measured = float(np.mean(noisy - clean)) / p_ref
        np.testing.assert_allclose(measured, 10 ** (-snr / 10.0), rtol=0.1)
    # 噪声地板统一：只有地板真正主导时才压平，-10 dB 下各窗能量比应远小于干净时的 100 倍
    clean = _pooled_energy(ws)
    noisy = _pooled_energy(add_noise(ws, -10.0, 1, p_ref=p_ref))
    assert float(clean.max() / clean.min()) > 50.0, "构造的窗能量差异不够大，测试无意义"
    assert float(noisy.max() / noisy.min()) < 1.3, "global 口径下噪声地板应当统一"
    # 反向对照：per_window 口径下能量比必须完全保住（这正是要禁掉的那个捷径）
    per = _pooled_energy(add_noise(ws, -10.0, 1))
    assert float(per.max() / per.min()) > 50.0


def test_two_references_differ():
    """两种口径必须给出不同的加噪结果，否则说明参数没被真正读进去。"""
    ws = _windows()
    p_ref = float(np.mean(np.concatenate([w.ravel() for w in ws]) ** 2))
    a = add_noise(ws, -10.0, 1)
    b = add_noise(ws, -10.0, 1, p_ref=p_ref)
    assert any(not np.allclose(x, y) for x, y in zip(a, b))
