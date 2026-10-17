"""数据溯源测试：证明每个任务-工况的缓存来自它声明的原始 CSV（齿轮↔轴承不许串）。

三件事一旦写错，指标看上去仍然"正常"，只有从原始 CSV 反算才能发现：
    1) 路径/映射写错 → 齿轮任务读成轴承文件；
    2) 归一化统计量或分位边界被换过 → 表示内容悄悄变了；
    3) 缓存数组与原始文件不对应 → 结果不可复现。

审计自带反证（D 项：换成另一套数据集必须对不上），所以它不可能是"永远通过"。
"""
import pytest

from audit_provenance import audit_one, find_main_dataset_dir, find_main_protocol
from config.protocol import load_protocol


@pytest.mark.parametrize("task,cond", [("gear", "20_0"), ("bearing", "20_0")])
def test_cache_comes_from_declared_raw_files(task, cond):
    protocol = load_protocol(find_main_protocol())
    r = audit_one(protocol, find_main_dataset_dir(protocol), task, cond)

    assert r["A_normalization"]["pass"], "归一化统计量与原始 CSV 不符：" + str(r["A_normalization"])
    assert r["B_bin_edges"]["pass"], "分位边界与原始 CSV 不符：" + str(r["B_bin_edges"])
    assert r["C_end_to_end"]["pass"], "缓存数组无法由原始 CSV 重建：" + str(r["C_end_to_end"])
    assert r["D_cross_dataset_negative_control"]["pass"], "反证失败：换另一套数据集竟然也对得上"
    assert r["E_official_file_list"]["pass"], "本地 CSV 与官方清单 sha 不符：" + str(r["E_official_file_list"])
    assert r["F_split_hash"]["pass"], "划分指纹对不上且无法解释：" + str(r["F_split_hash"])


def test_gear_and_bearing_class_names_are_disjoint():
    """齿轮与轴承的类别名必须来自各自的原始目录，不能互相污染。"""
    from data.seu_dds import class_names

    assert set(class_names("gear")) == {"Health", "Chipped", "Miss", "Root", "Surface"}
    assert set(class_names("bearing")) == {"health", "ball", "inner", "outer", "comb"}
    assert not (set(class_names("gear")) & set(class_names("bearing")))