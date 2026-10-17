"""SEU 传动系统动力学模拟器（DDS）数据集的类别名 —— 全项目唯一真源。

原始数据的类别名就是文件名本身（cathysiyu/Mechanical-datasets 的 gearbox 目录）：
    gearset/    Health_<cond>.csv  Chipped_<cond>.csv  Miss_<cond>.csv  Root_<cond>.csv  Surface_<cond>.csv
    bearingset/ health_<cond>.csv  ball_<cond>.csv    inner_<cond>.csv  outer_<cond>.csv  comb_<cond>.csv
    <cond> ∈ {20_0, 30_2}

为什么单独建一个文件：这些中文名过去散落在各个画图脚本里，同一个英文类名
在不同图里出现了不同译名（例如 Chipped 在流程图脚本里写"齿面剥落"、在总流程
图里写"齿面崩裂"；Health 一处写"正常"、一处写"健康"）。读者因此无法判断
哪张图的哪些类是齿轮数据、哪些是轴承数据 —— 这正是"最后的结果是不是轴
承的"这类疑问的来源。凡是要写类别中文名的地方，一律从这里取。
"""
from __future__ import annotations

# 齿轮任务（原始目录 gearset/）：故障在齿轮上
GEAR_CN: dict[str, str] = {
    "Health": "健康",
    "Chipped": "齿面剥落",
    "Miss": "缺齿",
    "Root": "齿根裂纹",
    "Surface": "齿面磨损",
}

# 轴承任务（原始目录 bearingset/）：故障在轴承上
BEARING_CN: dict[str, str] = {
    "health": "健康",
    "ball": "滚动体",
    "inner": "内圈",
    "outer": "外圈",
    "comb": "保持架（组合）",
}

TASK_CN: dict[str, str] = {"gear": "齿轮", "bearing": "轴承"}
TASK_EN: dict[str, str] = {"gear": "Gear", "bearing": "Bearing"}

#: 每个任务的**官方原始文件名前缀**，用于论文里说明"类别取自哪个文件"
TASK_SOURCE_DIR: dict[str, str] = {"gear": "gearset", "bearing": "bearingset"}


def cn_map(task: str) -> dict[str, str]:
    """返回该任务的 {英文类名: 中文名}。任务名不认识时直接报错，不设默认。"""
    key = str(task).lower()
    if key == "gear":
        return dict(GEAR_CN)
    if key == "bearing":
        return dict(BEARING_CN)
    raise ValueError(f"未知任务: {task}（只能是 gear / bearing）")


def cn_name(task: str, name: str) -> str:
    table = cn_map(task)
    if name not in table:
        raise KeyError(f"{task} 任务没有类别 {name!r}（已有 {sorted(table)}）")
    return table[name]


def cn_names(task: str, names) -> list[str]:
    return [cn_name(task, n) for n in names]


def labeled(task: str, name: str) -> str:
    """图里用的双向标签，例如 "齿面剥落（Chipped）"：中英都给，杜绝歧义。"""
    return f"{cn_name(task, name)}（{name}）"