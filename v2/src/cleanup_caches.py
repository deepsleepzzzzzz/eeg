"""清理与当前主协议无关的历史数据缓存（可重建，删掉不影响任何已有结果）。

默认只做 dry-run；确认后用 --apply 真正删除。

安全设计：
  * 只处理 artifacts/var_* 目录
  * 逐个校验绝对路径在 artifacts 之下、目录名匹配 var_<12位十六进制>
  * 只删内容【仅有 dataset/】的目录；含别的文件一律跳过并在报告中列出
  * 判据按 **(task, condition) 逐个比对**：目录里只要**有一个**组合的 split_hash
    与当前主协议同组合的 split_hash 相同，就说明它还能被当前协议复用 → 保留。
    只有**所有组合都对不上**才算旧协议缓存 → 可删。

    2026-10-16 修的坑：原判据只拿 `--task/--condition`（默认 gear/30_2）**一个**指纹去比，
    而 var_* 目录是按"数据相关配置"哈希命名的，一个目录里可能同时装着
    20-0 与 30-2、齿轮与轴承；于是装着 gear/20_0 的当前协议缓存会被误判成"旧协议"。
    实测差距：错误判据报 14 个可删（0.58 GB），正确判据只有 5 个（0.11 GB）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正删除（默认仅列出）")
    ap.add_argument("--task", default="gear")
    ap.add_argument("--condition", default="30_2")
    args = ap.parse_args()

    from config.protocol import dataset_dir, load_protocol
    proto_path = None
    for f in sorted((ROOT / "configs" / "protocols").glob("main_*.yaml")):
        if str(load_protocol(f).get("status")) == "main":
            proto_path = f
    if proto_path is None:
        raise RuntimeError("找不到 status=main 的主协议")
    proto = load_protocol(proto_path)
    # 当前主协议**四个**任务-工况组合的指纹，逐个比对（见文件头的说明）
    combos = [(t, c) for t in ("gear", "bearing") for c in ("20_0", "30_2")]
    cur = {}
    for t_, c_ in combos:
        meta_path = dataset_dir(proto, t_, c_) / "meta.json"
        cur[(t_, c_)] = json.loads(meta_path.read_text(encoding="utf-8"))["split_hash"]

    art = ROOT / "artifacts"
    targets, kept, skipped, freed = [], 0, [], 0
    for d in sorted(art.glob("var_*")):
        if not d.is_dir():
            continue
        full = Path(str(d.resolve()))
        if full.parent != art.resolve() or not d.name.startswith("var_"):
            skipped.append((d.name, "路径校验未通过"))
            continue
        if any(p.name != "dataset" for p in d.iterdir()):
            skipped.append((d.name, "含非 dataset 内容"))
            continue
        seen = {}
        for mp in d.rglob("meta.json"):
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                continue
            seen[(m.get("task"), m.get("condition"))] = m.get("split_hash")
        if any(k in cur and h == cur[k] for k, h in seen.items()):
            kept += 1
            continue
        nbytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        targets.append((d, nbytes))
        freed += nbytes

    print(f"主协议 {proto['name']}  比对 {len(cur)} 个任务-工况组合")
    for (t_, c_), h in cur.items():
        print(f"    {t_}/{c_}: {h[:16]}")
    print(f"  与主协议一致（保留）: {kept}")
    print(f"  属于旧协议（可删）  : {len(targets)}   合计 {freed/1024**3:.2f} GB")
    print(f"  跳过（异常）        : {len(skipped)}")
    for n, why in skipped[:5]:
        print(f"      {n}: {why}")

    if not args.apply:
        print("\n这是 dry-run。确认后加 --apply 真正删除。")
        return
    for d, _ in targets:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n已删除 {len(targets)} 个缓存目录，释放约 {freed/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
