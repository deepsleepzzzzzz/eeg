"""修改前的文件备份 —— 项目硬规则的一部分（见 AGENTS.md 第九节）。

规则
----
对**交付物**（`reports/*.md`、`reports/*.png`、`configs/**/*.yaml` 以及论文正文）
的任何修改，必须在**写入之前**先用本工具留一份副本；改完连同备份一起 git 提交。
代码由"每完成一件事就提交一次"的纪律保护（git 自带版本），但交付物改动频繁、
影响面大，额外留物理副本，随时能肉眼对照。

用法
----
    # 改之前：备份当前内容
    python src/utils/backup_file.py <tag> <文件> [<文件>...]

    # 补录：备份某个历史版本（例如昨天那次改动之前的版本）
    python src/utils/backup_file.py <tag> --from-git <git-rev> <文件>...

备份命名（放在项目根的 backups/）：
    论文初稿_v1.backup_before_<tag>_<YYYYMMDD>.md
同名冲突时自动加 -2、-3。每次备份都会往 backups/README.md 追加一行台账。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = PROJECT_ROOT / "backups"
GIT = r"C:\Users\liang\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe"


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        sys.stdout.write(msg.encode("ascii", "backslashreplace").decode("ascii") + "\n")


def _dest_name(src: Path, tag: str, day: str) -> str:
    stem, suffix = src.stem, src.suffix
    base = f"{stem}.backup_before_{tag}_{day}{suffix}"
    dest = BACKUP_DIR / base
    k = 2
    while dest.exists():
        dest = BACKUP_DIR / f"{stem}.backup_before_{tag}_{day}-{k}{suffix}"
        k += 1
    return dest.name


def backup(paths, tag: str, from_git: str | None = None, note: str = "") -> list[Path]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    made: list[Path] = []
    for raw in paths:
        src = Path(raw)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        if not src.exists():
            raise FileNotFoundError(f"源文件不存在: {src}")
        if from_git:
            rel = src.relative_to(PROJECT_ROOT).as_posix()
            data = subprocess.run(
                [GIT, "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
                 "-C", str(PROJECT_ROOT), "show", f"{from_git}:{rel}"],
                capture_output=True)
            if data.returncode != 0:
                raise RuntimeError(f"git show 失败: {from_git}:{rel}\n{data.stderr.decode('utf-8', 'replace')}")
            blob = data.stdout
            # 仓库里存的是 LF，而工作树是 CRLF（core.autocrlf=true）。
            # 备份必须与「改动前的工作树文件」逐字节可比，所以按源文件当前的换行风格还原。
            style = b"\r\n" if (src.exists() and b"\r\n" in src.read_bytes()[:4000]) else b"\n"
            if style == b"\r\n" and b"\r\n" not in blob:
                blob = blob.replace(b"\n", b"\r\n")
        else:
            blob = src.read_bytes()
        dest = BACKUP_DIR / _dest_name(src, tag, day)
        dest.write_bytes(blob)
        made.append(dest)
        origin = f"git {from_git}" if from_git else "工作区当前内容"
        rel_src = src.relative_to(PROJECT_ROOT).as_posix()
        line = (f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {tag} | `{rel_src}` | "
                f"`backups/{dest.name}` | {origin} | {note or '-'} |")
        readme = BACKUP_DIR / "README.md"
        if not readme.exists():
            readme.write_text(
                "# 交付物修改备份台账\n\n"
                "由 `src/utils/backup_file.py` 自动记录。**改交付物之前先备份**（见 AGENTS.md 第九节）。\n\n"
                "| 时间 | tag | 源文件 | 备份文件 | 来源 | 说明 |\n"
                "|---|---|---|---|---|---|\n",
                encoding="utf-8", newline="")
        with open(readme, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")
        _safe_print(f"[backup] {rel_src} -> backups/{dest.name}")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--from-git", default=None, help="从该 git 版本取内容（补录历史版本用）")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    backup(a.paths, a.tag, a.from_git, a.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())