"""CLI: pull files from GitHub via API (contents -> fallback blobs).

Usage:
  python gh_pull.py <owner/repo> <outdir> <path1> [path2 ...]
"""
import json, base64, os, sys, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def gj(url, tries=8):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def blob(repo, sha, tries=8):
    last = None
    for i in range(tries):
        try:
            d = gj(f"https://api.github.com/repos/{repo}/git/blobs/{sha}")
            return base64.b64decode(d["content"])
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch(repo, path, outdir, outname=None):
    outname = outname or os.path.basename(path)
    dst = os.path.join(outdir, outname)
    info = gj(f"https://api.github.com/repos/{repo}/contents/{path}")
    if isinstance(info, list):
        raise RuntimeError(f"{path} is a directory")
    size = info.get("size", 0)
    if size and size <= 1_000_000 and info.get("content"):
        b = base64.b64decode(info["content"])
    else:
        b = blob(repo, info["sha"])
    open(dst, "wb").write(b)
    print(f"OK  {outname:34s} {len(b):>10,} bytes")
    return dst


if __name__ == "__main__":
    repo, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    for p in sys.argv[3:]:
        try:
            fetch(repo, p, outdir)
        except Exception as e:
            print(f"FAIL {p}: {type(e).__name__} {e}")
