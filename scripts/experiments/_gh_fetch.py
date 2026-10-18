"""Robust GitHub fetcher for fault-specimen photos (API-only, no raw.githubusercontent)."""
import json, base64, os, time, urllib.request, sys

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SPENT = {"n": 0}


def gj(url, tries=8):
    last = None
    for i in range(tries):
        try:
            SPENT["n"] += 1
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def tree(repo):
    d = gj(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
    return d.get("tree", [])


def blob_by_sha(repo, sha, tries=8):
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
    """Fetch one file: use contents API when small, blobs when >1MB."""
    outname = outname or os.path.basename(path)
    dst = os.path.join(outdir, outname)
    info = gj(f"https://api.github.com/repos/{repo}/contents/{path}")
    size = info.get("size", 0)
    if size and size <= 1_000_000 and info.get("content"):
        b = base64.b64decode(info["content"])
    else:
        b = blob_by_sha(repo, info["sha"])
    open(dst, "wb").write(b)
    print(f"OK  {outname:28s} {len(b):>9,} bytes  <- {repo}/{path}")
    return dst


if __name__ == "__main__":
    OUT = r"C:\Users\lgt11\Desktop\EEG\EEG\doc\figures\seu\teacher\fault_photos\_probe"
    os.makedirs(OUT, exist_ok=True)

    # 1) HUST bearing dataset (inner/outer/ball/combined) — original figures
    for f in ["F1.png", "F2.png", "F3.png", "F4.jpg"]:
        try:
            fetch("CHAOZHAO-1/HUSTbearing-dataset", f"IMG/{f}", OUT)
        except Exception as e:
            print("FAIL", f, type(e).__name__, e)

    # 2) KEC gear-fault repo sample images (check what they are)
    for i in [1, 2, 3]:
        p = f"Deployment/Browser Version/Test Images/{i}.jpg"
        try:
            fetch("Final-Year-Projects-KEC/GIT-Gear-Fault-Detection--Dataset-and-Other-Files-", p, OUT, f"kec_{i}.jpg")
        except Exception as e:
            print("FAIL kec", i, type(e).__name__, e)

    print("\nAPI calls spent:", SPENT["n"])
