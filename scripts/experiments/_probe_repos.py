import json, os, re, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url, tries=6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=45))
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


REPOS = [
    "fonderxu/Bearing-fault-diagnosis-datasets",
    "shayanMoodi/CWRU_BearingDataset",
    "MASTER111363/Cross-domain-bearing-fault-diagnosis-on-CWRU-dataset",
    "Gearboxdata/Gear-Box-Fault-Diagnosis-Data-Set",
    "Final-Year-Projects-KEC/GIT-Gear-Fault-Detection--Dataset-and-Other-Files-",
    "liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets",
    "CH-0909/UM-Gearbox-Dataset",
]

IMG = re.compile(r"\.(png|jpe?g|bmp|webp|gif)$", re.I)

for repo in REPOS:
    try:
        d = get_json(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
        imgs = [t["path"] for t in d.get("tree", []) if IMG.search(t["path"])]
        print(f"\n### {repo}  (imgs={len(imgs)})")
        for p in imgs[:25]:
            print("   ", p)
    except Exception as e:
        print(f"\n### {repo}  FAIL {e}")
