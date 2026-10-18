import json, base64, os, urllib.request

OUT = r"C:\Users\lgt11\Desktop\EEG\EEG\doc\figures\seu\teacher\fault_photos\_probe"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=45))


def fetch_api_file(repo, path, outname):
    d = get_json(f"https://api.github.com/repos/{repo}/contents/{path}")
    b = base64.b64decode(d["content"])
    p = os.path.join(OUT, outname)
    open(p, "wb").write(b)
    print(f"OK {outname} {len(b)} bytes  <- {repo}/{path}")


# HUST bearing dataset images
for f in ["F1.png", "F2.png", "F3.png", "F4.jpg"]:
    try:
        fetch_api_file("CHAOZHAO-1/HUSTbearing-dataset", f"IMG/{f}", f)
    except Exception as e:
        print("FAIL", f, e)
