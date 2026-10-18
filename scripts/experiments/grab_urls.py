"""Download a list of image URLs (one per line in a text file) into an output dir.

Usage: python grab_urls.py <urlfile> <outdir>
"""
import os, sys, urllib.request, urllib.parse, time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def dl(url, dst, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Referer": "https://cn.bing.com/"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if len(data) < 1000:
                raise RuntimeError(f"too small ({len(data)})")
            open(dst, "wb").write(data)
            return len(data)
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


if __name__ == "__main__":
    urlfile, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    urls = [l.strip() for l in open(urlfile, encoding="utf-8") if l.strip() and not l.startswith("#")]
    for n, u in enumerate(urls, 1):
        base = os.path.basename(urllib.parse.urlparse(u).path.split("?")[0])
        name = f"{n:02d}_{base}"
        try:
            sz = dl(u, os.path.join(outdir, name))
            print(f"OK  {name:52s} {sz:>9,}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__} {e}")
