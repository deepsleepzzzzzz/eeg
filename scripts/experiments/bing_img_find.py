"""Search Bing Images (HTML endpoint) and report candidate image URLs grouped by host.

Usage: python bing_img_find.py "query1" "query2" ...
"""
import re, sys, time, urllib.parse, urllib.request, collections

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEAD = {"User-Agent": UA, "Referer": "https://cn.bing.com/",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8"}

# hosts proven reachable from this sandbox (can actually download binaries)
GOOD = ("pub.mdpi-res.com", "ai2-s2-public.s3.amazonaws.com", "static-01.extrica.com",
        "mm.bing.net", "media.springernature.com", "cdn.ncbi.nlm.nih.gov",
        "data.mendeley.com", "arxiv.org", "static-content.springer.com")


def search(q, first=1):
    url = ("https://cn.bing.com/images/search?q=" + urllib.parse.quote(q)
           + f"&form=HDRSC2&first={first}")
    req = urllib.request.Request(url, headers=HEAD)
    html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
    return re.findall(r'murl&quot;:&quot;(.+?)&quot;', html)


if __name__ == "__main__":
    for q in sys.argv[1:]:
        try:
            urls = search(q)
        except Exception as e:
            print(f"### {q}: FAIL {e}")
            continue
        good, other = [], collections.Counter()
        for u in urls:
            h = urllib.parse.urlparse(u).netloc
            other[h] += 1
            if any(g in h for g in GOOD):
                good.append(u)
        print(f"\n### {q}   (total {len(urls)})")
        for u in good:
            print("   GOOD:", u[:150])
        print("   hosts:", dict(other))
        time.sleep(1)
