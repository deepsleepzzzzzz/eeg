"""Assemble the 'fault morphology photo' figure: 2 rows x 4 cols + sources.

Row 1 (gear):    Chipped / Miss / Surface / Root   (real photos)
Row 2 (bearing): Inner / Outer / Ball / Comb        (real photos, HUST)

All source figures are openly licensed (CC BY 4.0).
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

P = r"C:\Users\lgt11\Desktop\EEG\EEG\doc\figures\seu\teacher\fault_photos\_probe"
OUT = r"C:\Users\lgt11\Desktop\EEG\EEG\doc\figures\seu\teacher\fault_photos"


def F(sz, bold=False):
    for p in ([r"C:\Windows\Fonts\msyhbd.ttc"] if bold else []) + \
             [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def find_vlines(im, x_from, x_to, dark=60, min_frac=0.80):
    g = np.asarray(im.convert("L"))
    seg = g[:, x_from:x_to]
    frac = (seg < dark).mean(axis=0)
    xs, inside = [], False
    for i, f in enumerate(frac):
        if f >= min_frac and not inside:
            xs.append(x_from + i); inside = True
        elif f < min_frac:
            inside = False
    merged = []
    for x in xs:
        if merged and x - merged[-1] <= 4:
            continue
        merged.append(x)
    return merged


def fit(im, w, h, pad=0):
    box = im.getbbox() or (0, 0, *im.size)
    c = im.crop(box)
    r = min((w - 2 * pad) / c.width, (h - 2 * pad) / c.height)
    return c.resize((max(1, int(c.width * r)), max(1, int(c.height * r))), Image.LANCZOS)


def main():
    gear3 = Image.open(os.path.join(P, "gear", "01_machines-13-00893-g003.png"))
    mat16 = Image.open(os.path.join(P, "gear3", "09_materials-16-04095-g010.png"))
    coat = Image.open(os.path.join(P, "gear2", "09_coatings-09-00042-g002.png"))
    hust = Image.open(os.path.join(P, "F2.png"))

    W = 1760
    X0, CW, CH = 30, (1760 - 60) // 4, 300
    yG = 100
    yB = yG + CH + 66
    H = yB + CH + 60 + 46 + 100       # bearing cell taller + src lines
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)

    # ---------------- gear row ----------------
    gw, gh = gear3.size
    gtop = int(gh * 0.865)            # drop the printed "(a)/(b)" caption strip
    gear_cells = [
        ("Chipped  齿面剥落/崩齿", gear3.crop((int(gw * 0.505), 0, gw, gtop))),
        ("Miss  缺齿",           gear3.crop((0, 0, int(gw * 0.495), gtop))),
    ]
    mw, mh = mat16.size
    # panel (a) upper part: tooth-flank pitting
    gear_cells.append(("Surface  齿面磨损/点蚀",
                       mat16.crop((int(mw * 0.02), int(mh * 0.08), int(mw * 0.335), int(mh * 0.47)))))
    # panel (a) lower part: zoom on the crack line at the tooth root
    gear_cells.append(("Root  齿根裂纹",
                       mat16.crop((int(mw * 0.06), int(mh * 0.51), int(mw * 0.24), int(mh * 0.99)))))

    # ---------------- bearing row (HUST) ----------------
    vl = find_vlines(hust, 200, hust.width)
    vw, vh = hust.size
    bounds = vl[:5] if len(vl) >= 5 else [241 + i * (vw - 241) // 4 for i in range(5)]
    order = [(0, "Inner  内圈"), (1, "Outer  外圈"), (2, "Ball  滚动体"), (3, "Comb  复合")]
    b_cells = [(lab, hust.crop((bounds[i] + 3, 0, bounds[i + 1] - 3, vh))) for i, lab in order]

    for i, (lab, img) in enumerate(gear_cells):
        cx = X0 + i * CW
        dr.rectangle([cx, yG, cx + CW - 10, yG + CH], outline="#c8c8c8", width=2)
        t = fit(img, CW - 16, CH - 44, pad=6)
        sheet.paste(t, (cx + (CW - 10 - t.width) // 2, yG + 34 + (CH - 44 - t.height) // 2))
        dr.text((cx + 8, yG + 8), lab, fill="#1a1a1a", font=F(20, True))

    for i, (lab, img) in enumerate(b_cells):
        cx = X0 + i * CW
        dr.rectangle([cx, yB, cx + CW - 10, yB + CH + 34], outline="#c8c8c8", width=2)
        t = fit(img, CW - 16, CH + 20, pad=4)
        sheet.paste(t, (cx + (CW - 10 - t.width) // 2, yB + 34 + (CH + 20 - t.height) // 2))
        dr.text((cx + 8, yB + 8), lab, fill="#1a1a1a", font=F(20, True))

    dr.text((X0, 32), "齿轮箱与滚动轴承各类故障的实物形态对照",
            fill="#111111", font=F(30, True))
    dr.text((X0, 70), "上排：齿轮故障实物照片（缺齿 / 崩齿 / 齿面点蚀 / 齿根裂纹）    下排：滚动轴承故障实物照片（红圈标出损伤位置）",
            fill="#555555", font=F(17))

    yy = yB + CH + 34 + 18
    for line in [
        "图片来源（均为开放获取、CC BY 4.0）：缺齿/崩齿 — Machines 2025, 13(10), 893；齿面磨损、齿根裂纹 — Materials 2023, 16(12), 4095；",
        "轴承内圈/外圈/滚动体/复合 — HUST bearing dataset（Thuan & Hong, BMC Research Notes 2023, doi:10.17632/cbv7jyx4p9）。",
        "说明：本文所用东南大学（SEU）数据集官方仅公开振动信号，未公开故障件实物照片；上图为公开数据集中同类型故障的实物照片，仅作形态参考。",
    ]:
        dr.text((X0, yy), line, fill="#666666", font=F(15))
        yy += 24

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "fault_specimens_real.png")
    sheet.save(dst, quality=92)
    print("saved", dst, sheet.size, "vlines:", bounds)


if __name__ == "__main__":
    main()
