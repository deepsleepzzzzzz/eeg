"""Build a labelled contact sheet from a directory of images.

Usage: python contact_sheet.py <imgdir> <out.jpg> [cols] [cellw]
"""
import os, sys, glob
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def font(sz):
    for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main(d, out, cols=5, cw=420):
    files = sorted([f for f in glob.glob(os.path.join(d, "*"))
                    if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")])
    ch = 330
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
    dr = ImageDraw.Draw(sheet)
    f = font(16)
    for i, path in enumerate(files):
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((cw - 12, ch - 40))
        except Exception as e:
            print("skip", path, e)
            continue
        r, c = divmod(i, cols)
        x, y = c * cw, r * ch
        dr.rectangle([x, y, x + cw - 1, y + ch - 1], outline="#bbbbbb")
        sheet.paste(im, (x + (cw - im.size[0]) // 2, y + 30 + (ch - 40 - im.size[1]) // 2))
        dr.text((x + 6, y + 6), f"{i+1:02d}  {os.path.basename(path)[:40]}", fill="#c00000", font=f)
    sheet.save(out, quality=82)
    print("saved", out, sheet.size, "n =", len(files))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 5,
         int(sys.argv[4]) if len(sys.argv) > 4 else 420)
