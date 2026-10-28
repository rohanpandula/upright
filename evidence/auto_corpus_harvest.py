"""Read-only harvest of Adobe's own Auto/Full transforms for the corpus curve.

Writes /tmp/auto_corpus_slots.jsonl. No catalog writes.
"""
import json, os, re, sqlite3, sys

DB = "/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"
OUT = "/tmp/auto_corpus_slots.jsonl"

RE_TRANSFORM = re.compile(r'UprightTransform_(\d)\s*=\s*"([^"]*)"')
RE_NUM = re.compile(r'(\w+)\s*=\s*(-?[\d.]+|"[^"]*")')
SCALARS = ["UprightFocalLength35mm", "UprightCenterNormX", "UprightCenterNormY",
           "UprightFocalMode", "UprightCenterMode", "UprightVersion",
           "PerspectiveUpright", "PerspectiveScale", "PerspectiveRotate"]

QUERY = """
SELECT d.image, d.text, i.fileWidth, i.fileHeight, i.orientation
FROM Adobe_imageDevelopSettings d
JOIN Adobe_images i ON i.id_local = d.image
WHERE d.text LIKE '%UprightTransform_1%'
  AND d.text LIKE '%UprightTransform_2%'
  AND i.fileWidth > 0 AND i.fileHeight > 0
"""


def parse(text):
    tr = {}
    for n, vals in RE_TRANSFORM.findall(text):
        try:
            nums = [float(v) for v in vals.split(",")]
        except ValueError:
            continue
        if len(nums) == 9:
            tr[int(n)] = nums
    meta = {}
    for key, raw in RE_NUM.findall(text):
        if key in SCALARS:
            meta[key] = raw.strip('"') if raw.startswith('"') else float(raw)
    return tr, meta


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = kept = 0
    with open(OUT, "w") as fh:
        for image, text, w, h, orient in con.execute(QUERY):
            n += 1
            tr, meta = parse(text)
            if 1 not in tr or 2 not in tr:
                continue
            ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            if max(abs(a - b) for a, b in zip(tr[1], ident)) < 1e-6:
                continue
            if max(abs(a - b) for a, b in zip(tr[2], ident)) < 1e-6:
                continue
            if not meta.get("UprightFocalLength35mm"):
                continue
            fh.write(json.dumps({"image_id": image, "width": w, "height": h,
                                 "orientation": orient,
                                 "transform": {str(k): v for k, v in tr.items()},
                                 "meta": meta}) + "\n")
            kept += 1
    print(f"scanned {n} rows, kept {kept} -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
