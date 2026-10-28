"""
Harvest Adobe's own Upright solver output from the live Lightroom Classic catalog.

Lightroom stores every Upright analysis it has ever run inside
Adobe_imageDevelopSettings.text, a Lua-source serialisation of the develop
settings.  The Upright results live in keys named UprightTransform_N, where N is
the *enum* value of crs:PerspectiveUpright:

    0 = Off     1 = Auto    2 = Full    3 = Level    4 = Vertical   5 = Guided

All five slots are precomputed and cached for every analysed photo, which means
the catalog is a large, free corpus of exactly the algorithm we are reverse
engineering -- Adobe's real inputs and outputs, produced by the shipped binary.

Outputs JSONL, one record per photo.
"""

import argparse
import json
import os
import re
import sqlite3
import sys

DEFAULT_DB = "/Volumes/SSD/Lightroom/Lightroom Catalog.lrcat"

# UprightTransform_N = "a,b,c,d,e,f,g,h,i"  (row-major 3x3, normalised [0,1] frame)
RE_TRANSFORM = re.compile(r'UprightTransform_(\d)\s*=\s*"([^"]*)"')
RE_NUM = re.compile(r'(\w+)\s*=\s*(-?[\d.]+|"[^"]*")')

SCALARS = [
    "PerspectiveUpright", "PerspectiveVertical", "PerspectiveHorizontal",
    "PerspectiveRotate", "PerspectiveScale", "PerspectiveX", "PerspectiveY",
    "PerspectiveAspect", "CropConstrainToWarp",
    "UprightVersion", "UprightFocalMode", "UprightFocalLength35mm",
    "UprightCenterMode", "UprightCenterNormX", "UprightCenterNormY",
    "UprightTransformCount", "UprightFourSegmentsCount", "UprightPreview",
    "UprightDependentDigest",
]

QUERY = """
SELECT d.image, d.text, f.baseName, f.extension, fo.pathFromRoot,
       r.absolutePath, i.fileWidth, i.fileHeight, i.orientation
FROM Adobe_imageDevelopSettings d
JOIN Adobe_images i ON i.id_local = d.image
JOIN AgLibraryFile f ON f.id_local = i.rootFile
JOIN AgLibraryFolder fo ON fo.id_local = f.folder
JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
WHERE d.text LIKE '%UprightTransform_1%'
  AND i.fileWidth > 0 AND i.fileHeight > 0
"""


def parse_text(text):
    """Pull the Upright keys out of the Lua-source settings blob."""
    out = {"transforms": {}}
    for n, vals in RE_TRANSFORM.findall(text):
        try:
            nums = [float(v) for v in vals.split(",")]
        except ValueError:
            continue
        if len(nums) == 9:
            out["transforms"][int(n)] = nums
    for key, raw in RE_NUM.findall(text):
        if key in SCALARS and key not in out:
            out[key] = raw.strip('"') if raw.startswith('"') else float(raw)
    # the nested Preset/Parameters block can shadow top-level values; LR's own
    # disambiguation is "last wins", so re-scan allowing overwrite for scalars
    for key, raw in RE_NUM.findall(text):
        if key in SCALARS:
            out[key] = raw.strip('"') if raw.startswith('"') else float(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "lr_transforms.jsonl"))
    ap.add_argument("--only-auto", action="store_true",
                    help="keep only photos whose Auto slot differs from identity")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cur = con.cursor()
    n = kept = 0
    with open(args.out, "w") as fh:
        for row in cur.execute(QUERY):
            n += 1
            (image, text, base, ext, path_from_root, abs_root,
             fw, fh_, orient) = row
            p = parse_text(text)
            tr = p["transforms"]
            if 1 not in tr:
                continue
            ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            if args.only_auto and max(abs(a - b) for a, b in zip(tr[1], ident)) < 1e-6:
                continue
            full = os.path.join(abs_root, path_from_root, f"{base}.{ext}")
            rec = {
                "image_id": image, "path": full, "width": fw, "height": fh_,
                "orientation": orient,
                "transform": {str(k): v for k, v in sorted(tr.items())},
                "meta": {k: p.get(k) for k in SCALARS if k in p},
            }
            fh.write(json.dumps(rec) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"scanned {n} analysed photos, kept {kept} -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
