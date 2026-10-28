"""Cross-check: pull crs:Upright* keys straight out of the exported TIFFs' XMP.

Lightroom writes develop settings into exported files when
AgMetadata_writeOutputReferredDevelopSettingsToXmp is on. This is an
independent channel from the Lua SDK dump, so agreement between the two
confirms we are reading Adobe's real solver output.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lrtest", "out")

KEYS = ["PerspectiveUpright", "PerspectiveVertical", "PerspectiveHorizontal",
        "PerspectiveRotate", "PerspectiveAspect", "PerspectiveScale",
        "PerspectiveX", "PerspectiveY", "CropConstrainToWarp",
        "UprightVersion", "UprightFocalMode", "UprightFocalLength35mm",
        "UprightCenterMode", "UprightCenterNormX", "UprightCenterNormY",
        "UprightTransformCount", "UprightDependentDigest", "UprightPreview",
        "UprightFourSegmentsCount"]
for i in range(6):
    KEYS.append(f"UprightTransform_{i}")
for i in range(4):
    KEYS.append(f"UprightFourSegments_{i}")


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*", "*.tif")))
    if not files:
        print(f"no exported files under {OUT}")
        return
    args = ["exiftool", "-j", "-G1"]
    for k in KEYS:
        args.append(f"-XMP-crs:{k}")
    args += files
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print("exiftool failed:", res.stderr[:500])
        return
    data = json.loads(res.stdout)
    for rec in data:
        src = rec.get("SourceFile", "")
        rel = os.path.relpath(src, OUT)
        hits = {k: v for k, v in rec.items()
                if k != "SourceFile" and k.split(":")[-1] in KEYS and v not in ("", None)}
        print(f"\n=== {rel} ===")
        if not hits:
            print("   (no crs:Upright / Perspective keys)")
            continue
        for k in sorted(hits):
            v = hits[k]
            if len(str(v)) > 90:
                v = str(v)[:90] + "..."
            print(f"   {k.split(':')[-1]:<28} = {v}")


if __name__ == "__main__":
    sys.exit(main())
