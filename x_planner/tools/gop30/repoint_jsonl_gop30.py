#!/usr/bin/env python
"""Build NEW GOP-30 source dirs on shared CPFS (originals untouched).

For each video source, writes a fresh data.jsonl/.index/manifest.json under
NEW_BASE/sources/<name>/ with the `video` paths repointed to the GOP-30 copies.
The original source dirs are NOT modified. Then point the two video sources in
the data-config YAML at these new dirs (printed at the end).

A video whose GOP-30 copy is missing (conversion failed) keeps its ORIGINAL path
as a safe fallback (slow but won't break).

  python repoint_jsonl_gop30.py --source-root /data/sources \
      --output-root /data/gop30
"""
import os, json, argparse, shutil
import numpy as np

SRC_ROOT = ""
SOURCES: list[str] = []
SOURCES_OUT = ""
VIDEO_OUT = ""
STRIP_PREFIX = ""

def gop30_video_path(orig):  # keep in sync with convert_gop30_batch.gop30_video_path
    rel = (
        orig[len(STRIP_PREFIX):]
        if STRIP_PREFIX and orig.startswith(STRIP_PREFIX)
        else orig.lstrip("/")
    )
    return os.path.join(VIDEO_OUT, rel)

ap = argparse.ArgumentParser()
ap.add_argument("--source-root", default=os.environ.get("XPLANNER_GOP30_SOURCE_ROOT"))
ap.add_argument("--output-root", default=os.environ.get("XPLANNER_GOP30_OUTPUT_ROOT"))
ap.add_argument("--strip-prefix", default=os.environ.get("XPLANNER_MEDIA_ROOT", ""))
ap.add_argument("--sources", default="vlm3r,vsi590k_video",
                help="comma-separated source directories below --source-root")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--clean", action="store_true")
args = ap.parse_args()
if not args.source_root or not args.output_root:
    ap.error("--source-root and --output-root are required")
SRC_ROOT = os.path.abspath(args.source_root)
SOURCES = [value.strip() for value in args.sources.split(",") if value.strip()]
output_root = os.path.abspath(args.output_root)
SOURCES_OUT = os.path.join(output_root, "sources")
VIDEO_OUT = os.path.join(output_root, "videos")
STRIP_PREFIX = os.path.abspath(args.strip_prefix).rstrip("/") + "/" if args.strip_prefix else ""

def remap(p):
    if not isinstance(p, str) or not p.strip():
        return p, False
    dst = gop30_video_path(p.strip())
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst, True
    return p, False  # fallback to original

def build_index(jsonl_path, index_path):
    offs, off = [], 0
    with open(jsonl_path, "rb") as f:
        for line in f:
            offs.append(off); off += len(line)
    np.array(offs, dtype="<u8").tofile(index_path)
    return len(offs)

for name in SOURCES:
    sd = os.path.join(SRC_ROOT, name)
    od = os.path.join(SOURCES_OUT, name)
    if args.clean:
        shutil.rmtree(od, ignore_errors=True); print(f"[{name}] removed {od}"); continue
    os.makedirs(od, exist_ok=True)
    n=remap_n=fb=0
    out = os.path.join(od, "data.jsonl")
    with open(os.path.join(sd, "data.jsonl")) as fin, open(out, "w") as fout:
        for line in fin:
            r = json.loads(line); n += 1
            v = r.get("video")
            if v is not None:
                if isinstance(v, list):
                    nv=[]
                    for p in v:
                        q,ok=remap(p); nv.append(q); remap_n+=ok; fb+=(not ok)
                    r["video"]=nv
                else:
                    q,ok=remap(v); r["video"]=q; remap_n+=ok; fb+=(not ok)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.dry_run:
        os.unlink(out)
        print(f"[{name}] DRY lines={n} remapped={remap_n} fallback(orig)={fb}")
        continue
    cnt = build_index(out, os.path.join(od, "data.index"))
    assert cnt == n, f"index {cnt} != lines {n}"
    mf = os.path.join(sd, "manifest.json")
    if os.path.exists(mf): shutil.copy2(mf, os.path.join(od, "manifest.json"))
    print(f"[{name}] -> {od}  lines={n} remapped={remap_n} fallback={fb} index={cnt}")

if not args.clean and not args.dry_run:
    print("\nNow point the two video sources in the data-config YAML at:")
    for name in SOURCES:
        print(f"  {name}: path: {os.path.join(SOURCES_OUT, name)}")
print("done." + ("  (dry-run)" if args.dry_run else ""))
