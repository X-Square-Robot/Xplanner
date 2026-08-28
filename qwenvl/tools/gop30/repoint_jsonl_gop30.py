#!/usr/bin/env python
"""Build NEW GOP-30 source dirs on shared CPFS (originals untouched).

For each video source, writes a fresh data.jsonl/.index/manifest.json under
NEW_BASE/sources/<name>/ with the `video` paths repointed to the GOP-30 copies.
The original source dirs are NOT modified. Then point the two video sources in
the data-config YAML at these new dirs (printed at the end).

A video whose GOP-30 copy is missing (conversion failed) keeps its ORIGINAL path
as a safe fallback (slow but won't break).

  python repoint_jsonl_gop30.py             # build new source dirs
  python repoint_jsonl_gop30.py --dry-run   # report counts only
  python repoint_jsonl_gop30.py --clean     # remove the new source dirs
"""
import os, json, argparse, shutil
import numpy as np

SRC_ROOT     = "/mnt/data/x2robot_v2/cyril/multimodal_datasets_v3"
SOURCES      = ["vlm3r", "vsi590k_video"]
NEW_BASE     = "/mnt/cpfs/zbl-cpfs-new/open_data/cyril/vqa_gop30"
SOURCES_OUT  = os.path.join(NEW_BASE, "sources")
VIDEO_OUT    = os.path.join(NEW_BASE, "videos")
STRIP_PREFIX = "/mnt/cpfs/zbl-cpfs-new/open_data/"

def gop30_video_path(orig):  # keep in sync with convert_gop30_batch.gop30_video_path
    rel = orig[len(STRIP_PREFIX):] if orig.startswith(STRIP_PREFIX) else orig.lstrip("/")
    return os.path.join(VIDEO_OUT, rel)

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--clean", action="store_true")
args = ap.parse_args()

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
