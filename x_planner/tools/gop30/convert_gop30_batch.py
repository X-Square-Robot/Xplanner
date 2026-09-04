#!/usr/bin/env python
"""Batch GOP-30 transcode for the VQA video sources (non-destructive, shared CPFS).

Output goes to shared CPFS so all training nodes can read it; ORIGINALS ARE NEVER
TOUCHED. Each unique video -> a GOP-30 (keyframe every 30 frames) H.264 copy under
VIDEO_OUT, mirroring the path below the open_data prefix.

Speed/space: training samples <=32 frames at fps=1, so the native 60fps is ~15-30x
wasted. --fps re-encodes at a low fps (timestamp-faithful: same duration => same
sampled timestamps; only the frame is rounded to a coarser grid). Cuts encode CPU,
output size, AND training-time decode all at once.

  python convert_gop30_batch.py --source-root /data/sources \
      --output-root /data/gop30 [--jobs 16] [--threads 2] [--fps 4]
                                [--downscale 640] [--preset veryfast] [--crf 23]
                                [--timeout 600] [--limit N]
Resumable (skips existing verified outputs); per-video timeout + skip-on-error.
"""
import os, json, argparse, subprocess, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
SRC_ROOT = ""
SOURCES: list[str] = []
VIDEO_OUT = ""
STRIP_PREFIX = ""
FAILLOG = ""

def gop30_video_path(orig: str) -> str:
    """orig -> gop30 output path (single source of truth; repoint imports this)."""
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
ap.add_argument("--failure-log", help="defaults to <output-root>/gop30_failures.log")
ap.add_argument("--ffmpeg", default=FFMPEG)
ap.add_argument("--ffprobe", default=FFPROBE)
ap.add_argument("--jobs", type=int, default=24)
ap.add_argument("--threads", type=int, default=2, help="ffmpeg -threads per job (0=auto)")
# dataset_v2-aligned defaults: native fps, native resolution, CPU decode.
ap.add_argument("--fps", type=float, default=0.0, help="output fps (0=keep source, dataset_v2-aligned)")
ap.add_argument("--downscale", type=int, default=0, help="cap long side (0=keep resolution, dataset_v2-aligned)")
ap.add_argument("--preset", default="veryfast", help="libx264 preset (dataset_v2 uses 'medium'; veryfast is faster, same frames/GOP)")
ap.add_argument("--crf", type=int, default=23)
ap.add_argument("--timeout", type=int, default=600)
ap.add_argument("--gpus", default="", help="comma GPU ids for NVDEC decode, e.g. 0,3 (empty=CPU decode)")
ap.add_argument("--shard", default="0/1", help="i/N -> this node does slice i of N (multi-node)")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
if not args.source_root or not args.output_root:
    ap.error("--source-root and --output-root are required")
SRC_ROOT = os.path.abspath(args.source_root)
SOURCES = [value.strip() for value in args.sources.split(",") if value.strip()]
VIDEO_OUT = os.path.join(os.path.abspath(args.output_root), "videos")
STRIP_PREFIX = os.path.abspath(args.strip_prefix).rstrip("/") + "/" if args.strip_prefix else ""
FAILLOG = args.failure_log or os.path.join(os.path.abspath(args.output_root), "gop30_failures.log")
FFMPEG = args.ffmpeg
FFPROBE = args.ffprobe
GPUS = [g for g in args.gpus.split(",") if g.strip() != ""]

def collect_videos():
    vids = set()
    for name in SOURCES:
        with open(os.path.join(SRC_ROOT, name, "data.jsonl")) as f:
            for line in f:
                v = json.loads(line).get("video")
                if v is None: continue
                for p in (v if isinstance(v, list) else [v]):
                    if isinstance(p, str) and p.strip():
                        vids.add(p.strip())
    return sorted(vids)

def verify_gop(path):
    try:
        out = subprocess.run(
            [FFPROBE,"-v","error","-select_streams","v:0","-skip_frame","nokey",
             "-show_entries","frame=pts_time","-of","csv=p=0", path],
            capture_output=True, text=True, timeout=60).stdout.split()
        return len(out) >= 2
    except Exception:
        return False

def build_vf():
    chain = []
    if args.fps and args.fps > 0:
        chain.append(f"fps={args.fps}")
    if args.downscale and args.downscale > 0:
        # cap long side to --downscale, keep aspect, even dims
        chain.append(f"scale='if(gt(iw,ih),min({args.downscale},iw),-2)':"
                     f"'if(gt(iw,ih),-2,min({args.downscale},ih))'")
    return ["-vf", ",".join(chain)] if chain else []

def convert_one(task):
    src_abs, gpu = task
    dst = gop30_video_path(src_abs)
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and verify_gop(dst):
        return ("skip", src_abs, 0.0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.mp4"
    th = ["-threads", str(args.threads)] if args.threads > 0 else []
    hw = ["-hwaccel","cuda","-hwaccel_device",str(gpu)] if gpu is not None else []
    # ffmpeg flags aligned with dataset_v2 tools/distributed_gop/worker.py:
    #   libx264, -g 30 -keyint_min 30 -sc_threshold 0 -pix_fmt yuv420p
    cmd = [FFMPEG,"-y","-v","error", *hw, "-i",src_abs,
           "-c:v","libx264","-preset",args.preset,"-crf",str(args.crf), *build_vf(), *th,
           "-g","30","-keyint_min","30","-sc_threshold","0",
           "-vsync","cfr","-pix_fmt","yuv420p","-an", tmp]
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp): os.unlink(tmp)
        return ("timeout", src_abs, time.time()-t)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp): os.unlink(tmp)
        return ("error:"+r.stderr.strip()[-160:], src_abs, time.time()-t)
    os.replace(tmp, dst)
    return ("ok", src_abs, time.time()-t)

def main():
    os.makedirs(VIDEO_OUT, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(FAILLOG)), exist_ok=True)
    vids = collect_videos()                       # sorted -> deterministic across nodes
    si, sn = (int(x) for x in args.shard.split("/"))
    vids = vids[si::sn]                            # this node's disjoint slice
    if args.limit: vids = vids[:args.limit]
    tasks = [(v, GPUS[i % len(GPUS)] if GPUS else None) for i, v in enumerate(vids)]
    print(f"[{time.strftime('%H:%M:%S')}] shard {si}/{sn}: {len(vids)} videos -> {VIDEO_OUT}\n"
          f"  jobs={args.jobs} threads={args.threads} fps={args.fps} downscale={args.downscale} "
          f"preset={args.preset} crf={args.crf} gpus={GPUS or 'CPU'}", flush=True)
    counts={"ok":0,"skip":0,"timeout":0,"error":0}; done=0; t0=time.time(); enc_t=0.0
    with open(FAILLOG,"a") as fl, ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs={ex.submit(convert_one,t):t for t in tasks}
        for fut in as_completed(futs):
            status,v,dt = fut.result(); done+=1
            key = "error" if status.startswith("error") else status
            counts[key]=counts.get(key,0)+1
            if key=="ok": enc_t+=dt
            if key in ("timeout","error"):
                fl.write(f"{key}\t{dt:.0f}s\t{v}\t{status}\n"); fl.flush()
            if done%100==0 or done==len(vids):
                el=time.time()-t0; rate=done/el if el else 0
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(vids)} ok={counts['ok']} "
                      f"skip={counts['skip']} timeout={counts['timeout']} err={counts['error']} "
                      f"rate={rate:.2f}/s avg_enc={enc_t/max(counts['ok'],1):.1f}s "
                      f"eta={(len(vids)-done)/rate/60 if rate else 0:.0f}min", flush=True)
    print(f"DONE {counts}  failures -> {FAILLOG}", flush=True)

if __name__ == "__main__":
    main()
