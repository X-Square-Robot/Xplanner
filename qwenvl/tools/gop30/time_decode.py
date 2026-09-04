#!/usr/bin/env python
"""Time the dataset's own decode on original vs GOP-30, sequential vs keyframe-seek.

Proves: (a) GOP-30 + seek is fast, (b) you MUST lower av_sequential_span_max
(GOP-30 + forced-sequential is still slow).
"""
import sys, time, math, threading
sys.path.insert(0, "/x2robot_v2/cyril/x2robot_dataset_v2")
from x2robot_dataset_v2.readers.frame_decoder import X2RobotFrameDecoder

ORIG = "/mnt/cpfs/zbl-cpfs-new/open_data/Multi-Modal-dataset/VLM-3R-DATA/videos/arkitscenes/videos/40958767.mp4"
GOP  = "/mnt/data/x2robot_v2/cyril/gop30_videos" + ORIG
VIDEO_FPS, VIDEO_MAXLEN = 1, 32

def probe(path):
    import av
    c = av.open(path, "r")
    s = next(x for x in c.streams if x.type == "video")
    total = int(s.frames or 0)
    dur = float(s.duration * s.time_base) if s.duration else 0.0
    c.close()
    return total, dur

def sample_idxs(total, dur):
    n = min(total, VIDEO_MAXLEN, max(1, math.floor(dur * VIDEO_FPS)))
    import numpy as np
    return np.linspace(0, total - 1, n).astype(int).tolist()

def timed(path, span_max, idxs, timeout=90):
    dec = X2RobotFrameDecoder(backend="av")
    dec.av_sequential_span_max = span_max
    box = {}
    def work():
        t = time.time()
        frames = dec.decode_frames(path, idxs)
        box["t"] = time.time() - t
        box["n"] = len(frames)
    th = threading.Thread(target=work, daemon=True); th.start(); th.join(timeout)
    if th.is_alive():
        return f">{timeout}s (TIMEOUT/hang)", None
    return f"{box['t']:.1f}s", box["n"]

for tag, path in [("ORIGINAL", ORIG), ("GOP30", GOP)]:
    total, dur = probe(path)
    idxs = sample_idxs(total, dur)
    print(f"\n=== {tag} ===  total={total} dur={dur:.1f}s  sample {len(idxs)} frames idx[0,1,-1]={idxs[0],idxs[1],idxs[-1]}")
    t_seq,  n_seq  = timed(path, 10**9, idxs)   # forced-sequential (current behavior)
    print(f"  forced-sequential (span_max=1e9, CURRENT): {t_seq}  frames={n_seq}")
    t_seek, n_seek = timed(path, 128,   idxs)   # keyframe-seek
    print(f"  keyframe-seek     (span_max=128):          {t_seek}  frames={n_seek}")
