"""Split ONE long traffic video into N consecutive clips (default 4) - handy for demoing
a four-road junction when you only have a single recording.
Usage: python scripts/split_video.py my_traffic.mp4 --parts 4 --out sample_videos
"""
import argparse
from pathlib import Path

import cv2


def split(src, parts, out):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    per = total // parts
    out.mkdir(parents=True, exist_ok=True)
    for p in range(parts):
        path = out / f"{Path(src).stem}_part{p + 1}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(per):
            ok, frame = cap.read()
            if not ok:
                break
            vw.write(frame)
        vw.release()
        print("created", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--parts", type=int, default=4); ap.add_argument("--out", default="sample_videos")
    a = ap.parse_args()
    split(a.video, a.parts, Path(a.out))
