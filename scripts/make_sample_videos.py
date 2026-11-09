"""Generate SYNTHETIC test videos (moving boxes on a road) to smoke-test the pipeline.

IMPORTANT: YOLO is trained on real photographs, so it will usually NOT detect these
cartoon boxes. They only prove that upload -> frames -> processing -> DB -> dashboard
works. For real detection results use real traffic footage (see README section 6).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def make_video(path, n_vehicles, seconds=8, fps=25, size=(640, 360), seed=0):
    rnd = random.Random(seed)
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not vw.isOpened():
        vw = cv2.VideoWriter(str(path.with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    cars = [{"x": rnd.uniform(0, w), "lane": rnd.randint(0, 3), "v": rnd.uniform(40, 120),
             "w": rnd.randint(40, 70), "h": 24} for _ in range(n_vehicles)]
    for f in range(seconds * fps):
        img = np.full((h, w, 3), 60, np.uint8)
        for i in range(1, 4):
            cv2.line(img, (0, int(h * i / 4)), (w, int(h * i / 4)), (200, 200, 200), 1, cv2.LINE_AA)
        for c in cars:
            c["x"] = (c["x"] + c["v"] / fps) % (w + 80)
            y = int(h * (c["lane"] + 0.5) / 4 - c["h"] / 2)
            cv2.rectangle(img, (int(c["x"]) - c["w"], y), (int(c["x"]), y + c["h"]), (255, 255, 255), -1)
        vw.write(img)
    vw.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_videos")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for i, (name, n) in enumerate([("north_heavy", 22), ("east_light", 4), ("south_medium", 11), ("west_very_light", 2)]):
        make_video(out / f"synthetic_{name}.mp4", n, seed=i)
        print("created", out / f"synthetic_{name}.mp4")
