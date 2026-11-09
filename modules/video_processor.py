"""Video / webcam processing: frame extraction -> preprocessing -> YOLO -> counting -> density."""
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

import config
from modules import database as db
from modules import density
from modules.detector import annotate, preprocess


class VideoError(RuntimeError):
    pass


def _encode(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _sample(t, counts):
    d = density.describe(counts)
    return {"t": round(t, 2), "total": d["total"], "pcu": d["pcu"], "level": d["level"], **d["counts"]}


class RoadJob:
    """Processes one uploaded video in a background thread."""

    def __init__(self, road_id, session_id, name, path, detector_factory, on_done):
        self.road_id, self.session_id, self.name, self.path = road_id, session_id, name, path
        self.factory, self.on_done = detector_factory, on_done
        self.status, self.error, self.progress = "queued", None, 0
        self.latest_jpeg, self.peak_jpeg, self.peak_pcu = None, None, -1.0
        self.samples, self.current = [], None
        self.unique = {k: set() for k in config.CLASS_NAMES}
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"road-{road_id}")

    @property
    def finished(self):
        return self.status in ("completed", "error", "cancelled")

    def snapshot_samples(self):
        with self._lock:
            return list(self.samples)

    # ---------------------------------------------------------------- worker
    def run(self):
        det = None
        try:
            self.status = "processing"
            db.update_road(self.road_id, status="processing")
            det = self.factory()
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise VideoError("The video file could not be opened (unsupported codec or corrupt file).")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps == fps and fps > 0 else 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(fps / config.PROCESS_FPS)))
            db.update_road(self.road_id, total_frames=total, duration=round(total / fps, 1) if total else 0)
            idx = 0
            while not self.cancel.is_set():
                if not cap.grab():
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        self._process(frame, idx / fps, det)
                idx += 1
                if total:
                    self.progress = min(99, int(idx * 100 / total))
            cap.release()
            if self.cancel.is_set():
                self.status = "cancelled"
                db.update_road(self.road_id, status="cancelled")
                return
            if not self.samples:
                raise VideoError("No readable frames were found in this video.")
            self._finish(det)
        except Exception as exc:                         # report any failure to the UI
            self.error, self.status = str(exc), "error"
            if self.samples:
                db.add_samples(self.road_id, self.samples)
            db.update_road(self.road_id, status="error", error=self.error[:500])
        finally:
            self.on_done(self.session_id)

    def _process(self, frame, t, det):
        frame = preprocess(frame)
        dets = det.detect(frame)
        counts = {k: 0 for k in config.CLASS_NAMES}
        for d in dets:
            counts[d["label"]] += 1
            if d.get("id") is not None:
                self.unique[d["label"]].add(d["id"])
        s = _sample(t, counts)
        img = annotate(frame, dets, f"{self.name} Road", s["total"], s["level"], s["pcu"])
        jpg = _encode(img)
        with self._lock:
            self.samples.append(s)
            self.current = density.describe(counts)
            self.latest_jpeg = jpg
            if s["pcu"] > self.peak_pcu:
                self.peak_pcu, self.peak_jpeg = s["pcu"], jpg

    def _finish(self, det):
        s = self.samples
        db.add_samples(self.road_id, s)
        totals = np.array([x["total"] for x in s])
        pcus = np.array([x["pcu"] for x in s])
        snap = None
        if self.peak_jpeg:
            config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            snap = f"road_{self.road_id}.jpg"
            (config.SNAPSHOT_DIR / snap).write_bytes(self.peak_jpeg)
        uniq = {k: len(v) for k, v in self.unique.items()}
        tracking = bool(getattr(det, "tracking", False))
        db.update_road(
            self.road_id, status="completed", progress=100, frames_processed=len(s),
            avg_total=round(float(totals.mean()), 2), max_total=int(totals.max()),
            avg_pcu=round(float(pcus.mean()), 2), max_pcu=round(float(pcus.max()), 2),
            avg_level=density.classify(float(pcus.mean())), peak_level=density.classify(float(pcus.max())),
            **{f"avg_{k}": round(float(np.mean([x[k] for x in s])), 2) for k in config.CLASS_NAMES},
            unique_total=sum(uniq.values()) if tracking else 0, unique_json=json.dumps(uniq),
            tracking=int(tracking), snapshot=snap)
        self.progress, self.status = 100, "completed"


class JobManager:
    def __init__(self, detector_factory, on_session_done=None):
        self.factory, self.jobs, self.on_session_done = detector_factory, {}, on_session_done
        self._lock = threading.Lock()

    def start(self, road_id, session_id, name, path):
        job = RoadJob(road_id, session_id, name, path, self.factory, self._road_done)
        with self._lock:
            self.jobs[road_id] = job
        job.thread.start()
        return job

    def get(self, road_id):
        return self.jobs.get(road_id)

    def cancel_session(self, session_id):
        for j in list(self.jobs.values()):
            if j.session_id == session_id:
                j.cancel.set()

    def _road_done(self, session_id):
        active = [j for j in self.jobs.values() if j.session_id == session_id and not j.finished]
        if not active and self.on_session_done:
            try:
                self.on_session_done(session_id)
            except Exception:
                pass

    def wait_all(self, timeout=60):
        for j in list(self.jobs.values()):
            j.thread.join(timeout)


class LiveCamera:
    """Webcam analysis running in a background thread (server-side capture)."""

    def __init__(self, detector_factory):
        self.factory = detector_factory
        self.running, self.error, self.thread = False, None, None
        self.latest_jpeg, self.current = None, None
        self.samples, self.unique = [], {k: set() for k in config.CLASS_NAMES}
        self.started = None
        self._lock = threading.Lock()

    def start(self, camera_index=0):
        if self.running:
            return
        self.error, self.samples, self.current, self.latest_jpeg = None, [], None, None
        self.unique = {k: set() for k in config.CLASS_NAMES}
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            raise VideoError(f"Webcam {camera_index} could not be opened. Check that it exists and is not used by another app.")
        try:
            det = self.factory()
        except Exception:
            cap.release()
            raise
        self.running, self.started = True, time.time()
        self.thread = threading.Thread(target=self._loop, args=(cap, det), daemon=True)
        self.thread.start()

    def _loop(self, cap, det):
        last_sample = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.error = "Webcam stopped delivering frames."
                    break
                frame = preprocess(frame)
                dets = det.detect(frame)
                counts = {k: 0 for k in config.CLASS_NAMES}
                for d in dets:
                    counts[d["label"]] += 1
                    if d.get("id") is not None:
                        self.unique[d["label"]].add(d["id"])
                t = time.time() - self.started
                s = _sample(t, counts)
                jpg = _encode(annotate(frame, dets, "Live Camera", s["total"], s["level"], s["pcu"]))
                with self._lock:
                    self.latest_jpeg, self.current = jpg, density.describe(counts)
                    if t - last_sample >= 0.5:
                        self.samples.append(s)
                        last_sample = t
        except Exception as exc:
            self.error = str(exc)
        finally:
            cap.release()
            self.running = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(5)

    def snapshot_samples(self):
        with self._lock:
            return list(self.samples)

    def save_to_history(self):
        """Persist the live run as a session; returns session id or None."""
        s = self.snapshot_samples()
        if len(s) < 3:
            return None
        sid = db.create_session(f"Live webcam {db.now()}", source="webcam")
        rid = db.add_road(sid, "Webcam", "webcam", status="completed")
        db.add_samples(rid, s)
        tot, pcu = np.array([x["total"] for x in s]), np.array([x["pcu"] for x in s])
        uniq = {k: len(v) for k, v in self.unique.items()}
        db.update_road(rid, progress=100, frames_processed=len(s), avg_total=float(tot.mean()), max_total=int(tot.max()),
                       avg_pcu=float(pcu.mean()), max_pcu=float(pcu.max()), avg_level=density.classify(float(pcu.mean())),
                       peak_level=density.classify(float(pcu.max())), unique_total=sum(uniq.values()),
                       unique_json=json.dumps(uniq), duration=round(s[-1]["t"], 1),
                       **{f"avg_{k}": float(np.mean([x[k] for x in s])) for k in config.CLASS_NAMES})
        return sid
