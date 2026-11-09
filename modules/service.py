"""Analytics service: combines DB rows + live job state into the JSON the UI renders."""
import json
import threading

import config
from modules import database as db
from modules import density
from modules.predictor import CongestionPredictor
from modules.signal_optimizer import optimize


class Analytics:
    def __init__(self, jobs):
        self.jobs = jobs
        self.predictor = CongestionPredictor()
        self._cache, self._lock = {}, threading.Lock()

    # ------------------------------------------------------------ helpers
    def samples_for(self, road):
        job = self.jobs.get(road["id"])
        if job is not None and not job.finished:
            return job.snapshot_samples()
        return db.get_samples(road["id"])

    def predict(self, road, samples, final):
        key = (road["id"], len(samples), final)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        hist = db.history_samples(road["road_name"], road["session_id"]) if final else None
        result = self.predictor.predict(samples, history=hist, use_ml=final)
        with self._lock:
            if len(self._cache) > 500:
                self._cache.clear()
            self._cache[key] = result
        return result

    def on_session_done(self, session_id):
        """Called when every road in a session finished: store the final signal plan."""
        done = [r for r in db.get_roads(session_id) if r["status"] == "completed"]
        if len(done) >= 2:
            db.save_signal_plan(session_id, optimize(self._ordered({r["road_name"]: r["avg_pcu"] for r in done})))

    @staticmethod
    def _ordered(scores):
        order = {r: i for i, r in enumerate(config.ROADS)}
        return dict(sorted(scores.items(), key=lambda kv: order.get(kv[0], 99)))

    # ------------------------------------------------------------ overview
    def overview(self, session_id):
        sess = db.get_session(session_id)
        if not sess:
            return None
        roads_out, class_totals, any_active = [], {k: 0.0 for k in config.CLASS_NAMES}, False
        for r in db.get_roads(session_id):
            job = self.jobs.get(r["id"])
            live = job is not None and not job.finished
            any_active = any_active or live or r["status"] in ("queued", "processing")
            samples = self.samples_for(r)
            if live:
                status, progress, error = job.status, job.progress, None
                now = job.current or density.describe({})
                mean_pcu = sum(s["pcu"] for s in samples) / len(samples) if samples else 0.0
                peak = max((s["pcu"] for s in samples), default=0.0)
                avg = {"total": round(sum(s["total"] for s in samples) / len(samples), 1) if samples else 0,
                       "pcu": round(mean_pcu, 2), "level": density.classify(mean_pcu)}
                uniq = {k: len(v) for k, v in job.unique.items()}
                tracking, frames = None, len(samples)
            else:
                status, progress, error = r["status"], r["progress"], r["error"]
                counts = {k: r[f"avg_{k}"] for k in config.CLASS_NAMES}
                now = density.describe({k: round(v) for k, v in counts.items()})
                now.update(pcu=r["avg_pcu"], level=r["avg_level"], percent=density.percent(r["avg_pcu"]))
                avg = {"total": r["avg_total"], "pcu": r["avg_pcu"], "level": r["avg_level"]}
                peak, tracking, frames = r["max_pcu"], bool(r["tracking"]), r["frames_processed"]
                uniq = json.loads(r["unique_json"] or "{}")
            for k in config.CLASS_NAMES:
                class_totals[k] += now["counts"][k]
            roads_out.append({
                "id": r["id"], "name": r["road_name"], "status": status, "progress": progress, "error": error,
                "live": live, "now": now, "avg": avg, "peak": {"pcu": round(peak, 2), "level": density.classify(peak)},
                "unique": {"total": sum(uniq.values()), "by_class": uniq, "tracking": tracking},
                "prediction": self.predict(r, samples, final=not live) if samples else None,
                "frames": frames, "snapshot": r["snapshot"], "duration": r["duration"], "filename": r["filename"],
            })
        ok = [x for x in roads_out if x["status"] in ("processing", "completed") and x["frames"]]
        plan, provisional = db.get_signal_plan(session_id), False
        if plan is None and len(ok) >= 2:
            plan, provisional = optimize(self._ordered({x["name"]: x["avg"]["pcu"] for x in ok})), True
        mean_pcu = sum(x["now"]["pcu"] for x in ok) / len(ok) if ok else 0
        return {
            "session": sess, "active": any_active, "roads": roads_out,
            "totals": {"vehicles_in_view": sum(x["now"]["total"] for x in ok),
                       "unique_vehicles": sum(x["unique"]["total"] for x in roads_out),
                       "class_counts": {k: round(v) for k, v in class_totals.items()},
                       "level": density.classify(mean_pcu), "mean_pcu": round(mean_pcu, 2),
                       "percent": density.percent(mean_pcu)},
            "plan": plan, "plan_provisional": provisional,
        }

    def report_data(self, session_id):
        """Everything the chart/PDF/CSV builders need (completed or running)."""
        ov = self.overview(session_id)
        if not ov:
            return None
        ov["samples"] = {r["name"]: self.samples_for(db.get_road(r["id"])) for r in ov["roads"]}
        return ov
