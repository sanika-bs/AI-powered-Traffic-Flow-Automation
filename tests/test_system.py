"""Automated tests. Run:  python -m unittest discover -s tests -v
YOLO itself is replaced by a lightweight contour-based stand-in so the tests run anywhere
(no GPU / internet / model download). The real YOLO path is covered by manual test cases TC-20+."""
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules import database as db
from modules import density, reports
from modules.predictor import CongestionPredictor, bucketize
from modules.signal_optimizer import optimize
from scripts.make_sample_videos import make_video


class FakeDetector:
    """Finds white rectangles (the synthetic 'vehicles') and reports them as cars."""
    tracking = False

    def detect(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 235, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 300:
                out.append({"label": "car", "conf": 0.9, "box": (x, y, x + w, y + h), "id": None})
        return out


class DensityTests(unittest.TestCase):
    def test_classification_boundaries(self):                       # TC-01
        self.assertEqual(density.classify(0), "Low")
        self.assertEqual(density.classify(config.LOW_MAX_PCU), "Low")
        self.assertEqual(density.classify(config.LOW_MAX_PCU + .1), "Medium")
        self.assertEqual(density.classify(config.MEDIUM_MAX_PCU + .1), "High")

    def test_pcu_weights(self):                                      # TC-02
        self.assertEqual(density.pcu_of({"car": 2, "bus": 1, "motorcycle": 2}), 2 + 2.5 + 1.0)


class SignalTests(unittest.TestCase):
    def check(self, scores, **kw):
        plan = optimize(scores, **kw)
        greens = [r["green"] for r in plan["roads"]]
        self.assertEqual(sum(greens) + len(greens) * plan["yellow"], plan["cycle"])
        self.assertTrue(all(plan["min_green"] <= g <= plan["max_green"] for g in greens))
        return plan

    def test_busier_road_gets_more_green(self):                      # TC-03
        p = self.check({"N": 30, "E": 5, "S": 15, "W": 1})
        g = {r["road"]: r["green"] for r in p["roads"]}
        self.assertGreater(g["N"], g["S"]); self.assertGreater(g["S"], g["E"]); self.assertGreaterEqual(g["E"], g["W"])

    def test_no_starvation_with_empty_road(self):                    # TC-04
        p = self.check({"N": 100, "E": 0, "S": 0, "W": 0})
        self.assertTrue(all(r["green"] >= config.MIN_GREEN for r in p["roads"]))

    def test_equal_demand_equal_green(self):                         # TC-05
        g = [r["green"] for r in self.check({"a": 7, "b": 7, "c": 7, "d": 7})["roads"]]
        self.assertLessEqual(max(g) - min(g), 1)

    def test_all_zero_and_two_roads(self):                           # TC-06
        self.check({"a": 0, "b": 0}); self.check({"a": 50, "b": 0})

    def test_many_random_cases_invariants(self):                     # TC-07
        rng = np.random.default_rng(1)
        for _ in range(300):
            n = int(rng.integers(2, 7)); self.check({f"r{i}": float(rng.exponential(12)) for i in range(n)})

    def test_wait_reduction_non_negative_for_skewed_traffic(self):   # TC-08
        self.assertGreater(optimize({"N": 40, "E": 2, "S": 3, "W": 2})["reduction_pct"], 0)


class PredictorTests(unittest.TestCase):
    def series(self, f, n=60):
        return [{"t": i * .5, "pcu": f(i)} for i in range(n)]

    def test_rising_trend_detected(self):                            # TC-09
        p = CongestionPredictor().predict(self.series(lambda i: 2 + i * .3, 40), use_ml=False)
        self.assertEqual(p["trend"], "Rising"); self.assertGreater(p["predicted_pcu"], p["current_pcu"])

    def test_ml_used_with_history(self):                             # TC-10
        hist = [self.series(lambda i: 10 + 6 * np.sin(i / 6), 200)]
        p = CongestionPredictor().predict(self.series(lambda i: 10 + 6 * np.sin(i / 6), 120), history=hist, use_ml=True)
        self.assertIn("Gradient Boosting", p["method"]); self.assertGreaterEqual(p["predicted_pcu"], 0)

    def test_tiny_and_empty_inputs(self):                            # TC-11
        self.assertEqual(CongestionPredictor().predict([])["method"], "no data")
        self.assertEqual(CongestionPredictor().predict([{"t": 0, "pcu": 5}])["method"], "persistence")
        self.assertEqual(bucketize([]), [])


class DatabaseTests(unittest.TestCase):
    def test_roundtrip(self):                                        # TC-12
        with tempfile.TemporaryDirectory() as d:
            db.init_db(Path(d) / "t.db")
            sid = db.create_session("x"); rid = db.add_road(sid, "North", "a.mp4")
            db.add_samples(rid, [{"t": 0, "total": 3, "pcu": 3.0, "level": "Low", "car": 3, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}])
            db.update_road(rid, status="completed", avg_pcu=3.0)
            self.assertEqual(len(db.get_samples(rid)), 1); self.assertEqual(db.get_road(rid)["status"], "completed")
            with self.assertRaises(ValueError):
                db.update_road(rid, evil="DROP TABLE roads")
            db.save_signal_plan(sid, {"cycle": 120}); self.assertEqual(db.get_signal_plan(sid)["cycle"], 120)
            db.delete_session(sid); self.assertIsNone(db.get_road(rid))   # cascade


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        config.SNAPSHOT_DIR = t / "snap"
        cls.app = create_app({"TESTING": True, "DATABASE": str(t / "t.db"), "UPLOAD_DIR": str(t / "up")}, detector_factory=FakeDetector)
        cls.c = cls.app.test_client()
        cls.vids = {}
        for i, (road, n) in enumerate([("north", 18), ("east", 3), ("south", 9), ("west", 1)]):
            p = t / f"{road}.mp4"; make_video(p, n, seconds=6, fps=20, seed=i); cls.vids[road] = p

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pages_render_empty(self):                               # TC-13
        for url in ["/", "/upload", "/live", "/dashboard", "/density", "/prediction", "/signals", "/reports"]:
            self.assertEqual(self.c.get(url).status_code, 200, url)
        self.assertEqual(self.c.get("/api/session/999/overview").status_code, 404)

    def test_upload_validation(self):                                # TC-14
        r = self.c.post("/upload", data={}, follow_redirects=True)
        self.assertIn(b"at least one", r.data)
        r = self.c.post("/upload", data={"video_north": (io.BytesIO(b"hello"), "evil.exe")}, follow_redirects=True)
        self.assertIn(b"not a supported", r.data)
        r = self.c.post("/upload", data={"video_north": (io.BytesIO(b"not really a video"), "fake.mp4")}, follow_redirects=True)
        self.assertIn(b"could not be read", r.data)

    def test_simulate_api_validation(self):                          # TC-15
        post = lambda j: self.c.post("/api/simulate", json=j)
        self.assertEqual(post({}).status_code, 400)
        self.assertEqual(post({"roads": {"a": {"car": -1}, "b": {"car": 1}}}).status_code, 400)
        self.assertEqual(post({"roads": {"a": {"car": "x"}, "b": {"car": 1}}}).status_code, 400)
        r = post({"roads": {"North": {"car": 30}, "East": {"car": 2}}})
        self.assertEqual(r.status_code, 200); g = {x["road"]: x["green"] for x in r.json["roads"]}
        self.assertGreater(g["North"], g["East"])

    def test_full_pipeline(self):                                    # TC-16..19
        data = {f"video_{k}": (open(p, "rb"), p.name) for k, p in self.vids.items()}
        data["session_name"] = "pytest run"
        r = self.c.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)
        sid = int(r.headers["Location"].split("session=")[1])
        self.app.extensions["jobs"].wait_all(120)
        time.sleep(.5)
        ov = self.c.get(f"/api/session/{sid}/overview").json
        self.assertEqual([x["status"] for x in ov["roads"]], ["completed"] * 4, [x["error"] for x in ov["roads"]])
        avg = {x["name"]: x["avg"]["pcu"] for x in ov["roads"]}
        self.assertGreater(avg["North"], avg["South"]); self.assertGreater(avg["South"], avg["East"])
        plan = ov["plan"]; self.assertIsNotNone(plan); self.assertFalse(ov["plan_provisional"])
        g = {x["road"]: x["green"] for x in plan["roads"]}
        self.assertGreater(g["North"], g["West"]); self.assertGreaterEqual(min(g.values()), config.MIN_GREEN)
        self.assertTrue(all(x["prediction"]["method"] for x in ov["roads"]))
        for kind in reports.CHARTS:
            resp = self.c.get(f"/chart/{sid}/{kind}.png"); self.assertEqual(resp.status_code, 200); self.assertTrue(resp.data.startswith(b"\x89PNG"))
        pdf = self.c.get(f"/report/{sid}.pdf"); self.assertTrue(pdf.data.startswith(b"%PDF"))
        csv = self.c.get(f"/report/{sid}.csv").data.decode(); self.assertIn("North", csv); self.assertIn("signal plan", csv)
        for page in ["dashboard", "density", "prediction", "signals", "reports"]:
            self.assertEqual(self.c.get(f"/{page}?session={sid}").status_code, 200)
        self.assertEqual(self.c.get(f"/snapshot/{ov['roads'][0]['id']}").status_code, 200)
        self.assertEqual(self.c.post(f"/api/session/{sid}/delete").status_code, 200)       # TC-19
        self.assertEqual(self.c.get(f"/api/session/{sid}/overview").status_code, 404)


if __name__ == "__main__":
    unittest.main()
