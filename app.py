"""Flask application - AI-Powered Traffic Flow Optimization System."""
import json
import os
import time
import uuid
from pathlib import Path

import cv2
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import config
from modules import database as db
from modules import density, reports
from modules.detector import DetectorError, create_detector
from modules.service import Analytics
from modules.signal_optimizer import optimize
from modules.video_processor import JobManager, LiveCamera, VideoError


def create_app(test_config=None, detector_factory=None):
    app = Flask(__name__)
    app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
                      DATABASE=str(config.DATABASE), UPLOAD_DIR=str(config.UPLOAD_DIR))
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    db.init_db(app.config["DATABASE"])

    factory = detector_factory or create_detector
    jobs = JobManager(factory)
    svc = Analytics(jobs)
    jobs.on_session_done = svc.on_session_done
    live = LiveCamera(factory)
    app.extensions.update(jobs=jobs, svc=svc, live=live)

    # ------------------------------------------------------------ helpers
    def pick_session():
        sid = request.args.get("session", type=int) or db.latest_session_id()
        if sid and not db.get_session(sid):
            abort(404)
        return sid

    def page(template, **kw):
        sid = pick_session()
        return render_template(template, sid=sid, sessions=db.list_sessions(), roads=config.ROADS, cfg=config, **kw)

    def allowed(fn):
        return "." in fn and fn.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

    @app.template_filter("pcu")
    def _pcu(v):
        return f"{float(v):.1f}"

    # ------------------------------------------------------------ pages
    @app.route("/")
    def index():
        sessions = db.list_sessions(8)
        stats = {"sessions": len(db.list_sessions()), "vehicles": sum(s["unique_total"] for s in sessions),
                 "frames": sum(s["frames"] for s in sessions)}
        return render_template("index.html", sid=db.latest_session_id(), sessions=sessions, stats=stats)

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html", sid=db.latest_session_id(), roads=config.ROADS,
                                   exts=sorted(config.ALLOWED_EXTENSIONS), max_mb=config.MAX_UPLOAD_MB)
        name = (request.form.get("session_name") or "").strip()[:80] or f"Analysis {db.now()}"
        saved, errors = [], []
        for road in config.ROADS:
            f = request.files.get(f"video_{road.lower()}")
            if not f or not f.filename:
                continue
            if not allowed(f.filename):
                errors.append(f"{road}: '{f.filename}' is not a supported video type ({', '.join(sorted(config.ALLOWED_EXTENSIONS))}).")
                continue
            ext = f.filename.rsplit(".", 1)[1].lower()
            base = secure_filename(f.filename.rsplit(".", 1)[0]) or "video"
            fname = f"{uuid.uuid4().hex[:8]}_{base}.{ext}"
            path = Path(app.config["UPLOAD_DIR"]) / fname
            f.save(path)
            cap = cv2.VideoCapture(str(path))
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if not ok:
                path.unlink(missing_ok=True)
                errors.append(f"{road}: '{f.filename}' could not be read as a video (corrupt or unsupported codec).")
                continue
            saved.append((road, fname, path))
        for e in errors:
            flash(e, "danger")
        if not saved:
            if not errors:
                flash("Please choose at least one traffic video.", "warning")
            return redirect(url_for("upload"))
        sid = db.create_session(name)
        for road, fname, path in saved:
            rid = db.add_road(sid, road, fname)
            jobs.start(rid, sid, road, path)
        if len(saved) < 2:
            flash("Only one road uploaded - signal optimization needs at least two roads (you can still use the what-if simulator).", "info")
        return redirect(url_for("dashboard", session=sid))

    @app.route("/dashboard")
    def dashboard():
        return page("dashboard.html")

    @app.route("/live")
    def live_page():
        return render_template("live.html", sid=db.latest_session_id(), running=live.running)

    @app.route("/density")
    def density_page():
        return page("density.html")

    @app.route("/prediction")
    def prediction_page():
        return page("prediction.html")

    @app.route("/signals")
    def signals_page():
        return page("signals.html")

    @app.route("/reports")
    def reports_page():
        return page("reports.html")

    # ------------------------------------------------------------ JSON API
    @app.route("/api/session/<int:sid>/overview")
    def api_overview(sid):
        ov = svc.overview(sid)
        if ov is None:
            return jsonify(error="Session not found"), 404
        return jsonify(ov)

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        data = request.get_json(silent=True) or {}
        roads = data.get("roads")
        if not isinstance(roads, dict) or not 2 <= len(roads) <= 8:
            return jsonify(error="Provide 2-8 roads as {road: {car, motorcycle, bus, truck, bicycle}}"), 400
        scores = {}
        for name, counts in roads.items():
            try:
                clean = {k: int(counts.get(k, 0)) for k in config.CLASS_NAMES}
            except (TypeError, ValueError, AttributeError):
                return jsonify(error=f"Invalid counts for road '{name}'"), 400
            if any(v < 0 or v > 500 for v in clean.values()):
                return jsonify(error=f"Counts for '{name}' must be between 0 and 500"), 400
            scores[str(name)[:20]] = density.pcu_of(clean)
        plan = optimize(scores)
        return jsonify(plan)

    @app.route("/api/session/<int:sid>/delete", methods=["POST"])
    def api_delete(sid):
        if not db.get_session(sid):
            return jsonify(error="Session not found"), 404
        jobs.cancel_session(sid)
        for r in db.get_roads(sid):
            if r["filename"] and r["filename"] != "webcam":
                (Path(app.config["UPLOAD_DIR"]) / r["filename"]).unlink(missing_ok=True)
            if r["snapshot"]:
                (config.SNAPSHOT_DIR / r["snapshot"]).unlink(missing_ok=True)
        time.sleep(0.3)
        db.delete_session(sid)
        return jsonify(ok=True)

    # ------------------------------------------------------------ streams
    @app.route("/stream/road/<int:road_id>")
    def stream_road(road_id):
        road = db.get_road(road_id)
        if not road:
            abort(404)
        job = jobs.get(road_id)

        def gen():
            last = None
            while True:
                jpg = job.latest_jpeg if job else None
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                if job is None or (job.finished and jpg is last):
                    break
                time.sleep(0.05)

        if job is None:                                   # finished in an earlier run -> peak snapshot
            if road["snapshot"]:
                return send_from_directory(config.SNAPSHOT_DIR, road["snapshot"])
            abort(404)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snapshot/<int:road_id>")
    def snapshot(road_id):
        road = db.get_road(road_id)
        if not road or not road["snapshot"]:
            abort(404)
        return send_from_directory(config.SNAPSHOT_DIR, road["snapshot"])

    # ------------------------------------------------------------ live webcam
    @app.route("/api/live/start", methods=["POST"])
    def live_start():
        cam = (request.get_json(silent=True) or {}).get("camera", 0)
        if not isinstance(cam, int) or not 0 <= cam <= 9:
            return jsonify(error="Camera index must be an integer 0-9"), 400
        try:
            live.start(cam)
        except (VideoError, DetectorError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(ok=True)

    @app.route("/api/live/stop", methods=["POST"])
    def live_stop():
        live.stop()
        sid = live.save_to_history()
        return jsonify(ok=True, session=sid)

    @app.route("/api/live/stats")
    def live_stats():
        samples = live.snapshot_samples()
        pred = svc.predictor.predict(samples, use_ml=False) if samples else None
        return jsonify(running=live.running, error=live.error, now=live.current, prediction=pred,
                       unique=sum(len(v) for v in live.unique.values()),
                       series=[{"t": s["t"], "pcu": s["pcu"]} for s in samples[-120:]])

    @app.route("/live_feed")
    def live_feed():
        def gen():
            last = None
            while live.running:
                jpg = live.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ------------------------------------------------------------ charts & reports
    @app.route("/chart/<int:sid>/<kind>.png")
    def chart(sid, kind):
        if kind not in reports.CHARTS:
            abort(404)
        data = svc.report_data(sid)
        if data is None:
            abort(404)
        return Response(reports.chart_png(kind, data), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.route("/report/<int:sid>.pdf")
    def report_pdf(sid):
        data = svc.report_data(sid)
        if data is None:
            abort(404)
        return Response(reports.build_pdf(data), mimetype="application/pdf",
                        headers={"Content-Disposition": f"attachment; filename=traffic_report_{sid}.pdf"})

    @app.route("/report/<int:sid>.csv")
    def report_csv(sid):
        data = svc.report_data(sid)
        if data is None:
            abort(404)
        return Response(reports.build_csv(data), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=traffic_data_{sid}.csv"})

    # ------------------------------------------------------------ errors
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        flash(f"Upload too large. Maximum total size is {config.MAX_UPLOAD_MB} MB.", "danger")
        return redirect(url_for("upload"))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="Page or session not found.", sid=None), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("error.html", code=500, message="Something went wrong on the server. Check the terminal log.", sid=None), 500

    return app


if __name__ == "__main__":
    application = create_app()
    print("\n  Traffic Flow Optimization System -> http://127.0.0.1:5000\n")
    application.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
