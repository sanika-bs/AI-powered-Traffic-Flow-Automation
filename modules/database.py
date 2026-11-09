"""SQLite persistence layer."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS roads(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    road_name TEXT NOT NULL, filename TEXT, status TEXT NOT NULL DEFAULT 'queued', error TEXT,
    progress INTEGER DEFAULT 0, frames_processed INTEGER DEFAULT 0, total_frames INTEGER DEFAULT 0,
    duration REAL DEFAULT 0, avg_total REAL DEFAULT 0, max_total INTEGER DEFAULT 0,
    avg_pcu REAL DEFAULT 0, max_pcu REAL DEFAULT 0, avg_level TEXT DEFAULT 'Low', peak_level TEXT DEFAULT 'Low',
    avg_car REAL DEFAULT 0, avg_motorcycle REAL DEFAULT 0, avg_bus REAL DEFAULT 0,
    avg_truck REAL DEFAULT 0, avg_bicycle REAL DEFAULT 0,
    unique_total INTEGER DEFAULT 0, unique_json TEXT DEFAULT '{}', tracking INTEGER DEFAULT 0, snapshot TEXT);
CREATE TABLE IF NOT EXISTS samples(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id INTEGER NOT NULL REFERENCES roads(id) ON DELETE CASCADE,
    t REAL, total INTEGER, pcu REAL, level TEXT,
    car INTEGER, motorcycle INTEGER, bus INTEGER, truck INTEGER, bicycle INTEGER);
CREATE INDEX IF NOT EXISTS idx_samples_road ON samples(road_id);
CREATE TABLE IF NOT EXISTS signal_plans(
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    plan_json TEXT NOT NULL, created_at TEXT NOT NULL);
"""
ROAD_COLUMNS = {"status", "error", "progress", "frames_processed", "total_frames", "duration", "avg_total",
                "max_total", "avg_pcu", "max_pcu", "avg_level", "peak_level", "avg_car", "avg_motorcycle",
                "avg_bus", "avg_truck", "avg_bicycle", "unique_total", "unique_json", "tracking", "snapshot"}
SAMPLE_COLS = ("t", "total", "pcu", "level", "car", "motorcycle", "bus", "truck", "bicycle")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def init_db(path):
    global DB_PATH
    DB_PATH = str(path)
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with conn() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE roads SET status='interrupted', error='Server stopped during processing' "
                  "WHERE status IN ('queued','processing')")


def create_session(name, source="upload"):
    with conn() as c:
        return c.execute("INSERT INTO sessions(name,source,created_at) VALUES(?,?,?)", (name, source, now())).lastrowid


def add_road(session_id, road_name, filename, status="queued"):
    with conn() as c:
        return c.execute("INSERT INTO roads(session_id,road_name,filename,status) VALUES(?,?,?,?)",
                         (session_id, road_name, filename, status)).lastrowid


def update_road(road_id, **fields):
    bad = set(fields) - ROAD_COLUMNS
    if bad:
        raise ValueError(f"Unknown road columns: {bad}")
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE roads SET {sets} WHERE id=?", (*fields.values(), road_id))


def add_samples(road_id, samples):
    rows = [(road_id, *[s[k] for k in SAMPLE_COLS]) for s in samples]
    with conn() as c:
        c.executemany("INSERT INTO samples(road_id,t,total,pcu,level,car,motorcycle,bus,truck,bicycle) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?)", rows)


def get_samples(road_id):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT t,total,pcu,level,car,motorcycle,bus,truck,bicycle "
                                           "FROM samples WHERE road_id=? ORDER BY t", (road_id,))]


def get_road(road_id):
    with conn() as c:
        r = c.execute("SELECT * FROM roads WHERE id=?", (road_id,)).fetchone()
        return dict(r) if r else None


def get_roads(session_id):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM roads WHERE session_id=? ORDER BY id", (session_id,))]


def get_session(session_id):
    with conn() as c:
        r = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(r) if r else None


def list_sessions(limit=200):
    """Sessions with road count, completed count and combined average PCU."""
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT s.*, COUNT(r.id) AS road_count, "
            "SUM(CASE WHEN r.status='completed' THEN 1 ELSE 0 END) AS done_count, "
            "COALESCE(AVG(CASE WHEN r.status='completed' THEN r.avg_pcu END),0) AS mean_pcu, "
            "COALESCE(SUM(r.unique_total),0) AS unique_total, COALESCE(SUM(r.frames_processed),0) AS frames "
            "FROM sessions s LEFT JOIN roads r ON r.session_id=s.id GROUP BY s.id ORDER BY s.id DESC LIMIT ?", (limit,))]


def latest_session_id():
    with conn() as c:
        r = c.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return r["id"] if r else None


def history_samples(road_name, exclude_session):
    """Earlier samples-lists for the same road name (used as historical training data)."""
    with conn() as c:
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM roads WHERE road_name=? AND session_id!=? AND status='completed' ORDER BY id DESC LIMIT 10",
            (road_name, exclude_session))]
    return [get_samples(i) for i in ids]


def save_signal_plan(session_id, plan):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO signal_plans(session_id,plan_json,created_at) VALUES(?,?,?)",
                  (session_id, json.dumps(plan), now()))


def get_signal_plan(session_id):
    with conn() as c:
        r = c.execute("SELECT plan_json FROM signal_plans WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(r["plan_json"]) if r else None


def delete_session(session_id):
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
