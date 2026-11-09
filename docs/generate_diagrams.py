"""Generates docs/workflow.png and docs/architecture.png  (python docs/generate_diagrams.py)"""
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, color="#13315c", fc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#0b2545", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=fc, fontsize=fs, wrap=True, weight="bold")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))


def workflow():
    fig = Figure(figsize=(13, 4.2)); ax = fig.add_subplot(111); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    steps = ["Video Input\n(upload / webcam)", "Frame\nExtraction\n(OpenCV)", "Pre-\nprocessing\n(resize, CLAHE)", "YOLOv8\nDetection\n(+ByteTrack)",
             "Vehicle\nCounting\n(per class)", "Density\nClass (PCU)\nLow/Med/High", "Congestion\nPrediction\n(ML / trend)", "Adaptive\nSignal\nTiming", "Dashboard\n& Reports\n(SQLite)"]
    w = 1.25
    for i, s in enumerate(steps):
        x = 0.25 + i * 1.43
        box(ax, x, 1.4, w, 1.4, s, color="#00b4d8" if i in (3, 6, 7) else "#13315c", fs=8)
        if i: arrow(ax, x - 0.18, 2.1, x, 2.1)
    ax.text(6.5, 3.7, "System Workflow", ha="center", fontsize=14, weight="bold", color="#0b2545")
    ax.text(6.5, 0.7, "Highlighted blocks are the AI components", ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "workflow.png", dpi=130, bbox_inches="tight")


def architecture():
    fig = Figure(figsize=(11, 7)); ax = fig.add_subplot(111); ax.set_xlim(0, 11); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(5.5, 7.05, "System Architecture", ha="center", fontsize=14, weight="bold", color="#0b2545")
    box(ax, .4, 5.5, 10.2, 1.2, "PRESENTATION  -  Browser: HTML / CSS / Bootstrap / JavaScript\nHome · Upload · Live · Dashboard · Density · Prediction · Signals · Reports", "#1b4b82", fs=9)
    box(ax, .4, 3.95, 10.2, 1.2, "APPLICATION  -  Flask app.py\nroutes · validation · MJPEG streaming · JSON API · PDF/CSV export", "#13315c", fs=9)
    for i, t in enumerate(["video_processor\nthreads, frame loop", "detector\nYOLOv8 + tracking", "density\nPCU + classes", "predictor\nGradBoost / trend", "signal_optimizer\nadaptive green"]):
        box(ax, .4 + i * 2.06, 2.15, 1.9, 1.35, t, "#00b4d8", fs=8.5)
    box(ax, .4, .35, 3.2, 1.3, "SQLite  database/traffic.db\nsessions · roads · samples · signal_plans", "#13315c", fs=8.5)
    box(ax, 3.9, .35, 3.2, 1.3, "Files\nuploads/ (videos)\nstatic/outputs/ (snapshots)", "#13315c", fs=8.5)
    box(ax, 7.4, .35, 3.2, 1.3, "Reporting  (matplotlib)\ncharts · PDF · CSV", "#13315c", fs=8.5)
    for i in range(5): arrow(ax, 1.35 + i * 2.06, 3.95, 1.35 + i * 2.06, 3.52)
    for x in (1.35, 5.47, 9.59): arrow(ax, x, 2.15, x, 1.68)
    arrow(ax, 5.5, 5.5, 5.5, 5.15)
    fig.savefig(OUT / "architecture.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    workflow(); architecture(); print("diagrams written to", OUT)
