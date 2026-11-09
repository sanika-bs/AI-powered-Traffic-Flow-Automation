"""Matplotlib charts, downloadable PDF report and CSV export (thread-safe: uses Figure, not pyplot)."""
import csv
import io

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import config

LEVEL_COLOR = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
ROAD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def _empty(msg):
    fig = Figure(figsize=(7, 3.4))
    fig.add_subplot(111).text(0.5, 0.5, msg, ha="center", va="center", color="#777")
    fig.axes[0].axis("off")
    return fig


def fig_timeline(data):
    roads = [r for r in data["roads"] if data["samples"].get(r["name"])]
    if not roads:
        return _empty("No samples yet")
    fig = Figure(figsize=(8, 3.8))
    ax = fig.add_subplot(111)
    for i, r in enumerate(roads):
        s = data["samples"][r["name"]]
        ax.plot([x["t"] for x in s], [x["pcu"] for x in s], label=r["name"], color=ROAD_COLORS[i % 5], lw=1.6)
    ax.axhline(config.LOW_MAX_PCU, color="#f39c12", ls="--", lw=1)
    ax.axhline(config.MEDIUM_MAX_PCU, color="#e74c3c", ls="--", lw=1)
    ax.set(title="Traffic load over time (PCU per frame)", xlabel="Video time (s)", ylabel="PCU")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def fig_density(data):
    roads = [r for r in data["roads"] if r["frames"]]
    if not roads:
        return _empty("No samples yet")
    fig = Figure(figsize=(6, 3.6))
    ax = fig.add_subplot(111)
    vals = [r["avg"]["pcu"] for r in roads]
    bars = ax.bar([r["name"] for r in roads], vals, color=[LEVEL_COLOR[r["avg"]["level"]] for r in roads])
    for b, r in zip(bars, roads):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), r["avg"]["level"], ha="center", va="bottom", fontsize=9)
    ax.set(title="Average traffic density per road", ylabel="Average PCU"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def fig_classes(data):
    roads = [r for r in data["roads"] if r["frames"]]
    if not roads:
        return _empty("No samples yet")
    fig = Figure(figsize=(6, 3.6))
    ax = fig.add_subplot(111)
    bottom = [0] * len(roads)
    for k, c in zip(config.CLASS_NAMES, ["#3498db", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6"]):
        v = [r["now"]["counts"][k] for r in roads]
        ax.bar([r["name"] for r in roads], v, bottom=bottom, label=k, color=c)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.set(title="Vehicle categories (avg per frame)", ylabel="Vehicles"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def fig_signals(data):
    plan = data.get("plan")
    if not plan:
        return _empty("Signal plan needs at least two analysed roads")
    fig = Figure(figsize=(6, 3.6))
    ax = fig.add_subplot(111)
    names = [r["road"] for r in plan["roads"]]
    x = range(len(names))
    ax.bar([i - .2 for i in x], [r["fixed_green"] for r in plan["roads"]], .4, label="Fixed (equal split)", color="#95a5a6")
    ax.bar([i + .2 for i in x], [r["green"] for r in plan["roads"]], .4, label="Adaptive (AI)", color="#27ae60")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set(title="Green time per road (s)", ylabel="Seconds"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def fig_prediction(data):
    roads = [r for r in data["roads"] if r.get("prediction") and r["frames"]]
    if not roads:
        return _empty("No samples yet")
    fig = Figure(figsize=(6, 3.6))
    ax = fig.add_subplot(111)
    names = [r["name"] for r in roads]
    x = range(len(names))
    ax.bar([i - .2 for i in x], [r["prediction"]["current_pcu"] for r in roads], .4, label="Current", color="#3498db")
    ax.bar([i + .2 for i in x], [r["prediction"]["predicted_pcu"] for r in roads], .4, label="Predicted",
           color=[LEVEL_COLOR[r["prediction"]["level"]] for r in roads])
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set(title="Current vs predicted traffic load (PCU)", ylabel="PCU"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


CHARTS = {"timeline": fig_timeline, "density": fig_density, "classes": fig_classes,
          "signals": fig_signals, "prediction": fig_prediction}


def chart_png(kind, data):
    fig = CHARTS[kind](data)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return buf.getvalue()


def build_pdf(data):
    buf = io.BytesIO()
    sess, tot = data["session"], data["totals"]
    with PdfPages(buf) as pdf:
        fig = Figure(figsize=(8.27, 11.69))
        t = fig.text
        t(.08, .94, "AI-Powered Traffic Flow Optimization System", fontsize=17, weight="bold", color="#0b2545")
        t(.08, .91, f"Traffic Analysis Report - {sess['name']}", fontsize=12)
        t(.08, .89, f"Generated for session #{sess['id']} created {sess['created_at']}", fontsize=9, color="#666")
        y = .84
        t(.08, y, f"Overall density: {tot['level']}   |   Mean load: {tot['mean_pcu']} PCU   |   "
                  f"Unique vehicles (tracked): {tot['unique_vehicles']}", fontsize=10)
        y -= .04
        t(.08, y, "Per-road results", fontsize=12, weight="bold")
        for r in data["roads"]:
            y -= .035
            p = r["prediction"] or {}
            t(.08, y, f"{r['name']:<7} avg {r['avg']['total']} veh / {r['avg']['pcu']} PCU  -> {r['avg']['level']:<6}  "
                      f"predicted: {p.get('level', '-')} ({p.get('trend', '-')})  status: {r['status']}",
              fontsize=9, family="monospace")
        plan = data.get("plan")
        y -= .05
        t(.08, y, "Recommended signal plan (simulation)", fontsize=12, weight="bold")
        if plan:
            for row in plan["roads"]:
                y -= .033
                t(.08, y, f"{row['road']:<7} green {row['green']:>3}s  ({row['share_pct']}% of budget)  "
                          f"density {row['level']}", fontsize=9, family="monospace")
            y -= .04
            t(.08, y, f"Cycle {plan['cycle']}s | est. avg wait fixed {plan['avg_wait_fixed']}s -> adaptive "
                      f"{plan['avg_wait_adaptive']}s ({plan['reduction_pct']}% lower, model estimate)", fontsize=9)
            y -= .03
            t(.08, y, plan["note"], fontsize=7.5, color="#777")
        else:
            y -= .033
            t(.08, y, "Needs at least two analysed roads.", fontsize=9)
        pdf.savefig(fig)
        for kind in CHARTS:
            pdf.savefig(CHARTS[kind](data))
    return buf.getvalue()


def build_csv(data):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["road", "t_seconds", "total_vehicles", "pcu", "density", *config.CLASS_NAMES])
    for road, samples in data["samples"].items():
        for s in samples:
            w.writerow([road, s["t"], s["total"], s["pcu"], s["level"], *[s[k] for k in config.CLASS_NAMES]])
    out.write("\n")
    w.writerow(["# signal plan", "road", "green_s", "density", "avg_pcu"])
    for row in (data.get("plan") or {}).get("roads", []):
        w.writerow(["", row["road"], row["green"], row["level"], row["score"]])
    return out.getvalue()
