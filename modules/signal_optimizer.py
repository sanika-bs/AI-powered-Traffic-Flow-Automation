"""Adaptive signal timing (software simulation only - no real signal hardware is controlled).

Green time is shared in proportion to measured demand (PCU), then clamped to
[MIN_GREEN, MAX_GREEN] by iterative water-filling so no road is ever starved,
and rounded to whole seconds that add up exactly to the green budget.
"""
import math
import config
from modules import density

EPS = 0.5   # smoothing so an empty road still gets a proportional (and non-zero) claim


def optimize(scores, cycle=None, min_green=None, max_green=None, yellow=None):
    """scores: ordered {road: demand_pcu}. Returns a full plan dict."""
    cycle = cycle or config.CYCLE_TIME
    yellow = config.YELLOW_TIME if yellow is None else yellow
    min_green = config.MIN_GREEN if min_green is None else min_green
    max_green = config.MAX_GREEN if max_green is None else max_green
    roads = list(scores)
    n = len(roads)
    if n == 0:
        raise ValueError("At least one road is required")
    budget = cycle - n * yellow
    if budget <= 0:
        raise ValueError("Cycle time too short for the number of roads")
    mn = min(min_green, budget // n)               # always feasible
    mx = max(max_green, math.ceil(budget / n))
    demand = {r: max(float(scores[r]), 0.0) + EPS for r in roads}

    alloc, free, remaining = {}, set(roads), float(budget)
    for _ in range(n + 1):
        if not free:
            break
        td = sum(demand[r] for r in free)
        prop = {r: remaining * demand[r] / td for r in free}
        low = [r for r in free if prop[r] < mn]
        high = [r for r in free if prop[r] > mx]
        if not low and not high:
            alloc.update(prop)
            free = set()
            break
        for r in low:
            alloc[r] = mn
        for r in high:
            alloc[r] = mx
        for r in low + high:
            remaining -= alloc[r]
            free.discard(r)
    green = {r: min(mx, max(mn, int(round(alloc.get(r, mn))))) for r in roads}

    diff, guard = budget - sum(green.values()), 0          # make the sum exact
    while diff != 0 and guard < 10 * budget:
        guard += 1
        if diff > 0:
            cands = sorted((r for r in roads if green[r] < mx), key=lambda r: -demand[r])
            if not cands:
                break
            green[cands[0]] += 1
            diff -= 1
        else:
            cands = sorted((r for r in roads if green[r] > mn), key=lambda r: demand[r])
            if not cands:
                break
            green[cands[0]] -= 1
            diff += 1

    total_cycle = sum(green.values()) + n * yellow
    fixed = budget / n
    real_total = sum(max(float(scores[r]), 0.0) for r in roads)
    weights = {r: (max(float(scores[r]), 0.0) / real_total if real_total > 0 else 1 / n) for r in roads}

    def wait(g):                                            # mean uniform-arrival delay (s)
        red = total_cycle - g - yellow
        return red * red / (2.0 * total_cycle)

    phases, t, rows = [], 0, []
    for r in roads:
        g = green[r]
        phases.append({"road": r, "start": t, "green": g, "yellow": yellow, "red": total_cycle - g - yellow})
        t += g + yellow
        rows.append({"road": r, "score": round(float(scores[r]), 2), "level": density.classify(scores[r]),
                     "green": g, "share_pct": round(100 * g / budget, 1), "fixed_green": round(fixed, 1),
                     "wait_adaptive": round(wait(g), 1), "wait_fixed": round(wait(fixed), 1)})
    w_ad = sum(weights[r] * wait(green[r]) for r in roads)
    w_fx = sum(weights[r] * wait(fixed) for r in roads)
    return {
        "cycle": total_cycle, "yellow": yellow, "green_budget": budget,
        "min_green": mn, "max_green": mx, "phases": phases, "roads": rows,
        "avg_wait_adaptive": round(w_ad, 1), "avg_wait_fixed": round(w_fx, 1),
        "reduction_pct": round(100 * (w_fx - w_ad) / w_fx, 1) if w_fx > 0 else 0.0,
        "note": "Software simulation. Wait times use a simplified uniform-arrival delay model and are estimates, not field measurements.",
    }
