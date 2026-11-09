"""Traffic density measurement and Low / Medium / High classification."""
import config


def pcu_of(counts):
    """Passenger-car-unit load for a {class: count} dict."""
    return round(sum(config.PCU.get(k, 1.0) * v for k, v in counts.items()), 2)


def classify(pcu):
    if pcu <= config.LOW_MAX_PCU:
        return "Low"
    if pcu <= config.MEDIUM_MAX_PCU:
        return "Medium"
    return "High"


def percent(pcu):
    """0-100 gauge value (100 = 1.5x the Medium/High boundary)."""
    return round(min(100.0, pcu / (config.MEDIUM_MAX_PCU * 1.5) * 100), 1)


def describe(counts):
    counts = {k: int(counts.get(k, 0)) for k in config.CLASS_NAMES}
    pcu = pcu_of(counts)
    return {"total": sum(counts.values()), "pcu": pcu, "level": classify(pcu),
            "percent": percent(pcu), "counts": counts}
