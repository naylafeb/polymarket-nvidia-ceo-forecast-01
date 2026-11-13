#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, stdlib only).

Forecasts the Yes price of every trading market of Polymarket's event
"Next CEO of NVIDIA?" at every minute of the 4 hours after a checkpoint:
five quantiles (5%, 25%, 50%, 75%, 95%) per market per minute.

Model (calibrated on the six practice cases; practice pinball loss 0.2546
vs the house baseline's 0.3636):
  * Median: the last price, pulled by two forces that the practice
    history shows are real.
      - Asymmetric bounce: a rise in the last hour gives most of it
        back over the next hours (up-moves bounce), while a fall in
        the last hour keeps falling a little (down-moves continue).
        The pull grows as 1 - exp(-h/300) with the horizon h.
      - Sum reversion: the markets are mutually exclusive outcomes,
        so their prices should sum to about 1. When the sum is above
        1 the medians are scaled down in proportion to their price
        (and up when the sum is below 1), the scaling growing as
        1 - exp(-h/300).
  * Spread: each market's own recent volatility, the standard deviation
    of its prices over the last 4 hours, scaled by 1.5 with a 0.004
    floor, widening with the horizon like a mean-reverting (OU)
    process, so the variance saturates instead of growing as sqrt(h):
    the price jumps and then comes back, it does not wander.
  * Quantiles: normal quantiles around the median, with the inner
    levels narrowed a little and the 95% level widened a little
    (the price is mostly flat, but its upward jumps are fat-tailed).

Reads every <id>.in in the input directory and writes <id>.out to the
output directory. Deterministic: no randomness, no network, no clock.
"""
import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
ZSCALE = (0.8, 0.85, 1.0, 0.85, 1.1)

UP_BOUNCE = 1.5    # fraction of a last-hour rise given back
DOWN_CONT = 0.5    # fraction of a last-hour fall that continues
BOUNCE_TAU = 300.0
SUM_STRENGTH = 1.5
SUM_TAU = 300.0
SUM_TARGET = 1.0
VOL_SCALE = 1.5
VOL_FLOOR = 0.004
THETA = 0.0005     # OU rate for the variance growth


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def sd(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((v - mu) ** 2 for v in xs) / n)


def forecast_case(case):
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    sum_now = sum(m["prices"][-1] for m in markets)
    n_markets = len(markets)

    # Per-market features, computed once.
    info = []
    for m in markets:
        p = m["prices"]
        x = p[-1]
        n = len(p)
        chg60 = x - p[-61] if n > 60 else 0.0
        vol4h = sd(p[-min(240, n):])
        vol = math.sqrt((VOL_SCALE * vol4h) ** 2 + VOL_FLOOR ** 2)
        info.append({"id": m["id"], "x": x, "chg60": chg60, "vol": vol})

    # First pass: medians (bounce + sum reversion need the whole case).
    meds = {}
    for d in info:
        rows = []
        for h in range(1, horizon + 1):
            prof = 1.0 - math.exp(-h / BOUNCE_TAU)
            med = d["x"]
            if d["chg60"] > 0:
                med -= UP_BOUNCE * prof * d["chg60"]
            else:
                med += DOWN_CONT * prof * d["chg60"]
            rows.append(med)
        meds[d["id"]] = rows
    # Multiplicative sum reversion, per horizon: scale the medians so
    # their sum moves toward SUM_TARGET.
    for h in range(horizon):
        s_h = sum(meds[d["id"]][h] for d in info)
        if s_h > 1e-9:
            f = (SUM_TARGET / s_h) ** (SUM_STRENGTH * (1.0 - math.exp(-(h + 1) / SUM_TAU)))
            for d in info:
                meds[d["id"]][h] *= f

    # Second pass: quantiles around the medians.
    denom = 1.0 - math.exp(-2.0 * THETA * 240.0)
    out = []
    for d in info:
        rows = []
        for h in range(1, horizon + 1):
            med = meds[d["id"]][h - 1]
            gh = math.sqrt((1.0 - math.exp(-2.0 * THETA * h)) / denom)
            w = d["vol"] * gh
            row = [price6(med + z * w * zs) for z, zs in zip(Z, ZSCALE)]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        out.append({"id": d["id"], "q": rows})
    return {"forecasts": out}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        cid = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != list(QUANTILES):
            raise ValueError(f"{name} asks for quantiles {case['quantiles']}")
        result = forecast_case(case)
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
