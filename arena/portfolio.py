"""Paper portfolios: valuation, target-weight rebalancing with costs."""
from __future__ import annotations

import pandas as pd

from .data import Market


def new_portfolio(cash: float) -> dict:
    return {"cash": cash, "positions": {}, "costs_paid": 0.0, "trades": 0, "turnover": 0.0}


def value_at_close(p: dict, m: Market, date: pd.Timestamp) -> float:
    v = p["cash"]
    for t, sh in p["positions"].items():
        px = m.close_on(t, date)
        if px is not None:
            v += sh * px
    return v


def weights_at_close(p: dict, m: Market, date: pd.Timestamp) -> dict:
    total = value_at_close(p, m, date)
    out = {}
    for t, sh in p["positions"].items():
        px = m.close_on(t, date)
        if px is not None and total > 0:
            out[t] = round(sh * px / total, 4)
    return out


def rebalance_at_open(p: dict, targets: dict, m: Market, date: pd.Timestamp,
                      cost: float, min_trade_fraction: float) -> list[dict]:
    """Move portfolio toward target weights at `date`'s open.

    Tickers with no bar on `date` (market shut) are left untouched.
    Positions not named in targets are sold.
    """
    # Price each name: today's open if trading, else last close (for valuation only)
    names = set(p["positions"]) | set(targets)
    px_open = {t: m.open_on(t, date) for t in names}
    valuation_px = {t: (px_open[t] if px_open[t] is not None else m.close_on(t, date)) for t in names}
    total = p["cash"] + sum(sh * (valuation_px[t] or 0) for t, sh in p["positions"].items())
    min_trade = total * min_trade_fraction

    sells, buys = [], []
    for t in names:
        px = px_open[t]
        if px is None or px <= 0:
            continue
        cur_val = p["positions"].get(t, 0.0) * px
        diff = targets.get(t, 0.0) * total - cur_val
        if abs(diff) < min_trade and not (targets.get(t, 0.0) == 0 and cur_val > 0 and cur_val < min_trade):
            continue
        (buys if diff > 0 else sells).append((t, diff, px))

    fills = []
    for t, diff, px in sells:
        notional = min(-diff, p["positions"][t] * px)
        sh = notional / px
        p["positions"][t] -= sh
        if p["positions"][t] * px < 0.01:
            p["positions"].pop(t)
        fee = notional * cost
        p["cash"] += notional - fee
        _book(p, notional, fee)
        fills.append({"ticker": t, "side": "SELL", "gbp": round(notional, 2), "price": round(px, 4)})

    need = sum(diff * (1 + cost) for _, diff, _ in buys)
    scale = min(1.0, p["cash"] / need) if need > 0 else 1.0
    for t, diff, px in buys:
        notional = diff * scale
        if notional < 1:
            continue
        fee = notional * cost
        p["positions"][t] = p["positions"].get(t, 0.0) + notional / px
        p["cash"] -= notional + fee
        _book(p, notional, fee)
        fills.append({"ticker": t, "side": "BUY", "gbp": round(notional, 2), "price": round(px, 4)})
    p["cash"] = max(p["cash"], 0.0) if abs(p["cash"]) < 1e-6 else p["cash"]
    return fills


def _book(p, notional, fee):
    p["costs_paid"] += fee
    p["trades"] += 1
    p["turnover"] += notional


def clean_targets(raw: dict, universe: list[str], cap: float) -> dict:
    """Keep known tickers, clip negatives and per-name cap, scale down if over 100%."""
    out = {}
    for t, w in (raw or {}).items():
        t = str(t).upper().strip()
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if t in universe and w > 0:
            out[t] = min(w, cap)
    s = sum(out.values())
    if s > 1.0:
        out = {t: w / s for t, w in out.items()}
    return {t: round(w, 4) for t, w in out.items() if w >= 0.005}
