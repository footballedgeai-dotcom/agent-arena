"""Daily arena cycle.

1. Fill yesterday's orders at today's open.
2. Mark every portfolio to today's close.
3. Ask each agent for its next move (fills at the next open).
4. Save state, logs and the dashboard payload.

Usage:
    python -m arena.run                 # live: real data, Claude agents (needs ANTHROPIC_API_KEY)
    python -m arena.run --offline --days 60 --data-dir /tmp/arena   # synthetic test run
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import agents as ag
from .data import fetch_market
from .portfolio import (clean_targets, new_portfolio, rebalance_at_open, value_at_close, weights_at_close)

ROOT = Path(__file__).resolve().parent.parent


def load_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    u = cfg["universe"]
    cfg["tickers"] = u["uk"] + u["us"] + u["etfs"]
    return cfg


def load_state(path: Path, cfg: dict) -> dict:
    if path.exists():
        st = json.loads(path.read_text())
    else:
        st = {"start_date": None, "last_run_date": None, "currencies": {}, "agents": {}, "history": []}
    for a in cfg["agents"]:
        st["agents"].setdefault(a["id"], {"portfolio": new_portfolio(cfg["starting_cash_gbp"]),
                                          "pending": None, "memory": []})
    return st


def append_jsonl(path: Path, rows: list[dict]):
    if rows:
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def run_day(m, cfg: dict, st: dict, data_dir: Path, offline: bool, force: bool = False) -> bool:
    today = m.date
    today_s = today.date().isoformat()
    if st["last_run_date"] == today_s and not force:
        print(f"Already ran for {today_s}; nothing to do.")
        return False

    rng = random.Random(f"monkey-{today_s}")
    fills_log, decisions_log = [], []

    # 1. fill pending orders at today's open
    for a in cfg["agents"]:
        s = st["agents"][a["id"]]
        pend = s["pending"]
        if pend and pend["decided"] < today_s:
            fills = rebalance_at_open(s["portfolio"], pend["targets"], m, today,
                                      cfg["transaction_cost"], cfg["min_trade_fraction"])
            s["pending"] = None
            for f in fills:
                fills_log.append({"date": today_s, "agent": a["id"], **f})

    # 2. mark to market
    values = {a["id"]: round(value_at_close(st["agents"][a["id"]]["portfolio"], m, today), 2) for a in cfg["agents"]}
    st["history"] = [h for h in st["history"] if h["date"] != today_s] + [{"date": today_s, "values": values}]
    st["start_date"] = st["start_date"] or today_s

    # 3. decide
    for a in cfg["agents"]:
        s = st["agents"][a["id"]]
        p = s["portfolio"]
        total = values[a["id"]]
        weights = weights_at_close(p, m, today)
        cash_w = p["cash"] / total if total else 1.0
        d = ag.decide(a, weights, cash_w, total, m, cfg, s["memory"][-5:], rng, offline=offline)
        cap = 1.0 if a["kind"] == "buy_and_hold" else cfg["max_position_weight"]
        targets = None if d["targets"] is None else clean_targets(d["targets"], list(m.opens.columns), cap)
        if targets is not None:
            s["pending"] = {"decided": today_s, "targets": targets}
        if d.get("note"):
            s["memory"] = (s["memory"] + [f"{today_s}: {d['note']}"])[-10:]
        decisions_log.append({"date": today_s, "agent": a["id"], "action": "hold" if targets is None else "rebalance",
                              "targets": targets, "reasoning": d.get("reasoning", ""), "note": d.get("note", ""),
                              "usage": d.get("usage"), "error": d.get("error", False)})
        print(f"  {a['name']:<22} £{total:>10,.2f}  {'HOLD' if targets is None else 'REBALANCE'}")

    st["last_run_date"] = today_s
    st["currencies"] = m.currencies or st["currencies"]
    append_jsonl(data_dir / "fills.jsonl", fills_log)
    append_jsonl(data_dir / "decisions.jsonl", decisions_log)
    return True


def metrics(series: pd.Series, bench: pd.Series | None) -> dict:
    start = series.iloc[0]
    ret = series.iloc[-1] / start - 1
    dd = (series / series.cummax() - 1).min()
    daily = series.pct_change().dropna()
    sharpe = None
    if len(daily) >= 20 and daily.std() > 0:
        sharpe = round(float(daily.mean() / daily.std() * math.sqrt(252)), 2)
    out = {"return_pct": round(ret * 100, 2), "max_drawdown_pct": round(float(dd) * 100, 2), "sharpe": sharpe}
    if bench is not None:
        out["vs_benchmark_pct"] = round((ret - (bench.iloc[-1] / bench.iloc[0] - 1)) * 100, 2)
    return out


def export_dashboard(cfg: dict, st: dict, m, data_dir: Path, out_path: Path):
    hist = pd.DataFrame([{"date": h["date"], **h["values"]} for h in st["history"]]).set_index("date")
    bench = hist["benchmark"] if "benchmark" in hist else None
    decisions = []
    dpath = data_dir / "decisions.jsonl"
    if dpath.exists():
        decisions = [json.loads(l) for l in dpath.read_text().splitlines() if l.strip()]
    fills = []
    fpath = data_dir / "fills.jsonl"
    if fpath.exists():
        fills = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]

    agents_out = []
    for a in cfg["agents"]:
        s = st["agents"][a["id"]]
        p = s["portfolio"]
        mt = metrics(hist[a["id"]], bench)
        w = weights_at_close(p, m, m.date)
        agents_out.append({
            "id": a["id"], "name": a["name"], "color": a["color"], "kind": a["kind"],
            "blurb": (a.get("style") or a.get("description") or "").strip(),
            "value": hist[a["id"]].iloc[-1], **mt,
            "trades": p["trades"], "costs_paid": round(p["costs_paid"], 2),
            "turnover_x": round(p["turnover"] / cfg["starting_cash_gbp"], 2),
            "cash_pct": round(p["cash"] / hist[a["id"]].iloc[-1] * 100, 1),
            "positions": sorted(([t, round(x * 100, 1)] for t, x in w.items()), key=lambda r: -r[1]),
            "pending": s["pending"],
        })
    payload = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "as_of": st["last_run_date"], "start_date": st["start_date"], "days": len(hist),
        "starting_cash": cfg["starting_cash_gbp"],
        "agents": agents_out,
        "history": {"dates": list(hist.index), "series": {c: [round(v, 2) for v in hist[c]] for c in hist.columns}},
        "feed": [d for d in decisions if d["agent"] not in ("benchmark",)][-80:][::-1],
        "fills": fills[-60:][::-1],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, default=str, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="synthetic data + stub agents")
    ap.add_argument("--days", type=int, default=1, help="offline only: simulate this many days")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--dashboard", default=str(ROOT / "docs" / "data.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    st = load_state(data_dir / "state.json", cfg)

    if args.offline:
        from .fake import fake_market, upto
        full = fake_market(cfg["tickers"])
        dates = full.raw_closes.index[-args.days:]
        for d in dates:
            m = upto(full, d)
            print(d.date())
            run_day(m, cfg, st, data_dir, offline=True, force=args.force)
    else:
        m = fetch_market(cfg["tickers"], st.get("currencies"))
        if m.missing:
            print("No data for:", ", ".join(m.missing))
        print(f"Market date {m.date.date()}")
        run_day(m, cfg, st, data_dir, offline=False, force=args.force)

    (data_dir / "state.json").write_text(json.dumps(st, indent=1, default=str))
    export_dashboard(cfg, st, m, data_dir, Path(args.dashboard))
    print("Saved.")


if __name__ == "__main__":
    main()
