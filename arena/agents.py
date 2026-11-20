"""The workers. Each returns a decision: {"targets": {...} | None, "reasoning": str, "note": str}.

targets=None means "hold, no change".
"""
from __future__ import annotations

import json
import os
import random
import re

import pandas as pd

from .data import Market

RULES = """Rules of the competition:
- Paper money only, in GBP. Long-only. No leverage, no shorting.
- You set TARGET WEIGHTS for your whole portfolio. Anything you leave out is sold. Cash is whatever is left.
- Max {cap:.0%} in any single ticker. Weights must sum to 1.0 or less.
- Your orders fill at the NEXT trading day's open. Every trade costs {cost:.1%} each way, so churn is expensive.
- You are scored over months against a buy-and-hold global index tracker and a random monkey.
- Only use tickers from the universe provided.

Reply with ONLY a JSON object, no prose before or after:
{{"action": "rebalance" or "hold",
  "targets": {{"TICKER": weight, ...}},   // required if action is rebalance
  "reasoning": "max 80 words: what you did and why",
  "note_to_self": "max 40 words: what to watch next time"}}"""


def decide(agent: dict, portfolio_weights: dict, cash_weight: float, total_value: float,
           m: Market, cfg: dict, memory: list[str], rng: random.Random, offline: bool = False) -> dict:
    kind = agent["kind"]
    universe = list(m.opens.columns)
    if kind == "buy_and_hold":
        if portfolio_weights:
            return {"targets": None, "reasoning": "Holding the index. Never trades.", "note": ""}
        return {"targets": {cfg["benchmark_ticker"]: 1.0}, "reasoning": "Buying the global index tracker with everything.", "note": ""}
    if kind == "random":
        return _monkey(portfolio_weights, m, universe, rng)
    if kind == "llm":
        if offline:
            return _offline_stub(agent, m)
        return _llm(agent, portfolio_weights, cash_weight, total_value, m, cfg, memory)
    raise ValueError(f"unknown agent kind {kind}")


def _monkey(weights: dict, m: Market, universe: list[str], rng: random.Random) -> dict:
    d = m.date
    first_trading_day_of_month = not any(
        (x.month == d.month and x.year == d.year) for x in m.raw_closes.dropna(how="all").index if x < d)
    if weights and not first_trading_day_of_month:
        return {"targets": None, "reasoning": "Monkey is happy with its bananas.", "note": ""}
    picks = rng.sample(universe, k=min(8, len(universe)))
    raw = [rng.random() for _ in picks]
    s = sum(raw)
    targets = {t: min(r / s, 0.2) for t, r in zip(picks, raw)}
    return {"targets": targets, "reasoning": "Threw darts at the board: " + ", ".join(picks) + ".", "note": ""}


def _offline_stub(agent: dict, m: Market) -> dict:
    """Deterministic stand-in used for offline tests only."""
    f = m.features()
    col = {"momentum": "r20d%", "news": "r5d%", "value": "off_52w_high%", "contrarian": "r20d%"}[agent["id"]]
    asc = agent["id"] in ("value", "contrarian")
    picks = f[col].astype(float).sort_values(ascending=asc).index[:6]
    return {"targets": {t: 0.15 for t in picks}, "reasoning": f"[offline stub] top 6 by {col}", "note": "stub"}


def build_prompt(agent: dict, weights: dict, cash_weight: float, total_value: float,
                 m: Market, cfg: dict, memory: list[str]) -> tuple[str, str]:
    system = (f"You are '{agent['name']}', one of several AI portfolio managers in a paper-trading arena.\n"
              f"Your style: {agent['style'].strip()}\n\n"
              + RULES.format(cap=cfg["max_position_weight"], cost=cfg["transaction_cost"]))

    feats = m.features()
    parts = [f"Date: {m.date.date().isoformat()} (after the close).",
             f"Portfolio value: £{total_value:,.0f}. Cash: {cash_weight:.1%}.",
             "Current weights: " + (json.dumps(weights) if weights else "none (all cash)")]

    if "prices" in agent["inputs"]:
        parts.append("Price features (GBP terms, % returns):\n" + feats.to_csv())
    if "fundamentals" in agent["inputs"]:
        fdf = pd.DataFrame(m.fundamentals).T
        if not fdf.empty:
            parts.append("Fundamentals:\n" + fdf.to_csv())
    if "news" in agent["inputs"]:
        lines = []
        for t, items in m.news.items():
            for it in items:
                lines.append(f"{t} | {it.get('date')} | {it.get('source')} | {it['title']}")
        parts.append("Recent headlines (last few days):\n" + ("\n".join(lines) if lines else "none available"))
    if memory:
        parts.append("Your notes to self from previous days (oldest first):\n- " + "\n- ".join(memory))
    parts.append("Decide now. JSON only.")
    return system, "\n\n".join(parts)


def _llm(agent, weights, cash_weight, total_value, m, cfg, memory) -> dict:
    import anthropic

    system, user = build_prompt(agent, weights, cash_weight, total_value, m, cfg, memory)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    last_err = None
    for _ in range(2):
        try:
            resp = client.messages.create(model=cfg["model"], max_tokens=1500, system=system,
                                          messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            data = _parse_json(text)
            usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
            if data.get("action") == "hold":
                return {"targets": None, "reasoning": data.get("reasoning", ""), "note": data.get("note_to_self", ""), "usage": usage}
            return {"targets": data.get("targets") or {}, "reasoning": data.get("reasoning", ""),
                    "note": data.get("note_to_self", ""), "usage": usage}
        except Exception as e:
            last_err = e
    return {"targets": None, "reasoning": f"Error, holding positions: {last_err}", "note": "", "error": True}


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON in reply")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return json.loads(re.sub(r"(?<!:)//[^\n]*", "", match.group(0)))
