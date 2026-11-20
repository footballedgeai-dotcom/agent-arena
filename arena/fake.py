"""Synthetic market for offline tests. Never used in the live run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market


def fake_market(tickers: list[str], days: int = 320, seed: int = 7) -> Market:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp("2026-11-19"), periods=days)
    opens, closes = {}, {}
    for t in tickers:
        drift, vol = rng.normal(0.0003, 0.0004), rng.uniform(0.01, 0.03)
        start = rng.uniform(5, 300)
        c = start * np.exp(np.cumsum(rng.normal(drift, vol, days)))
        o = c * np.exp(rng.normal(0, vol / 3, days))
        s_c, s_o = pd.Series(c, idx), pd.Series(o, idx)
        if t.endswith(".L"):   # pretend a UK bank holiday
            s_c.iloc[-30] = np.nan
            s_o.iloc[-30] = np.nan
        opens[t], closes[t] = s_o, s_c
    o, c = pd.DataFrame(opens), pd.DataFrame(closes)
    news = {t: [{"title": f"{t} does something newsworthy", "source": "Fake Wire", "date": "2026-11-19"}] for t in tickers[:5]}
    return Market(date=idx[-1], opens=o, closes=c.ffill(), raw_closes=c, news=news,
                  fundamentals={t: {"pe": round(float(rng.uniform(5, 40)), 1)} for t in tickers})


def upto(m: Market, date: pd.Timestamp) -> Market:
    return Market(date=date, opens=m.opens.loc[:date], closes=m.closes.loc[:date],
                  raw_closes=m.raw_closes.loc[:date], fundamentals=m.fundamentals,
                  news=m.news, currencies=m.currencies)
