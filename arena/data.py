"""Market data: daily bars, FX, fundamentals and headlines, all converted to GBP."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class Market:
    date: pd.Timestamp                 # latest trading date in the data
    opens: pd.DataFrame                # GBP, raw (NaN where a market was shut)
    closes: pd.DataFrame               # GBP, forward-filled for marking
    raw_closes: pd.DataFrame           # GBP, not filled
    fundamentals: dict = field(default_factory=dict)
    news: dict = field(default_factory=dict)
    currencies: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    def has_bar(self, ticker: str, date: pd.Timestamp) -> bool:
        return ticker in self.opens.columns and date in self.opens.index and pd.notna(self.opens.at[date, ticker])

    def open_on(self, ticker: str, date: pd.Timestamp) -> float | None:
        return float(self.opens.at[date, ticker]) if self.has_bar(ticker, date) else None

    def close_on(self, ticker: str, date: pd.Timestamp) -> float | None:
        if ticker not in self.closes.columns:
            return None
        s = self.closes[ticker].loc[:date].dropna()
        return float(s.iloc[-1]) if len(s) else None

    def features(self) -> pd.DataFrame:
        """Per-ticker price features as of self.date (GBP terms)."""
        c = self.closes.loc[: self.date]
        rows = {}
        for t in c.columns:
            s = self.raw_closes[t].loc[: self.date].dropna()
            if len(s) < 25:
                continue
            last = s.iloc[-1]

            def ret(n):
                return round((last / s.iloc[-n - 1] - 1) * 100, 1) if len(s) > n else None

            daily = s.pct_change().dropna()
            hi = s.iloc[-252:].max()
            rows[t] = {
                "price_gbp": round(float(last), 2),
                "r1d%": ret(1), "r5d%": ret(5), "r20d%": ret(20), "r60d%": ret(60), "r120d%": ret(120),
                "vol20_ann%": round(float(daily.iloc[-20:].std() * np.sqrt(252) * 100), 1),
                "off_52w_high%": round(float((last / hi - 1) * 100), 1),
                "above_sma50": bool(last > s.iloc[-50:].mean()) if len(s) >= 50 else None,
                "above_sma200": bool(last > s.iloc[-200:].mean()) if len(s) >= 200 else None,
            }
        return pd.DataFrame(rows).T


def _to_gbp_factor(currency: str) -> str:
    return {"GBp": "pence", "GBX": "pence", "GBP": "gbp", "USD": "usd"}.get(currency, "usd")


def fetch_market(tickers: list[str], cached_currencies: dict | None = None, with_extras: bool = True) -> Market:
    import yfinance as yf

    cached_currencies = dict(cached_currencies or {})
    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers + ["GBPUSD=X"], period="1y", interval="1d",
                              auto_adjust=False, group_by="ticker", progress=False, threads=True)
            if raw is not None and not raw.empty:
                break
        except Exception as e:  # network hiccups on CI
            print(f"download attempt {attempt + 1} failed: {e}")
        time.sleep(5 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError("No market data returned")

    raw.index = _naive(raw.index)
    fx = raw["GBPUSD=X"]["Close"].dropna()
    opens, closes, missing = {}, {}, []
    for t in tickers:
        try:
            df = raw[t][["Open", "Close"]].dropna(how="all")
        except KeyError:
            df = pd.DataFrame()
        if df.empty:
            missing.append(t)
            continue
        if t not in cached_currencies:
            try:
                cached_currencies[t] = yf.Ticker(t).fast_info["currency"]
            except Exception:
                cached_currencies[t] = "GBp" if t.endswith(".L") else "USD"
        kind = _to_gbp_factor(cached_currencies[t])
        if t.endswith(".L") and kind == "gbp" and df["Close"].median() > 500:
            kind = "pence"  # Yahoo sometimes labels pence-quoted lines as GBP
        if kind == "pence":
            factor = pd.Series(0.01, index=df.index)
        elif kind == "gbp":
            factor = pd.Series(1.0, index=df.index)
        else:
            factor = 1.0 / fx.reindex(df.index.union(fx.index)).ffill().reindex(df.index)
        opens[t] = df["Open"] * factor
        closes[t] = df["Close"] * factor

    opens_df = pd.DataFrame(opens).sort_index()
    raw_closes = pd.DataFrame(closes).sort_index()
    opens_df.index = _naive(opens_df.index)
    raw_closes.index = _naive(raw_closes.index)
    date = raw_closes.dropna(how="all").index.max()

    fundamentals, news = {}, {}
    if with_extras:
        for t in opens:
            tk = yf.Ticker(t)
            try:
                info = tk.info or {}
                fundamentals[t] = {
                    "name": info.get("shortName"),
                    "sector": info.get("sector"),
                    "pe": _r(info.get("trailingPE")),
                    "fwd_pe": _r(info.get("forwardPE")),
                    "pb": _r(info.get("priceToBook")),
                    "div_yield%": _r(info.get("dividendYield")),
                    "profit_margin%": _r((info.get("profitMargins") or 0) * 100) if info.get("profitMargins") is not None else None,
                }
            except Exception:
                fundamentals[t] = {}
            try:
                news[t] = _parse_news(tk.news or [])
            except Exception:
                news[t] = []

    return Market(date=date, opens=opens_df, closes=raw_closes.ffill(), raw_closes=raw_closes,
                  fundamentals=fundamentals, news=news, currencies=cached_currencies, missing=missing)


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _r(x, n=2):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _parse_news(items: list, max_items: int = 3, max_age_days: int = 4) -> list[dict]:
    out = []
    now = datetime.now(timezone.utc)
    for it in items:
        c = it.get("content", it)
        title = c.get("title")
        ts = c.get("pubDate") or c.get("providerPublishTime")
        try:
            when = (datetime.fromtimestamp(ts, timezone.utc) if isinstance(ts, (int, float))
                    else datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
        except Exception:
            when = None
        if not title or (when and (now - when).days > max_age_days):
            continue
        src = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        out.append({"title": title, "source": src, "date": when.date().isoformat() if when else None})
        if len(out) >= max_items:
            break
    return out
