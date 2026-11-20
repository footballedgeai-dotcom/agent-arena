# Agent Arena

Four AI traders, a random monkey and an index fund each get £10,000 of **paper money**.
Every weekday night a GitHub Action fills yesterday's orders at today's open, marks everyone
to the close, then asks each agent for its next move. Results are committed to the repo and
shown on a phone-friendly dashboard.

No broker, no real money, nothing is ever traded for real.

## The field

| Agent | How it decides |
|---|---|
| Momentum | Claude, sees price trends only |
| News Sentiment | Claude, sees prices + latest headlines |
| Value | Claude, sees prices + P/E, P/B, yield |
| Contrarian | Claude, sees prices + headlines, buys sell-offs |
| Random Monkey | 8 random names, reshuffled monthly (the control) |
| Benchmark | 100% VWRL.L, bought once and held |

Rules: long-only, max 20% per name, 0.1% cost each way, fills at the next open (no look-ahead),
dividends ignored for everyone. Each LLM agent keeps short "notes to self" between days.

## Setup (once)

1. Push this folder to a new GitHub repo.
2. **Settings → Secrets and variables → Actions → New secret**: `ANTHROPIC_API_KEY`.
3. **Settings → Actions → General → Workflow permissions**: Read and write.
4. **Settings → Pages**: deploy from branch `main`, folder `/docs`.
   (Pages on a private repo needs a paid GitHub plan; otherwise make the repo public.)
5. **Actions → Daily arena run → Run workflow** to kick off day one.

It then runs automatically at 22:30 UTC, Monday–Friday.

## Tweaking

Everything lives in `config.yaml`: universe, costs, position cap, model, and each agent's
personality. Add an agent by adding an entry. A new agent starts with fresh £10k from the day it joins.

## Local test without internet or API key

```
pip install -r requirements.txt
python -m arena.run --offline --days 60 --data-dir /tmp/arena --dashboard /tmp/arena/data.json
```

Uses a synthetic market and stand-in agents to check the accounting.

## Files

- `arena/run.py` — the daily cycle and dashboard export
- `arena/portfolio.py` — valuation and rebalancing with costs
- `arena/agents.py` — prompts and the non-LLM agents
- `arena/data.py` — Yahoo Finance prices, FX to GBP, fundamentals, headlines
- `data/` — state, every decision (`decisions.jsonl`) and every fill (`fills.jsonl`)
- `docs/` — dashboard (`index.html` + generated `data.json`)
