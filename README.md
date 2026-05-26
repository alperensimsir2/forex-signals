# Forex Daily Signals

Backend pipeline that computes daily BUY/SELL signals for 28 major forex
pairs, using the same strategy stack as
[sp500-signals](https://github.com/your-org/sp500-signals): Primary 15/4 Pivot
Confluence + Secondary PSAR Trend Continuation.

Runs locally or on GitHub Actions cron. Writes static JSON for the mobile app.

## Layout

```
.
├── .github/workflows/daily-signals.yml   # cron (Mon–Fri 22:00 UTC)
├── forex_pairs.json                      # 28 pairs: symbol, EODHD code, display, category
├── schema/
│   ├── signals.schema.json               # signals-forex.json contract
│   └── pair.schema.json                  # per-pair history contract
├── src/
│   ├── indicators.py                     # same as stocks pipeline
│   ├── strategies.py                     # same as stocks pipeline
│   ├── consensus.py                      # same as stocks pipeline
│   ├── pips.py                           # pip_size(), change_pips()
│   ├── fetch.py                          # EODHD forex EOD
│   ├── pipeline.py                       # orchestrator
│   └── __main__.py                       # `python -m src`
├── cache/                                # Parquet OHLC per pair (gitignored)
└── out/
    ├── signals-forex.json                # home-screen payload
    └── pairs/{SYMBOL}.json               # per-pair history
```

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your EODHD API key (same variable as the stocks pipeline):

**PowerShell**

```powershell
$env:EODHD_API_KEY = "your_key_here"
```

**bash**

```bash
export EODHD_API_KEY=your_key_here
```

3. First run — build extended history cache (~28 API calls, targets 5 years via `from`/`to`):

```bash
python -m src --backfill
# optional: python -m src --backfill --backfill-years 5
```

Prints per-pair bar counts, start dates, and flags pairs with less than 3 years of data.

4. Daily run — refresh from EODHD and recompute signals:

```bash
python -m src
```

Outputs:

- `out/signals-forex.json` — all pairs with signals, pips, and 52-week metrics
- `out/pairs/EURUSD.json` — per-pair history for detail screens

Optional env vars: `CACHE_DIR` (default `cache`), `OUT_DIR` (default `out`).

## Forex-specific behavior

- **EODHD**: `GET /api/eod/{PAIR}.FOREX?period=d` — uses `close` for indicators;
  `adjusted_close` is stored as `adj_close` but equals `close`.
- **Volume**: not fetched, stored, or output. Strategy indicators do not use volume
  (price pivots, MACD, Stochastic, PSAR are all H/L/C or close-only).
- **Precision**: `close` and sparkline values keep full API precision (no
  stock-style rounding to 4 decimals).
- **Pips**: `change_pips` and `pip_size` on each signal row; JPY pairs use
  `0.01`, others `0.0001`.

## Volume audit (strategy code)

Confirmed in `indicators.py` and `strategies.py`: no references to volume.
Legacy ADX/ATR/Bollinger helpers in `indicators.py` also use only H/L/C or
close — none use volume.

## Cron schedule

Workflow runs at **22:00 UTC Monday–Friday** (`0 22 * * 1-5`). That is roughly
5pm ET during standard time (with buffer after the 5pm forex daily close). We
may adjust after confirming when EODHD finalizes the forex daily candle.

**v1 note:** Forex trades Sunday evening through Friday evening ET. The
Mon–Fri cron skips Sunday’s partial session; validate Sunday/Friday bar dates
during testing.

## Forex scanner (Phase S1)

Observational technical conditions on the latest daily bar (not trade signals):

```bash
python -m src.scanner
```

Writes `out/forex_scans.json` with `scans` (base conditions), `composites` (multi-condition setups with optional per-pair `quality`), and `pairs` (inverse index).

Validate composites over full cached history:

```bash
python -m src.composite_validation
```

## Character diagnostic (Phase F3a)

Trend vs mean-reversion vs random walk on daily cached OHLC:

```bash
python -m src.character_diagnostic
```

## Exit experiment (Phase F2b)

Compare alternative exit rules on **fixed baseline entries** (no API calls):

```bash
python -m src.exit_experiment
```

Prints a comparison table (all trades + older/newer half by entry date), flags
suspiciously strong results, and writes the best both-halves rule to
`out/forex_trades_improved.csv`.

## Export trade log for chart inspection

After backfill, export every trade from cached history (no API calls) for
TradingView lookup:

```bash
python -m src.export_trades
```

Writes `out/forex_trades.csv` (sorted by pair, then entry date) and prints
per-pair give-back lag summaries (peak_date → exit_date).

## Deploy

Not wired in this phase — run and verify locally first, then connect CI/deploy
when ready.
