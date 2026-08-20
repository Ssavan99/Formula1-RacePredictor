# Scripts

Operational and research tooling. None of it runs in CI; the scheduled
workflows only call the `f1predict` package.

| Script | What it is for |
|---|---|
| `backfill_practice.py` | One-shot FastF1 practice backfill, resumable in chunks (~13s/race) |
| `merge_backtests.py` | Combine backtest passes into one summary (walk-forward is deterministic, so passes score identical folds) |
| `run_pass.py` | Backtest a named subset of models — used to split long sweeps |
| `tune_anchor.py` | Sweep the grid-anchor weight on a validation fold inside the training era |
| `probe_llm.py` | Measure where an LLM's knowledge of past results actually ends |
| `score_llm.py` | Score the LLM on the window the probe proved it has not seen |
| `shot.js` | Headless screenshots of the site (puppeteer, serves `docs/` in-process) |

`shot.js` needs `npm install puppeteer@19` first; it is deliberately not a
committed dependency.
