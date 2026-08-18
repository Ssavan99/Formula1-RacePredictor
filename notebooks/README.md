# Original notebooks (2023)

These are the project's original notebooks, preserved unchanged. They are part
of the project's history and are deliberately not deleted or rewritten.

| Notebook | What it did |
|---|---|
| `scraper.ipynb` | Scraped the Ergast API, formula1.com and Wikipedia into `../Data/*.csv` |
| `dataPreparation.ipynb` | Merged and cleaned those CSVs into `../Data/Final.csv` |
| `modelling.ipynb` | Trained the MLP and SVM classifiers |
| `f1DataVisualization.ipynb` | Exploratory plots (see `../Plots/`) |

## They no longer run, and that is expected

- **The Ergast API shut down in early 2025.** Every URL in `scraper.ipynb`
  returns nothing. The live pipeline uses Jolpica-F1 instead.
- **formula1.com was redesigned**, so the `pd.read_html` qualifying scrape no
  longer parses.
- **The Selenium weather fallback calls `find_element_by_link_text`**, removed
  in Selenium 4.
- `scraper.ipynb` also references `i` before assignment, so it raises
  `NameError` on a clean kernel regardless.

The modern equivalents live in `f1predict/data/`. The original MLP and SVM
architectures are preserved verbatim in `f1predict/models/original.py` and are
still scored in every backtest.

`../Data/Final.csv` is kept because `f1predict/evaluate/original_leak.py` runs
against it to reproduce and quantify the original evaluation defects — see
`../results/honest_baseline.md`.
