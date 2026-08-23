# F1 Race Predictor

This project predicts the complete finishing order for a Formula 1 race. It produces two separate forecasts:

- A pre-weekend forecast based only on information available before the weekend begins.
- A post-qualifying forecast based on the confirmed grid, qualifying performance, practice information, and earlier race history.

The current model is a research MVP. Its results are promising, but it still needs more predictions on genuinely unseen races before its performance can be judged confidently.

## Data

Race results and session timing data come from FastF1 and its Jolpica-F1 interface.

- Historical results begin with the 2023 season.
- Drivers use all available Formula 1 history from that period, so rookies and late arrivals naturally have less history.
- Older seasons receive less weight: 2023 uses `0.25`, 2024 uses `0.50`, 2025 uses `0.75`, and 2026 uses `1.00`.
- Historical constructor assignments are preserved rather than assigning a driver's current team to older races.

Raw source records are preserved, and processed datasets include retrieval details and file hashes so results can be traced back to their inputs.

## How the predictions work

The pre-weekend model uses earlier driver form, teammate comparisons, experience, circuit history, constructor form, and constructor reliability.

The post-qualifying model keeps that information separate and adds:

- Official grid position and known grid changes
- Qualifying position and pace
- Practice laps, sectors, long runs, tyre use, and degradation
- Rolling reliability information
- Data-derived circuit and constructor profiles

The confirmed grid and qualifying information receive 70% of the final post-qualifying score. Practice and earlier history provide the remaining support.

DNFs are excluded from driver pace ratings. They are considered separately when estimating reliability and race uncertainty, so a mechanical failure does not automatically lower the driver's ability rating.

## Current results

The final evaluation uses 2026 rounds 1–6 for model selection and rounds 7–11 as the untouched test races.

| Metric | Result |
| --- | ---: |
| Average Spearman correlation | `0.754` |
| Average position error | `2.84` positions |
| Average Kendall correlation | `0.650` |
| Winner accuracy | `60%` |
| Average podium overlap | `53.3%` |

The post-qualifying model improved average Spearman correlation from `0.707` for the pre-weekend model to `0.754` and performed better in all five test races.

### Top-10 results

Selecting the ten drivers with the highest top-10 probabilities in each test race produced:

| Metric | Result |
| --- | ---: |
| Accuracy | `83.6%` |
| Precision | `82%` |
| Recall | `82%` |
| F1 score | `82%` |
| Brier score | `0.125` |

The model identified 41 of the 50 actual top-10 finishers. The official starting grid identified 40, while the pre-weekend model identified 39.

## Ranking and probability experiments

A pairwise logistic-ranking model was compared with the current scoring model using the same chronological races. It scored `0.746` Spearman compared with `0.754`, so it was not adopted.

Race simulations produce win, podium, top-10, and DNF probabilities. They include individual driver variation, shared constructor variation, reliability, and ordinary race incidents. Simulated outcomes are possible scenarios, not guarantees.

Sigmoid calibration improved win probabilities on both selection and test races. It did not consistently improve podium, top-10, or DNF probabilities, so those outputs continue to use the raw simulation probabilities.

## Project layout

- `src/f1_race_predictor/` contains data collection, normalization, features, training, evaluation, prediction, and archive code.
- `data/` contains source snapshots and reproducible processed datasets.
- `models/` contains trained model artifacts and version records.
- `predictions/` contains immutable prediction records and later evaluations.
- `notebooks/` contains exploratory analysis.
- `tests/` contains the important data, leakage, ordering, model, and archive checks.

## Installation

Python 3.14 is currently supported.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Main commands

```powershell
f1-ingest --help
f1-normalize --help
f1-features --help
f1-evaluate --help
f1-train --help
f1-weekend-data --help
f1-grid-snapshot --help
f1-post-qualifying-features --help
f1-train-post-qualifying --help
f1-probabilities --help
f1-predict --help
f1-register-model --help
f1-check
```

Each command accepts explicit input and output locations. Running `f1-check` executes formatting, linting, type-checking, and the test suite.

## Limitations

- The final test currently contains only five races.
- The top-10 improvement over using the official grid alone is modest.
- Qualifying and grid position have a strong influence on the post-qualifying forecast.
- Podium prediction is less reliable than top-10 prediction.
- Reliability and race incidents cannot be predicted with certainty.
- Driver and constructor performance are separated where possible, but the available data cannot remove every car-related influence.

The next meaningful evaluation is to save forecasts for upcoming races before their cutoffs and measure them after the official results are available.
