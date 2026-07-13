# Stock Signal ML Pipeline

An end-to-end, config-driven machine-learning pipeline that turns your
pseudocode into runnable code. It pulls **free public market data** (Yahoo
Finance via `yfinance` — no API key required), engineers technical indicators,
trains several classifiers to predict next-day direction, evaluates them, and
emits a **BUY / SELL / HOLD** signal with **ATR-based risk management**.

```
Load OHLCV  ->  Clean  ->  Feature engineering  ->  Label (up/down)
   ->  Prepare (drop NaN, normalize, 70/15/15 split)
   ->  Feature selection (Random Forest importance, top-N)
   ->  Train (XGBoost, Random Forest, LSTM, Logistic Regression)
   ->  Evaluate (accuracy / precision / recall / F1 / ROC-AUC)  ->  pick best
   ->  Predict latest bar  ->  Signal + risk plan (stop = 2*ATR, target = 3*ATR)
   ->  Retrain monthly
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# LSTM is optional; install the CPU build of torch to enable it:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Everything is controlled by `config.yaml` (ticker, dates, top-N features,
model hyperparameters, signal thresholds, risk settings).

```bash
# Train all models, evaluate, pick the best, and save artifacts/
python -m stockpipe train --config config.yaml

# Generate today's signal from the latest data using the saved best model
python -m stockpipe predict --config config.yaml

# Retrain on the newest data (run this monthly) and archive a snapshot
python -m stockpipe retrain --config config.yaml
```

Quick overrides without editing the config:

```bash
python -m stockpipe train --ticker MSFT --start 2016-01-01 --top-n 15
python -m stockpipe predict --ticker MSFT
```

### Example `predict` output

```json
{
  "ticker": "AAPL",
  "as_of": "2024-05-31",
  "model": "xgboost",
  "probability_up": 0.78,
  "signal": "BUY",
  "entry_price": 192.25,
  "atr": 3.41,
  "stop_loss": 185.43,
  "take_profit": 202.48,
  "risk_amount": 100.0,
  "position_size": 14.66
}
```

`position_size` is the number of shares/units that risks `risk_per_trade`
(default 1%) of `account_equity` given a stop `atr_stop_mult * ATR` away.

## What each piece maps to (your pseudocode → code)

| Pseudocode section        | Module |
|---------------------------|--------|
| Load / Clean data         | `stockpipe/data.py` |
| Feature Engineering       | `stockpipe/features.py` |
| Create Labels             | `stockpipe/labels.py` |
| Prepare Dataset           | `stockpipe/dataset.py` |
| Feature Selection         | `stockpipe/feature_selection.py` |
| Train Models              | `stockpipe/models.py` |
| Evaluate / Choose Best    | `stockpipe/evaluate.py`, `stockpipe/pipeline.py` |
| Prediction Loop / Signal  | `stockpipe/pipeline.py`, `stockpipe/risk.py` |
| Risk Management           | `stockpipe/risk.py` |
| Continuous Learning       | `stockpipe/pipeline.py::retrain` |

Artifacts (saved model, scaler, selected features, metrics, metadata) are
written to `artifacts/`.

## Automating the loops

The pseudocode's "Every Market Close" and "Every Month" loops are just
scheduled invocations. For example, with cron:

```cron
# Predict shortly after the US market close (approx. 21:10 UTC, weekdays)
10 21 * * 1-5  cd /path/to/repo && .venv/bin/python -m stockpipe predict >> predict.log 2>&1

# Retrain on the 1st of every month
0 2 1 * *      cd /path/to/repo && .venv/bin/python -m stockpipe retrain >> retrain.log 2>&1
```

## ⚠️ Important disclaimer

This is an educational/research tool, **not financial advice**. Predicting
next-day direction from technical indicators is extremely hard; out-of-sample
accuracy near a coin flip is normal. Some engineered features (raw EMA /
Bollinger price levels) are non-stationary and can degrade as price drifts —
consider using return/ratio-based features for live trading. Do not trade real
money based on this without your own rigorous backtesting and risk controls.
