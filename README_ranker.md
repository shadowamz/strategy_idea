# Cross-sectional stock ranker

## Input schema

The input is a wide CSV or Parquet file.

Without sizes, 40 stocks produce 80 quote columns plus `timestamp`:

```text
timestamp,
ask_SAP,bid_SAP,
ask_BMW,bid_BMW,
...
```

With level-1 sizes, 40 stocks produce 160 quote/depth columns plus `timestamp`:

```text
timestamp,
ask_SAP,bid_SAP,ask_size_SAP,bid_size_SAP,
ask_BMW,bid_BMW,ask_size_BMW,bid_size_BMW,
...
```

Example rows:

```csv
timestamp,ask_SAP,bid_SAP,ask_size_SAP,bid_size_SAP,ask_BMW,bid_BMW,ask_size_BMW,bid_size_BMW
2026-01-05 09:00:00,210.36,210.34,1200,1500,88.43,88.41,900,1100
2026-01-05 09:00:10,210.39,210.37,1000,1700,88.42,88.40,1300,800
```

The ticker is the suffix after the prefix. For example:

- `ask_SAP` -> ticker `SAP`
- `bid_size_BMW` -> ticker `BMW`

## Install

```bash
pip install -r requirements_ranker.txt
```

## Train

```bash
python cross_sectional_ranker.py \
  --input quotes.parquet \
  --timestamp-col timestamp \
  --ask-prefix ask_ \
  --bid-prefix bid_ \
  --ask-size-prefix ask_size_ \
  --bid-size-prefix bid_size_ \
  --base-frequency 10s \
  --ranking-frequency 1min \
  --horizon 10min \
  --corr-lookback-days 20 \
  --corr-top-k 5 \
  --target-mode raw \
  --output-dir outputs_ranker
```

To produce a rank every 10 seconds:

```bash
--ranking-frequency 10s
```

To rank volatility-adjusted future returns:

```bash
--target-mode volatility_adjusted
```

If sizes are absent, keep the default prefixes; the script automatically disables size features when no matching columns exist.

## Main features

- spread and spread in basis points;
- 10-second, 1-minute, 5-minute, 10-minute and 30-minute returns;
- return since the open;
- previous-close to current-open return;
- open-to-close returns for the previous three days;
- realized volatility;
- bid/ask depth and order imbalance;
- microprice deviation;
- equal-weight index returns and stock-minus-index returns;
- cross-sectional z-scores;
- positive, negative, inverse-negative and signed correlation peer signals.

For each day, correlation weights are estimated only from prior days.

## Output

```text
outputs_ranker/
  lambdarank_model.txt
  feature_importance.csv
  test_rankings.csv.gz
  metrics.json
  config.json
  tickers.json
```

`test_rankings.csv.gz` contains:

- model score;
- predicted rank;
- realized rank;
- future return;
- relevance label.
