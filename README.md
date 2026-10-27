# F-LLM: Monthly M3→M4 H18

This repository contains a compact AutoTimes-based implementation of the Monthly M3→M4 forecasting setting used in the paper.

The model projects time-series patches into the hidden space of a frozen LLaMA-2-7B backbone and maps the hidden states back to numeric forecasts. A lightweight observer learns a correction from the historical residual of the same forecaster. The target is the first 18 points of the M4 Monthly forecast horizon.

## Fixed setting

- source: M3 Monthly
- target: M4 Monthly
- context length: 48
- patch length: 24
- forecast length: 24, scored on the first 18 points
- plant: AutoTimes projection layers with a frozen LLaMA-2-7B backbone
- observer: `24 → 192 → Tanh → Dropout(0.1) → 24`
- plant training seed: `2022`
- observer training seed: `2027`
- feedback factor: `1.0`

The reported reference run for this setting is SMAPE `14.0861` for the corrected forecast and `13.8411` for the uncorrected AutoTimes forecast. The repository provides the training and evaluation path; model weights and datasets must be supplied by the user.

## Data and model files

Prepare the following files locally:

```
data/tsf/m3_monthly_dataset.tsf
data/tsf/m4_monthly_dataset.tsf
weights/llama-2-7b/...
```

The `.tsf` files are the standard AutoTimes M4 files. Each M4 target series must contain its observed values followed by the 18 held-out values used for scoring.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_monthly_m3_to_m4_h18.py \
  --m3 data/tsf/m3_monthly_dataset.tsf \
  --m4 data/tsf/m4_monthly_dataset.tsf \
  --llm-ckp-dir weights/llama-2-7b \
  --output-dir outputs/monthly_m3_m4_h18 \
  --device cuda:0
```

The command trains the source forecaster, trains the observer on source residuals, and evaluates RAW and corrected H18 predictions on all M4 Monthly series. It writes `metrics.json` and the two compact model checkpoints under the selected output directory.

The implementation follows the public AutoTimes model interface. The LLaMA weights remain frozen; only the numeric input and output projections and the observer are optimized.

