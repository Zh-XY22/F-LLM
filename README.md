# F-LLM: Monthly source-to-target forecasting

This repository contains the runnable AutoTimes-based F-LLM implementation for the fixed M3 Monthly to M4 Monthly task. The source model learns a numeric projection and output projection around a frozen LLaMA-2-7B backbone. A residual observer is then trained from source-only plant outputs and used with feedback factor 1.0 on the target task.

## Fixed configuration

- Source: M3 Monthly TSF file.
- Target: M4 Monthly TSF file.
- Context: 48 values.
- Numeric patch length: 24 values.
- Plant: AutoTimes S0, plant seed 2022.
- Plant projection: 24 -> 512 -> 4096 and 4096 -> 512 -> 24, tanh activation, dropout 0.1.
- Backbone: frozen LLaMA-2-7B with the AutoTimes numeric projection and output projection.
- Plant training: SMAPE objective, batch 16, Adam, learning rate 0.00001, weight decay 0, AMP, cosine scheduler with T=10, at most 10 epochs, patience 3, source validation checkpoint selection.
- Observer: 24 -> 192 -> 24, tanh activation, dropout 0.1, observer seed 2027.
- Observer training: source residual MSE, batch 256, AdamW, learning rate 0.0005, weight decay 0, cosine scheduler with T=5, exactly 5 epochs.
- Evaluation: two generated 24-value steps, score the first 18 target values only, report RAW and observer-corrected SMAPE.

The plant uses the same per-context normalization as AutoTimes. The observer receives the normalized historical residual, and its output is rescaled before being added to the plant forecast. Target future values are used only by the final scorer and by source training labels.

## Data and weights

Provide standard `.tsf` files through environment variables. The source file must contain M3 Monthly series and the target file must contain the 48,000 M4 Monthly series. Provide the LLaMA checkpoint through `LLAMA_CHECKPOINT`. The data and LLaMA weights are not part of this repository.

```bash
export M3_MONTHLY_TSF=/path/to/m3_monthly_dataset.tsf
export M4_MONTHLY_TSF=/path/to/m4_monthly_dataset.tsf
export LLAMA_CHECKPOINT=/path/to/llama-2-7b
```

## Install

```bash
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation compatible with the supplied LLaMA checkpoint. The run binds the model to the visible CUDA device selected by the environment.

## Run

From the repository root:

```bash
python scripts/run_monthly_m3_to_m4_h18.py \
  --config config/monthly_m3_to_m4_h18.json \
  --output-dir outputs/monthly_m3_to_m4_h18
```

The command trains the source plant, writes its checkpoint, builds the source residual cache, trains the observer, writes the fifth-epoch observer checkpoint, evaluates RAW and corrected forecasts, and writes `test/metrics.json`.

To run an individual stage after its dependencies exist:

```bash
python scripts/run_monthly_m3_to_m4_h18.py --stage plant
python scripts/run_monthly_m3_to_m4_h18.py --stage observer
python scripts/run_monthly_m3_to_m4_h18.py --stage test
```

## Outputs

The selected output directory contains the plant training record and checkpoint, source residual arrays and sample keys, observer training metrics and checkpoint, target series keys, RAW and corrected prediction arrays, and `test/metrics.json`. The metrics file records series count, valid point count, SMAPE, MSE, and MAE for both forecast modes.

## Attribution

The AutoTimes backbone follows the public AutoTimes implementation and uses its LLaMA numeric projection, frozen language model, output projection, normalization, and training conventions. See the original project and paper for attribution and citation requirements:

```text
https://github.com/thuml/AutoTimes
```

