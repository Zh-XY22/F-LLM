# F-LLM

F-LLM is a control-theoretic framework for time-series forecasting with large language models. This repository provides the complete runnable pipeline for the method: TSF data loading, source plant training, residual-cache construction, observer training, feedback inference, and target evaluation.

## Repository layout

- `fllm/`: F-LLM plant, residual observer, cache, feedback inference, and evaluation code.
- `autotimes/`: the bundled numeric projection, frozen language-model wrapper, TSF loader, and training utilities required by the released pipeline.
- `scripts/`: command-line entry point for the complete run and for each individual stage.
- `config/`: the single reproducibility configuration released with this repository.

## Released example configuration

The only included configuration is `config/monthly_m3_to_m4_h18.json`. It specifies the Monthly M3-to-M4 H18 example used to exercise the complete source-to-target forecasting pipeline. This example is provided for reproducibility and does not restrict the datasets or forecast horizons to which F-LLM can be applied.

The example uses a 48-value context, 24-value numeric patches, two generated 24-value steps, and scores the first 18 target values. The source plant is trained with a frozen LLaMA-2-7B language model and learnable numeric projection layers. A residual observer is trained from source-only plant outputs and applied through feedback during target inference.

## Data and model checkpoint

The released code reads standard `.tsf` files and does not redistribute datasets or language-model weights. Set the paths used by the released configuration:

```bash
export M3_MONTHLY_TSF=/path/to/m3_monthly_dataset.tsf
export M4_MONTHLY_TSF=/path/to/m4_monthly_dataset.tsf
export LLAMA_CHECKPOINT=/path/to/llama-2-7b
```

The source and target files must match the example configuration, and the checkpoint directory must contain the LLaMA model used by the numeric projection wrapper.

## Install

```bash
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation compatible with the supplied LLaMA checkpoint.

## Run

From the repository root:

```bash
python scripts/run_monthly_m3_to_m4_h18.py \
  --config config/monthly_m3_to_m4_h18.json \
  --output-dir outputs/monthly_m3_to_m4_h18
```

The complete command trains the source plant, writes its checkpoint, builds the source residual cache, trains the observer, runs RAW and feedback-corrected target inference, and writes `test/metrics.json`.

Individual stages can be run after their dependencies exist:

```bash
python scripts/run_monthly_m3_to_m4_h18.py --stage plant
python scripts/run_monthly_m3_to_m4_h18.py --stage observer
python scripts/run_monthly_m3_to_m4_h18.py --stage test
```

## Outputs

The output directory contains the plant training record and checkpoint, source residual arrays and sample keys, observer training metrics and checkpoint, target series keys, RAW and corrected prediction arrays, and `test/metrics.json`. The metrics file records SMAPE, MSE, and MAE for both forecast modes.

## Attribution

The `autotimes/` directory includes bundled model and data utilities that follow the public AutoTimes implementation. It is included here as an execution component of F-LLM. Please cite and comply with the upstream project when using those components:

```text
https://github.com/thuml/AutoTimes
```
