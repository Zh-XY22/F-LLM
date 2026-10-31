# F-LLM

F-LLM is a two-stage time-series forecasting method. A numeric encoder and
decoder surround a frozen pretrained language model. Stage 1 trains the numeric
projections with a local decoder sensitivity penalty. Stage 2 learns a correction
from historical, time-aligned forecast residuals. In multi-step prediction the
closed-loop trajectory feeds corrected values back into its own context; the raw
trajectory always feeds raw predictions back.

Paper: [Closing the Loop: A Control-Theoretic Framework for Time Series Forecasting with LLMs](https://arxiv.org/abs/2602.12756).

## Fixed Example

The single public configuration is `config/monthly_m3_to_m4_h18.json`:
M3 Monthly source training and validation, followed by M4 Monthly target testing.
Context length is 48, patch length 24, and only the first 18 generated values are
scored. This one-patch example has identical POSTHOC and CLOSED_LOOP outputs;
it does not by itself measure a feedback benefit.

The backbone is the AutoTimes numeric projection implementation with frozen
LLaMA-2-7B. The encoder and decoder use hidden width 512, Tanh and dropout 0.1.
The plant seed is 2022. Adam uses learning rate 1e-5, batch 16, AMP, cosine10
and at most 10 epochs, with source-validation patience 3. Equality updates the
best checkpoint. Selection uses the original mean of source-validation batch
L2 reconstruction losses, without a target-domain selection step. The selected
epoch is not forced. Gradients pass through the frozen backbone to the encoder;
frozen parameters are not updated.

Stage 1 prediction loss is **PAPER_L2**, not MSE, squared L2, RMSE or SMAPE:
`mean(norm((prediction - truth).reshape(B, -1, 24), 2, dim=-1))`.
Each patch norm is in physical units after plant denormalization; the two patches
are averaged separately. The training loss adds
`418507.34584479616 * mean(relu(ratio - 0.022296765353530645)^2)`.
The ratio uses decoder
output L2 distance divided by latent perturbation L2 norm, protected at 1e-8.
A private random direction is L2-normalized. Its norm is
`0.5 * 0.02 * RMS(latent)`, with the radius detached. Decoder dropout is disabled
only during regularization and its original modes are restored afterwards.
The perturbation noise seed is902021. Regularizer arithmetic is float32 in
decoder output coordinates; its latent computation graph is retained. Constants
are fixed from source-TRAIN gradient-budget calibration at initialization. This sampled local
penalty is not a global stability proof.

The observer is `Linear(24,192) -> Tanh -> Dropout(0.1) -> Linear(192,24)`.
Its input is historical `truth - prediction` residual divided by the current
valid-context standard deviation. Its output is multiplied by that deviation.
There is no added mean. Stage 2 uses **L2_PHYSICAL**: the Euclidean norm of the
masked residual error after restoring physical units, averaged across samples.
It is not MSE. Only source TRAIN cache rows are used for optimization.

Observer seed is 2027. AdamW uses batch256, learning rate 0.0005 and zero weight
decay. Epochs1-5 use cosine5; epochs6-20 use learning rate0.0001. The evaluation
endpoint is epoch5, fixed by source-VAL first18-point forecast SMAPE, not
selected using target results. This does not change either training objective.
This runner retains the full20-epoch training schedule.

## Data and Weights

Use **Linux with a CUDA GPU** and a CUDA-compatible PyTorch installation.
Tested versions and installation notes are in `ENVIRONMENT.md` and
`requirements.txt`. These files do not install a GPU driver or CUDA toolkit.

Provide standard M3 Monthly and M4 Monthly `.tsf` files and local LLaMA-2-7B
weights. They are not included. The TSF parser and native source-window loader
are included. Missing values follow the native removal rule; coordinates after
removal are recorded. Source TRAIN labels end no later than the source VAL future
origin. Observer source VAL predicts24 and reports both first18 and full24.
Target history ends before the final18 labels; those labels are used only for
scoring. All48000 target series are retained. Padding and validity masks are
recorded.

```bash
python -m pip install -r requirements.txt
export M3_MONTHLY_TSF=/your/data/m3_monthly_dataset.tsf
export M4_MONTHLY_TSF=/your/data/m4_monthly_dataset.tsf
export LLAMA_CHECKPOINT=/your/models/llama-2-7b
export CUDA_VISIBLE_DEVICES=0
```

Use a separate output directory; no output is written into this source tree.
The same resolved config is recorded once in that output directory and must
match on subsequent commands. There are no task-level configuration overrides.

## Train and Test

```bash
python scripts/run_monthly.py prepare-source --output-dir ../monthly-output
python scripts/run_monthly.py train-plant --output-dir ../monthly-output
python scripts/run_monthly.py cache --output-dir ../monthly-output
python scripts/run_monthly.py train-observer --output-dir ../monthly-output
python scripts/run_monthly.py evaluate --output-dir ../monthly-output
```

To test existing matched weights without training:

```bash
python scripts/run_monthly.py evaluate --output-dir ../monthly-evaluation \
  --plant-checkpoint /your/weights/best_adapter.pt \
  --observer-checkpoint /your/weights/epoch05.pt
```

## Verified Result

The complete training and testing pipeline was validated with the supplied
configuration and environment on all 48,000 target series (864,000 forecast
values). Source validation selected plant epoch2; evaluation used observer epoch5.

| Model | SMAPE | MSE | MAE |
|---|---:|---:|---:|
| Base predictor (RAW) | 13.857121 | 1971513.903610 | 596.859477 |
| F-LLM | 13.757228 | 1974138.444072 | 592.418012 |

Correction reduces SMAPE and MAE in this configuration, with a small increase
in MSE. H18 covers one patch, so these results measure single-patch correction.

## Outputs

Run the small CPU development tests with
`CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests`.

An output directory containing an evaluation task state is rejected before
target preparation; use a new directory rather than returning old metrics.
External pure weight dictionaries use restricted loading. Only locally generated
`resume.pt` files, which contain Python/NumPy RNG objects, use unrestricted
loading. Never insert an untrusted resume file into a run directory.

Plant adapters exclude the frozen backbone. Observer states, training curves,
source boundary manifests, cache masks and sample keys remain in the specified
output directory. Evaluation writes RAW, POSTHOC and CLOSED_LOOP arrays,
`metrics.json`, `series_keys.json`, `bindings.json` and
`independent_recompute.json` under `tasks/H18_EVALUATION/`.

Scoring uses float64: each series' SMAPE is the mean of
`200 * abs(error) / max(abs(prediction) + abs(truth), 1e-8)` over valid points,
followed by equal series averaging. MSE and MAE use global valid-element sums.
SSE, SAE, series count and valid-element count are also reported. The scoring
denominator uses a minimum epsilon before division; all-zero predictions and
targets contribute zero. Target labels are used only by the scorer, never by
the prediction interface.

## Attribution

The forecasting backbone, numeric projection layers and TSF loader derive from
AutoTimes by THUML, Tsinghua University. The upstream MIT copyright and license
are retained in `LICENSE`. See the upstream AutoTimes project:
https://github.com/thuml/AutoTimes . LLaMA weights must be supplied under their
own applicable license. F-LLM is the forecasting method described here;
AutoTimes is its backbone implementation source.
