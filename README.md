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

## Setup

Use **Linux with an NVIDIA CUDA GPU**, Git, curl and Conda. The full pipeline was
validated on a 32 GiB GPU with Python 3.8.20 and PyTorch 2.0.1 / CUDA 11.7.
Use the tested Python environment below for these dependency versions.
See `ENVIRONMENT.md` for platform details; the commands do not install a GPU driver.

```bash
git clone https://github.com/Zh-XY22/F-LLM.git
cd F-LLM
conda create -n fllm-l2 python=3.8.20 pip -y
conda activate fllm-l2
python -m pip install -r requirements.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'A CUDA-enabled PyTorch installation and working NVIDIA driver are required'"
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests
```

Run the remaining commands from the repository root in the same activated shell.
The examples keep input data, model weights and run outputs outside the source tree.

## Data and Weights

Download the complete TSF files from the Monash repository:

| Input | Source | Extracted filename | Series |
|---|---|---|---:|
| M3 Monthly | [Zenodo](https://zenodo.org/records/4656298) | `m3_monthly_dataset.tsf` | 1,428 |
| M4 Monthly | [Zenodo](https://zenodo.org/records/4656480) | `m4_monthly_dataset.tsf` | 48,000 |

```bash
mkdir -p ../fllm-data
curl --fail --location 'https://zenodo.org/records/4656298/files/m3_monthly_dataset.zip?download=1' --output ../fllm-data/m3_monthly_dataset.zip
curl --fail --location 'https://zenodo.org/records/4656480/files/m4_monthly_dataset.zip?download=1' --output ../fllm-data/m4_monthly_dataset.zip
python -m zipfile -e ../fllm-data/m3_monthly_dataset.zip ../fllm-data
python -m zipfile -e ../fllm-data/m4_monthly_dataset.zip ../fllm-data
export M3_MONTHLY_TSF="$(cd ../fllm-data && pwd)/m3_monthly_dataset.tsf"
export M4_MONTHLY_TSF="$(cd ../fllm-data && pwd)/m4_monthly_dataset.tsf"
```

Keep the final 18 values in each complete TSF series. The evaluation code splits
them off as target labels. A standalone M4 training CSV or a TSF with the final
18 values already removed would evaluate a different period.

The TSF parser and source-window loader are included. Missing values follow the
native removal rule; coordinates after removal are recorded. Source TRAIN labels
end no later than the source VAL future origin. Observer source VAL predicts 24
values and reports both the first 18 and full 24. Target history ends before the
final 18 labels; these labels are used only for scoring. All 48,000 target series
are retained, with padding and validity masks recorded.

Use the **base [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)**
model in Hugging Face format. Obtain access on its model page, then authenticate
and download it. The Chat model, GGUF files and original Meta `consolidated.*`
checkpoints are not the model format used by this configuration.

```bash
huggingface-cli login
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="../fllm-models/Llama-2-7b-hf",
    allow_patterns=[
        "config.json", "generation_config.json", "model*.safetensors",
        "model.safetensors.index.json", "LICENSE.txt",
    ],
)
PY
export LLAMA_CHECKPOINT="$(cd ../fllm-models/Llama-2-7b-hf && pwd)"
```

If matching weights are already available, set `LLAMA_CHECKPOINT` to that absolute
directory instead. It must contain `config.json` and all weight shards referenced
by the model index. No separate tokenizer, timestamp embeddings or pretrained
F-LLM adapter is needed to train from scratch.

Set all three input environment variables before running the public commands.
Use a new output directory outside this source tree. The same resolved config
is recorded there and must match on subsequent commands.

## Train and Test

```bash
python scripts/run_monthly.py prepare-source --output-dir ../monthly-output
python scripts/run_monthly.py train-plant --output-dir ../monthly-output
python scripts/run_monthly.py cache --output-dir ../monthly-output
python scripts/run_monthly.py train-observer --output-dir ../monthly-output
python scripts/run_monthly.py evaluate --output-dir ../monthly-output
python -m json.tool ../monthly-output/tasks/H18_EVALUATION/metrics.json
```

Run these five stages in order with the same output directory. `prepare-source`
creates source arrays and sample keys; `cache` creates residual training data
from the newly trained plant. `evaluate` automatically loads that plant's best
source-validation checkpoint and the observer's fixed epoch5 checkpoint.

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

For the same M3 Monthly to M4 Monthly setting, [paper Table 2](https://arxiv.org/html/2602.12756v2#S6.T2)
reports F-LLM SMAPE **14.535**. This configuration's validated **13.757228** is
lower by 0.777772 (about 5.35%). The paper table does not report MSE or MAE for
this setting, so those two metrics are provided here without a paper comparison.

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
