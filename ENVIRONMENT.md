# Validated Environment

Linux x86_64, Python3.8.20, PyTorch2.0.1 built with CUDA11.7, cuDNN8.5.0.
The tested Python packages are pinned in `requirements.txt`.

Create the Python 3.8.20 environment using the README setup commands, then run
`python -m pip install -r requirements.txt`. The Linux default pip build of
PyTorch 2.0.1 uses CUDA 11.7, as documented in the
[official installation instructions](https://pytorch.org/get-started/previous-versions/#v201).
Run `python -m pip check` and the README CUDA-availability check before training.
A CPU-only PyTorch installation cannot run this example. The Python package
requirements do not install an NVIDIA driver or a system CUDA toolkit.
CUDA11.7 was used for server validation;
other CUDA builds have not been asserted equivalent. Use a GPU with enough
memory for frozen LLaMA-2-7B and input-gradient training at batch16 (the validated
server has 32GiB GPUs). No distributed training is required.

These versions were used for the complete training and testing validation of
the supplied configuration.
