# Validated Environment

Linux x86_64, Python3.8.20, PyTorch2.0.1 built with CUDA11.7, cuDNN8.5.0.
The tested Python packages are pinned in `requirements.txt`.

Install a CUDA-enabled PyTorch2.0.1 build appropriate for your NVIDIA driver
before installing the remaining requirements. A CPU-only PyTorch installation
cannot run this example. The Python package requirements do not install an
NVIDIA driver or a system CUDA toolkit. CUDA11.7 was used for server validation;
other CUDA builds have not been asserted equivalent. Use a GPU with enough
memory for frozen LLaMA-2-7B and input-gradient training at batch16 (the validated
server has 32GiB GPUs). No distributed training is required.

These versions were used for the complete training and testing validation of
the supplied configuration.
