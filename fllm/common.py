"""One effective configuration. No legacy task-level override fields."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def load_config(path):
    config=json.loads(Path(path).read_text())
    expected={'method','source_tsf','target_tsf','llama_checkpoint','plant_seed','observer_seed',
              'plant','lipschitz','observer','evaluation'}
    if set(config)!=expected:
        raise ValueError('Unexpected or missing top-level configuration keys')
    if config['plant']['loss']!='PAPER_L2' or config['observer']['loss']!='L2_PHYSICAL':
        raise ValueError('This release contains only the paper-L2 example')
    if (config['plant']['seq_len'],config['plant']['label_len'],config['plant']['token_len'])!=(48,24,24):
        raise ValueError('Unexpected model dimensions')
    if (config['evaluation']['generated_points_per_step'],config['evaluation']['scored_points'])!=(24,18):
        raise ValueError('The public example scores the first18 of one generated24 patch')
    for key in ['source_tsf','target_tsf','llama_checkpoint']:
        value=config[key]
        if value.startswith('${') and value.endswith('}'):
            name=value[2:-1]
            if name not in os.environ:raise ValueError('Set environment variable '+name)
            config[key]=os.environ[name]
    return config


def model_args(config,output_dir):
    values=dict(config['plant'])
    values.update(use_multi_gpu=False,local_rank=0,gpu=0,llm_ckp_dir=config['llama_checkpoint'])
    return SimpleNamespace(**values)


def identities(config):
    pid=f"{config['plant']['loss']}__WITH_LIP__plant{config['plant_seed']}"
    return pid,pid+f"__obs{config['observer_seed']}__{config['observer']['loss']}"
