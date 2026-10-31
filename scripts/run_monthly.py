"""Single-example commands. Nothing runs on import."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description='F-LLM Monthly source training and H18 evaluation')
    parser.add_argument('command',choices=['prepare-source','train-plant','cache','train-observer','evaluate'])
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'config/monthly_m3_to_m4_h18.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plant-checkpoint',type=Path)
    parser.add_argument('--observer-checkpoint',type=Path)
    args=parser.parse_args()
    from fllm.common import load_config,identities
    from fllm.runtime import read,write,lease
    config=load_config(args.config);root=args.output_dir.resolve()
    code_root=Path(__file__).resolve().parents[1]
    if root==code_root or code_root in root.parents:raise ValueError('Use an output directory outside the source tree')
    root.mkdir(parents=True,exist_ok=True)
    with lease(root):
        if (root/'config.json').exists():
            if read(root/'config.json')!=config:raise ValueError('Existing output has a different effective configuration')
        else:write(root/'config.json',config)
    pid,oid=identities(config)
    if args.command=='prepare-source':
        from fllm.data import prepare_source
        with lease(root/'data'):prepare_source(root)
    elif args.command=='train-plant':
        from fllm.train import plant
        plant(root,dict(task_id=pid,depends_on='SOURCE_DATA'))
    elif args.command=='cache':
        from fllm.source import build_cache
        build_cache(root,pid)
    elif args.command=='train-observer':
        from fllm.source import train_observer
        train_observer(root,dict(task_id=oid,plant_id=pid))
    else:
        from fllm.data import prepare_target
        from fllm.evaluate import evaluate,reject_existing_evaluation
        with lease(root):
            reject_existing_evaluation(root)
            with lease(root/'data'):prepare_target(root)
            plant_checkpoint=args.plant_checkpoint or root/'tasks'/pid/'best_adapter.pt'
            observer_checkpoint=args.observer_checkpoint or root/'tasks'/oid/f"epoch{config['observer']['selected_epoch']:02d}.pt"
            evaluate(root,plant_checkpoint,observer_checkpoint)


if __name__=='__main__':main()
