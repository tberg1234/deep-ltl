#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 8
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True
    no_randomize_agent: bool = False
    no_randomize_objects: bool = False
    finetune_from: str | None = None
    finetune_seed: int | None = None


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'SafetyBallNav-v0',
            '--steps_per_process', '4096',
            '--batch_size', '2048',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '30_000_000',
            '--model_config', 'SafetyBallNav-v0',
            '--curriculum', 'SafetyBallNav-v0',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]
        if args.no_randomize_agent:
            command += ['--randomize_agent', 'false']
        if args.no_randomize_objects:
            command += ['--randomize_objects', 'false']
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')
        if args.finetune_from is not None:
            command += ['--finetune_from', args.finetune_from]
        if args.finetune_seed is not None:
            command += ['--finetune_seed', str(args.finetune_seed)]

        print(f"randomize_agent: {not args.no_randomize_agent}, randomize_objects: {not args.no_randomize_objects}")

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 2 --device cpu --name asd --seed 1 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        sys.exit(0)
