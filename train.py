"""
    Training interface of MelHuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
    Reference: (https://github.com/s3prl/s3prl/)
    Reference author: Shu-wen (Leo) Yang (https://github.com/leo19941227), Andy T. Liu (https://github.com/andi611)
"""
import os
import yaml
import glob
import random
import argparse
from shutil import copyfile
from argparse import Namespace
import torch
import numpy as np
from runner import Runner

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--runner_config', help='The yaml file for configuring the whole experiment, except the upstream model')
    parser.add_argument('-g', '--upstream_config', help='The yaml file for the upstream model')
    parser.add_argument('-n', '--expdir', help='Save experiment at this path')
    parser.add_argument('-f', '--frame_period', default=20, choices=[10,20], type=int)
    # Options
    parser.add_argument('-i', '--initial_weight', help='Initialize model with a specific weight')
    parser.add_argument('--init_optimizer_from_initial_weight', default=False, action='store_true', help='Initialize optimizer when set to True')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')

    args = parser.parse_args()

    os.makedirs(args.expdir, exist_ok=True)
    assert args.runner_config != None and args.upstream_config != None, 'Please specify .yaml config files.'

    with open(args.runner_config, 'r') as file:
        runner_config = yaml.load(file, Loader=yaml.FullLoader)

    copyfile(args.runner_config, f'{args.expdir}/config_runner.yaml')
    copyfile(args.upstream_config, f'{args.expdir}/config_model.yaml')

    return args, runner_config

def main():
    args, runner_config = get_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    runner = Runner(args, runner_config)
    runner.train()
    runner.logger.close()

if __name__ == '__main__':
    main()
