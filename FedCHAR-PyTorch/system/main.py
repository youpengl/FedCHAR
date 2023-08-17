#!/usr/bin/env python
import copy
import torch
import os
os.environ["WANDB_API_KEY"] = "0c132606c9498df7beb452a58eb58bd91b02bd50"
import time
import warnings
import numpy as np
import sys
#sys.path.append('/gemini/code')
import wandb
import yaml
import torch.backends.cudnn as cudnn
import random
from flcore.servers.serverchar import FedCHAR
from flcore.servers.serverchardc import FedCHAR_DC

from flcore.trainmodel.models import *
from utils.config_utils import argparser


def run(args):

    print("Creating server and clients ...")
    start = time.time()

    # Generate args.model

    if args.dataset == 'wisdm':
        args.model = HARCNN(in_channels=3, num_classes=args.num_classes, dim=3008).to(args.device)
    
    else:
        raise NotImplementedError

    print(args.model)

    # select algorithm

    if args.algorithm == "FedCHAR":
        server = FedCHAR(args)

    elif args.algorithm == "FedCHAR_DC":
        server = FedCHAR_DC(args)

    else:
        raise NotImplementedError

    server.train()

    if args.future_test:
        server.func_future_test()

    server.save_results()

    print(f"\nTime cost: {round((time.time()-start)/60, 2)}min.")




def main():
    warnings.simplefilter("ignore")
    args = argparser()
    if args.algorithm == 'FedCHAR':
        yaml_config = yaml.load(open('config/FedCHAR_config.yaml'), Loader=yaml.FullLoader)
    elif args.algorithm == 'FedCHAR_DC':
        yaml_config = yaml.load(open('config/FedCHAR_DC_config.yaml'), Loader=yaml.FullLoader)

    temp = copy.deepcopy(args)
    if args.mode == 'debug':
        vars(args).update(yaml_config)
        vars(args).update(vars(temp))
    elif args.mode == 'single_wandb':
        wandb.init(project=args.project, tags=[args.tag])
        vars(args).update(yaml_config)
        vars(args).update(vars(temp))
        wandb.config.update(vars(args))
        wandb.run.name = args.tag

    else:
        print('Mode invalid!')
        exit()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print('Use device: {}'.format(args.device_id))

    # seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    run(args)

if __name__ == "__main__":
    args = argparser()
    if args.mode == 'single_wandb':
        wandb.login()
    main()