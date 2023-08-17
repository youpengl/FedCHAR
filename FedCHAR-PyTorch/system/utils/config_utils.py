import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='debug', help='Choose mode such as debug, single_wandb')
    parser.add_argument('--project', type=str, default='FedCHAR', help='project name for wandb')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--algorithm', type=str, default='FedCHAR')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default='demo', help='wandb run name')
    args = parser.parse_args()
    return args