import argparse
import importlib
import time

import tensorflow.compat.v1 as tf
import sys
tf.disable_v2_behavior()
from flearn.utils.model_utils import read_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# GLOBAL PARAMETERS
OPTIMIZERS = ['FedSGD', 'FedAvg', 'Finetuning', 'Local', 'L2SGD', 'Ditto', 'FedCHAR', 'FedCHAR-DC']
DATASETS = ['HARBox', 'IMU', 'Depth', 'UWB', 'FMCW', 'WISDM', 'MobiAct']


MODEL_PARAMS = {

    'HARBox': (5, ),
    'IMU': (3, ),
    'Depth': (5, ),
    'UWB': (2, ),
    'FMCW': (6, ),
    'WISDM': (6, ),
    'MobiAct': (7, )
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer',
                        type=str,
                        choices=OPTIMIZERS,
                        default='FedCHAR')
    parser.add_argument('--dataset',
                        help='name of dataset',
                        type=str,
                        choices=DATASETS,
                        default='IMU')
    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        default='dnn')
    parser.add_argument('--eval_every',
                        help='evaluate every ____ communication rounds',
                        type=int,
                        default=1)
    parser.add_argument('--participation_ratio',
                        type=float,
                        default=0.5)
    parser.add_argument('--corrupted_ratio',
                        type=float,
                        default=0.)
    parser.add_argument('--batch_size',
                        help='batch size of local training',
                        type=int,
                        default=5)
    parser.add_argument('--learning_rate',
                        help='learning rate for the inner solver',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for random initialization',
                        type=int,
                        default=0)
    parser.add_argument('--sampling',
                        help='client sampling methods',
                        type=int,
                        default='2')
    parser.add_argument('--attack_type',
                        type=str,
                        default='B')
    parser.add_argument('--Robust',
                        type=int,
                        default=0)

    parser.add_argument('--lamda',
                        help='lambda in the objective',
                        type=float,
                        default=1.)
    parser.add_argument('--dynamic_lamda', # Do not use in our paper.
                        help='whether device-specific lam',
                        type=int,
                        default=0)
    parser.add_argument('--finetune_rounds',
                        type=int,
                        default=0)
    parser.add_argument('--decay_factor',
                        help='learning rate decay for finetuning',
                        type=float,
                        default=1.0)
    parser.add_argument('--initial_rounds',
                        type=int,
                        default=0)
    parser.add_argument('--remain_rounds',
                        type=int,
                        default=50)
    parser.add_argument('--num_of_clusters',
                        type=int,
                        default=1)
    parser.add_argument('--linkage',
                        type=str,
                        default='complete')
    parser.add_argument('--distance',
                        type=str,
                        default='cosine')
    parser.add_argument('--corrupted_seed',
                        help='seed for random select corrupt id',
                        type=int,
                        default=42)
    parser.add_argument('--epoch',
                        type=int,
                        default=2)
    parser.add_argument('--q',# Do not use in this paper.
                        type=int,
                        default=0) 
    parser.add_argument('--recluster_rounds',
                        type=int,
                        default=999)
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    # load selected model

    model_path = '%s.%s.%s' % ('flearn', 'models', parsed['model'])

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    if parsed['optimizer'] in ['L2SGD', 'Ditto', 'FedCHAR', 'FedNew']:
        opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    else:
        opt_path = 'flearn.trainers.%s' % parsed['optimizer']

    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['model_params'] = MODEL_PARAMS[model_path.split('.')[2]]


    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data

    train_path = os.path.join('./data', options['dataset'], 'train')
    test_path = os.path.join('./data', options['dataset'], 'test')
    dataset = read_data(train_path, test_path)


    t = optimizer(options, learner, dataset)
    start_time = time.time()
    t.train()
    end_time = time.time()
    print("Training process spends {} min".format((end_time-start_time)/60))
    
if __name__ == '__main__':
    main()





