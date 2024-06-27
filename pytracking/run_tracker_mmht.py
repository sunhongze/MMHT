import os
import sys
import argparse
import torch
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0, visdom_info=None):
    visdom_info = {} if visdom_info is None else visdom_info
    dataset = get_dataset(dataset_name)
    if sequence == 'train':
        train_list = [f.strip() for f in open('test.txt', 'r').readlines()]
        dataset = [dataset[i] for i in train_list]
    elif sequence == 'val':
        val_list = [f.strip() for f in open('val.txt', 'r').readlines()]
        dataset = [dataset[i] for i in val_list]
    elif sequence is not None:
        dataset = [dataset[s] for s in sequence]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]
    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name',  type=str, default='fusion', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='mmht_para', help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='fe108', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default='val', help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=False, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    args = parser.parse_args()

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, args.sequence, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port})


if __name__ == '__main__':
    main()

