import argparse
import torch
import sys
import os

launch_dir = os.getcwd()
if launch_dir not in sys.path:
    sys.path.append(launch_dir)
    sys.path.append(launch_dir + "/docker/thirdparty/SparseTransformer")

from training.trainer import do_train
from misc.utils import TrainingParams
from datasets.base_datasets import TrainingTuple

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HeLiOS model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train(params)
