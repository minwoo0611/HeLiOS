# Warsaw University of Technology

import os
import configparser
import time
import numpy as np
import sys

from datasets.quantization import PolarQuantizer, CartesianQuantizer

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        assert self.coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.coordinates}'

        if 'polar' in self.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = tuple([float(e) for e in params['quantization_step'].split(',')])
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
        elif 'cartesian' in self.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = params.getfloat('quantization_step')
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
        else:
            raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        # Use cosine similarity instead of Euclidean distance
        # When Euclidean distance is used, embedding normalization is optional
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)

        # Size of the local features from backbone network (only for MinkNet based models)
        self.feature_size = params.getint('feature_size', 256)
        if 'planes' in params:
            self.planes = tuple([int(e) for e in params['planes'].split(',')])
        else:
            self.planes = tuple([32, 64, 64])

        if 'layers' in params:
            self.layers = tuple([int(e) for e in params['layers'].split(',')])
        else:
            self.layers = tuple([1, 1, 1])

        self.num_top_down = params.getint('num_top_down', 1)
        self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
        self.radial_window_size = np.array([float(e) for e in params['radial_window_size'].split(',')])
        self.voxel_window_size = np.array([float(e) for e in params['voxel_window_size'].split(',')]) 
        self.expansion_rate = params.getfloat('expansion_rate', 1.0)


        self.num_clusters = params.getint('num_clusters', 8)
        self.cluster_dim = params.getint('cluster_dim', 32)
        self.use_token = params.getboolean('use_token', False)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')

class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.model_name = params.get('model_name', 'default_model_name')
        self.save_freq = params.getint('save_freq', 10)          # Model saving frequency (in epochs)
        self.num_workers = params.getint('num_workers', 0)

        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)
        # When batch_split_size is non-zero, multistage backpropagation is enabled
        self.batch_split_size = params.getint('batch_split_size', None)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.optimizer = params.get('optimizer', 'Adam')
        # self.scheduler = None
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        scheduler_milestones = params.get('scheduler_milestones')
        self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]

        self.weight_decay = params.getfloat('weight_decay', None)

        # Similarity measure: based on cosine similarity or Euclidean distance
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.set_aug_mode = params.getint('set_aug_mode', 1)    # Augmentation mode (1 is default)
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)
        self.eval_db_file = params.get('eval_db_file', None)
        self.eval_query_file = params.get('eval_query_file', None)

        params = config['LOSS']
        # Number of best positives (closest to the query) to consider
        self.positives_per_query = params.getint("positives_per_query", 4)
        # Temperatures (annealing parameter) and numbers of nearest neighbours to consider
        self.tau1 = params.getfloat('tau1', 0.01)
        self.margin = params.getfloat('margin', None)    # Margin used in loss function
        self.p_sp_distance = params.getfloat('p_sp_distance', 0.02)    # Distance between positive and selected positive
        self.sp_n_distance = params.getfloat('sp_n_distance', 0.19)    # Distance between selected positive and negative


        self.truncated_weight = params.getfloat('truncated_weight', 0.8)
        self.guided_p_sp_weight = params.getfloat('guided_p_sp_weight', 0.1)
        self.guided_sp_n_weight = params.getfloat('guided_sp_n_weight', 0.1)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')

