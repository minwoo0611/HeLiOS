
import torch.nn as nn
import torch

from models.helios import HeLiOS
from misc.utils import ModelParams
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.layers.transformer_block import SphereFormerBlock
from models.heliosfn import HeLiOS_FN



def model_factory(model_params: ModelParams):
    in_channels = 1
    block_module = SphereFormerBlock
    backbone = HeLiOS_FN(in_channels=in_channels, out_channels=model_params.feature_size, 
                            radial_window_size=model_params.radial_window_size, expansion_rate=model_params.expansion_rate, 
                            voxel_window_size=model_params.voxel_window_size, num_top_down=model_params.num_top_down, 
                            conv0_kernel_size=model_params.conv0_kernel_size,
                            block=block_module, layers=model_params.layers, planes=model_params.planes)

    if model_params.model == 'HeLiOS':
        model = HeLiOS(backbone=backbone, in_dim=model_params.feature_size, num_clusters=model_params.num_clusters, cluster_dim=model_params.cluster_dim, use_token=model_params.use_token)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

    return model

