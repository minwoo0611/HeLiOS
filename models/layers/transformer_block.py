# Implementation of Efficient Channel Attention ECA block

import numpy as np
import torch.nn as nn
import sptr
import torch
import MinkowskiEngine as ME
from models.layers.transformer import SparseMultiheadSASphereConcat
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class SphereTransformer(nn.Module):
    def __init__(self, dim = 256, num_heads=8, window_size_sphere = np.array([1.8, 1.8, 10.0]), window_size = np.array([4,4,4])):
        super().__init__()
        self.dim = dim
        self.num_heads = 4
        self.indice_key = 'sptr_0'
        self.window_size = window_size.astype(np.float32)
        self.shift_win = False
        self.attn = SparseMultiheadSASphereConcat(
            dim, 
            num_heads=num_heads, 
            indice_key=self.indice_key, 
            window_size=self.window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size= self.window_size / 24, 
            quant_size_sphere=window_size_sphere / 24, 
            rel_query=True, 
            rel_key=True, 
            rel_value=True,
            qkv_bias=True, 
            qk_scale=None, 
            a=0.05*0.25,
        )


    def forward(self, x):
        x_coord = torch.stack([x.C[:, 0], x.C[:, 1], x.C[:, 2], x.C[:, 3]], dim=1)

        input_tensor = sptr.SparseTrTensor(x.F, x_coord, spatial_shape=None, batch_size = None)
        output_tensor = self.attn(input_tensor)
        x = ME.SparseTensor(output_tensor.query_feats, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        return x

class SphereFormerLayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x: ME.SparseTensor):
        # feature descriptor on the global spatial information
        y_sparse = self.avg_pool(x)

        # Apply 1D convolution along the channel dimension
        y = self.conv(y_sparse.F.unsqueeze(1)).squeeze(1)
        # y is (batch_size, channels) tensor

        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor

        y_sparse = ME.SparseTensor(y, coordinate_manager=y_sparse.coordinate_manager,
                                   coordinate_map_key=y_sparse.coordinate_map_key)
        # y must be features reduced to the origin
        return self.broadcast_mul(x, y_sparse)


class SphereFormerBlock(BasicBlock):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dimension=3,
                 idx = 0,
                 radial_window_size=None,
                 expansion_rate=None,
                 voxel_window_size=None
                 ):
        super(SphereFormerBlock, self).__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            dimension=dimension)
        np_expansion = np.array([expansion_rate ** idx, expansion_rate ** idx, 1])
        self.atten = SphereTransformer(dim=planes, num_heads= planes // 16, window_size_sphere=radial_window_size * np_expansion, window_size=voxel_window_size)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.atten(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
