import torch
import torch.nn as nn
import MinkowskiEngine as ME
import sptr
import numpy as np
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.resnet import ResNetBase
from models.layers.transformer import SparseMultiheadSASphereConcat

class SphereTransformer(nn.Module):
    """Transformer module with sparse multi-head self-attention and spherical window constraints."""
    def __init__(self, dim=256, num_heads=8, window_size_sphere=np.array([1.8, 1.8, 10.0]), window_size=np.array([4, 4, 4])):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size.astype(np.float32)
        self.window_size_sphere = window_size_sphere
        self.indice_key = 'sptr_0'
        self.shift_win = False

        self.attn = SparseMultiheadSASphereConcat(
            embed_dim=self.dim,
            dim=dim,
            num_heads=num_heads,
            indice_key=self.indice_key,
            window_size=self.window_size,
            window_size_sphere=self.window_size_sphere,
            quant_size=self.window_size / 24,
            quant_size_sphere=self.window_size_sphere / 24,
            rel_query=True,
            rel_key=True,
            rel_value=True,
            qkv_bias=True,
            qk_scale=None,
            a=0.05 * 0.25
        )

    def forward(self, x):
        # Prepare input tensor for attention
        coords = x.C[:, [0, 1, 2, 3]]  # Extract coordinates
        input_tensor = sptr.SparseTrTensor(x.F, coords, spatial_shape=None, batch_size=None)
        
        # Apply attention
        output_tensor = self.attn(input_tensor)
        
        # Return sparse tensor with updated features
        return ME.SparseTensor(
            features=output_tensor.query_feats,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

class HeLiOS_FN(ResNetBase):
    """Feature Pyramid Network (FPN) using Minkowski ResNet blocks with spherical transformer attention."""
    def __init__(self, in_channels, out_channels, radial_window_size, expansion_rate, voxel_window_size,
                 num_top_down=1, conv0_kernel_size=5, block=BasicBlock, layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes), "Layers and planes must have the same length"
        assert len(layers) >= 1, "At least one layer is required"
        assert 0 <= num_top_down <= len(layers), "Invalid number of top-down layers"

        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        self.radial_window_size = radial_window_size
        self.expansion_rate = expansion_rate
        self.voxel_window_size = voxel_window_size

        super().__init__(in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        self.convs = nn.ModuleList()  # Bottom-up convolutions (stride=2)
        self.bn = nn.ModuleList()     # Bottom-up batch norms
        self.blocks = nn.ModuleList() # Bottom-up residual blocks
        self.attentions = nn.ModuleList() # Attention modules
        self.tconvs = nn.ModuleList() # Top-down transposed convolutions
        self.conv1x1 = nn.ModuleList() # Lateral 1x1 convolutions

        # Initial convolution
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=self.inplanes,
            kernel_size=self.conv0_kernel_size,
            dimension=D
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        # Bottom-up layers
        for idx, (plane, layer) in enumerate(zip(self.planes, self.layers)):
            self.convs.append(ME.MinkowskiConvolution(
                in_channels=self.inplanes,
                out_channels=self.inplanes,
                kernel_size=2,
                stride=2,
                dimension=D
            ))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(
                block=self.block,
                planes=plane,
                blocks=layer,
                idx=idx,
                radial_window_size=self.radial_window_size,
                expansion_rate=self.expansion_rate,
                voxel_window_size=self.voxel_window_size
            ))

        # Top-down and lateral connections
        idx = self.num_bottom_up - 2
        for i in range(self.num_top_down):
            expansion = np.array([self.expansion_rate ** idx, self.expansion_rate ** idx, 1])
            self.conv1x1.append(ME.MinkowskiConvolution(
                in_channels=self.planes[-1 - i],
                out_channels=self.lateral_dim,
                kernel_size=1,
                stride=1,
                dimension=D
            ))
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.lateral_dim,
                out_channels=self.lateral_dim,
                kernel_size=2,
                stride=2,
                dimension=D
            ))
            self.attentions.append(SphereTransformer(
                dim=self.lateral_dim,
                num_heads=self.lateral_dim // 16,
                window_size_sphere=self.radial_window_size * expansion,
                window_size=self.voxel_window_size
            ))
            idx -= 1

        # Final lateral connection
        last_plane = self.planes[-1 - self.num_top_down] if self.num_top_down < self.num_bottom_up else self.planes[0]
        self.conv1x1.append(ME.MinkowskiConvolution(
            in_channels=last_plane,
            out_channels=self.lateral_dim,
            kernel_size=1,
            stride=1,
            dimension=D
        ))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def wrap(self, x, F):
        """Wrap features into a sparse tensor."""
        return ME.SparseTensor(
            features=F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x._manager
        )

    def forward(self, x):
        feature_maps = []

        # Initial convolution
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # Bottom-up pass
        for idx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Downsample
            x = bn(x)
            x = self.relu(x)
            x = block(x)
            if self.num_bottom_up - 1 - self.num_top_down <= idx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down, "Incorrect number of feature maps"

        # Apply first lateral connection
        x = self.conv1x1[0](x)

        # Top-down pass
        for idx, (tconv, attention) in enumerate(zip(self.tconvs, self.attentions)):
            x = tconv(x)  # Upsample
            xconv = self.conv1x1[idx + 1](feature_maps[-idx - 1])
            xatten = attention(xconv)
            x = x + xatten

        return x