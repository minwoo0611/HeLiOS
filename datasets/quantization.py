import numpy as np
from typing import List
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass


class PolarQuantizer(Quantizer):
    def __init__(self, quant_step: List[float]):
        assert len(quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.theta_range = int(360. // self.quant_step[0])


    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(pc[:, 1], pc[:, 0]) * 180./np.pi
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        z = pc[:, 2]
        polar_pc = torch.stack([theta, dist, z], dim=1)
        # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
        polar_pc = polar_pc / self.quant_step
        quantized_polar_pc, ndx = sparse_quantize(polar_pc, quantization_size=1., return_index=True)
        # Return quantized coordinates and indices of selected elements
        return quantized_polar_pc, ndx


class CartesianQuantizer(Quantizer):
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)

        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx

