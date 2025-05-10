import numpy as np
import os

from datasets.base_datasets import PointCloudLoader


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        pc = np.fromfile(file_path, dtype=np.float32)

        # coords are within -1..1 range in each dimension
        pc = np.reshape(pc, (pc.shape[0] // 4, 4))
        pc = pc[:, 0:3]

        return pc
