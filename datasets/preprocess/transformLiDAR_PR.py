import numpy as np
import open3d as o3d
import os
import pandas as pd

def load_transformations(csv_file_path):
    # Read trajectory_transform.csv with timestamp as string
    df_gt = pd.read_csv(csv_file_path, dtype={'timestamp': str})
    # Map timestamp to transformation array [northing, easting, altitude, qx, qy, qz, qw]
    transformations = {row['timestamp']: row[['northing', 'easting', 'altitude', 'qx', 'qy', 'qz', 'qw']].values.astype(float) 
                      for _, row in df_gt.iterrows()}
    return transformations

def apply_transformation(pc, transformation):
    # Scale translation (northing, easting, altitude) by 0.01
    trans = transformation[:3] * 0.01 # 0.01 is from 1/crop_size
    # Get rotation matrix from quaternion (qw, qx, qy, qz)
    R = o3d.geometry.get_rotation_matrix_from_quaternion([transformation[6], transformation[3], transformation[4], transformation[5]])

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[0:3, 3] = trans

    # Apply transformation to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform(T)
    return np.asarray(pcd.points)

def load_pc(file_path):
    # Load .bin file as point cloud (x, y, z)
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pc

def save_pc(pc, output_path):
    # Save point cloud to .bin file
    pc.astype(np.float32).tofile(output_path)

def process_folder(folder_path, output_folder, transform_csv_path):
    # Load transformations from trajectory_transform.csv
    if not os.path.exists(transform_csv_path):
        print(f"Transformation file not found: {transform_csv_path}")
        return False

    transformations = load_transformations(transform_csv_path)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .bin file
    for file in os.listdir(folder_path):
        if file.endswith('.bin'):
            file_path = os.path.join(folder_path, file)
            timestamp = file.split('.')[0]
            if timestamp in transformations:
                pc = load_pc(file_path)
                transformed_pc = apply_transformation(pc, transformations[timestamp])
                save_pc(transformed_pc, os.path.join(output_folder, file))
                print(f"Transformed {file} for {os.path.basename(os.path.dirname(folder_path))}")
            else:
                print(f"No transformation found for timestamp {timestamp} in {os.path.basename(os.path.dirname(folder_path))}")

    return True


# Base directories and configurations
local_base_dir = '../../data/training/'
datasets = ['DCC04', 'DCC05']
sensors = ['Aeva', 'Ouster', 'Avia', 'Velodyne']

# Loop over datasets and sensors
for dataset in datasets:
    for sensor in sensors:
        # Define paths
        folder_path = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'LiDAR')
        output_folder = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'LiDAR_transformed')
        transform_csv_path = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'trajectory_transform.csv')

        # Check if input folder exists
        if not os.path.exists(folder_path):
            print(f"Input folder not found: {folder_path}")
            continue

        # Process the folder
        process_folder(folder_path, output_folder, transform_csv_path)
