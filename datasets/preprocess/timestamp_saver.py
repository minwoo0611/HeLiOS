import pandas as pd
import os
import glob

# Base directories
local_base_dir = '../../data/training/'
global_base_dir = '../../data/RAW/'

# Define the datasets and sensors
datasets = ['DCC04', 'DCC05']
sensors = ['Aeva', 'Ouster', 'Avia', 'Velodyne']

for dataset in datasets:
    for sensor in sensors:
        # Directory containing .bin files
        bin_dir = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'LiDAR')
        trajectory_path = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'trajectory.csv')
        trajectory_transform_path = os.path.join(local_base_dir, f"{dataset}-{sensor}", 'trajectory_transform.csv')
        global_path = os.path.join(global_base_dir, dataset, 'LiDAR_GT', f'global_{sensor}_gt.txt')
        local_path = os.path.join(global_base_dir, dataset, 'LiDAR_GT', f'{sensor}_gt.txt')

        # Check if global_path and local_path exist
        if not os.path.exists(global_path):
            print(f"Global file not found: {global_path}")
            continue
        if not os.path.exists(local_path):
            print(f"Local file not found: {local_path}")
            continue

        # Ensure output directory exists
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)

        # Load timestamps from .bin files
        bin_files = glob.glob(os.path.join(bin_dir, '*.bin'))
        timestamps = []
        for bin_file in bin_files:
            # Extract timestamp from filename (e.g., '1234567890.bin' -> '1234567890')
            filename = os.path.basename(bin_file)
            timestamp = filename.replace('.bin', '')
            try:
                # Ensure timestamp is a valid integer
                int(timestamp)
                timestamps.append(timestamp)
            except ValueError:
                print(f"Invalid timestamp in filename: {filename}")
                continue

        if not timestamps:
            print(f"No valid .bin files found in {bin_dir}")
            continue

        # Load the global trajectory file (for trajectory.csv)
        columns = ['timestamp', 'northing', 'easting', 'altitude', 'qx', 'qy', 'qz', 'qw']
        global_df = pd.read_csv(global_path, sep=' ', names=columns, usecols=['timestamp', 'northing', 'easting'])

        # Load the local trajectory file (for trajectory_transform.csv)
        local_df = pd.read_csv(local_path, sep=' ', names=columns)

        # Format timestamps in both DataFrames as strings
        global_df['timestamp'] = global_df['timestamp'].astype(int).astype(str)
        local_df['timestamp'] = local_df['timestamp'].astype(int).astype(str)

        # Filter both DataFrames to only include timestamps from .bin files
        global_df = global_df[global_df['timestamp'].isin(timestamps)]
        local_df = local_df[local_df['timestamp'].isin(timestamps)]

        # Process trajectory.csv (global data, only timestamp, northing, easting)
        if global_df.empty:
            print(f"No matching timestamps found in global file for {dataset}-{sensor}")
        else:
            trajectory_df = global_df[['timestamp', 'northing', 'easting']].copy()
            # Sort by timestamp
            trajectory_df['timestamp'] = trajectory_df['timestamp'].astype(int)
            trajectory_df = trajectory_df.sort_values('timestamp')
            trajectory_df['timestamp'] = trajectory_df['timestamp'].astype(str)
            # Save trajectory.csv
            trajectory_df.to_csv(trajectory_path, index=False)
            print(f"Generated trajectory.csv with {len(trajectory_df)} entries for {dataset}-{sensor}")

        # Process trajectory_transform.csv (local data, all columns)
        if local_df.empty:
            print(f"No matching timestamps found in local file for {dataset}-{sensor}")
        else:
            trajectory_transform_df = local_df[['timestamp', 'northing', 'easting', 'altitude', 'qx', 'qy', 'qz', 'qw']].copy()
            # Sort by timestamp
            trajectory_transform_df['timestamp'] = trajectory_transform_df['timestamp'].astype(int)
            trajectory_transform_df = trajectory_transform_df.sort_values('timestamp')
            trajectory_transform_df['timestamp'] = trajectory_transform_df['timestamp'].astype(str)
            # Save trajectory_transform.csv
            trajectory_transform_df.to_csv(trajectory_transform_path, index=False)
            print(f"Generated trajectory_transform.csv with {len(trajectory_transform_df)} entries for {dataset}-{sensor}")
