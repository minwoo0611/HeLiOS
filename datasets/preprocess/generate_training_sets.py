import numpy as np
import os
import pandas as pd
import pickle


def construct_query_dict(df_centroids, filename, overlap_matrix_file_path, threshold):
    queries = {}
    with open(overlap_matrix_file_path, 'rb') as file:
        for i, line in enumerate(file):
            if i >= len(df_centroids):
                break
            overlaps = np.fromstring(line, dtype=float, sep=' ')
            query = df_centroids.iloc[i]["file"]
            

            pos_overlap_list = []
            pos = np.where(overlaps > threshold)[0].tolist()
            pos = np.sort(pos)
            if i in pos:
                pos = np.delete(pos, np.where(pos == i))

            pos = pos[pos < len(df_centroids)]
            
            pos_overlap_list.append(overlaps[pos])

            semi_pos = np.where((overlaps > 0) & (overlaps <= threshold))[0].tolist()
            semi_pos = np.sort(semi_pos)
            semi_pos = semi_pos[semi_pos < len(df_centroids)]

            semi_pos_overlap_list = []
            semi_pos_overlap_list.append(overlaps[semi_pos])


            position = np.array([df_centroids.iloc[i]['northing'], df_centroids.iloc[i]['easting']])
            file_name = os.path.split(query)[1]

            queries[i] = {"id": i, "timestamp": int(file_name.split(".")[0]), "rel_scan_filepath": query, "positives": pos, "non_negatives": semi_pos, "position": position, "pos_overlap": pos_overlap_list, "semi_pos_overlap": semi_pos_overlap_list}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

base_path = "../../data/"
runs_folder = "training/"
filename = "trajectory.csv"
pointcloud_fols = "/LiDAR/"
overlap_matrix_file_path = base_path + "overlap_matrix_training.txt"
save_path = base_path + "training.pickle"
overlap_threshold = 0.5

all_folders = sorted(os.listdir(
    os.path.join(base_path, runs_folder)))

folders = []

# All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(
        base_path, runs_folder, folder, filename), sep=',')

    df_locations['timestamp'] = runs_folder+folder + \
        pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        row_df = pd.DataFrame([row])
        df_train = pd.concat([df_train, row_df], ignore_index=True)


print("Number of training submaps: "+str(len(df_train['file'])))
construct_query_dict(df_train, save_path, overlap_matrix_file_path, 0.5)


runs_folder = "validation/"
overlap_matrix_file_path = base_path + "overlap_matrix_Roundabout.txt"
save_path = base_path + "validation.pickle"
validation_set = ["Roundabout01"] # it can be sequence or sequence-sensor like (Roundabout01-Ouster). It can include multiple sequence-sensor

all_folders = sorted(os.listdir(
    os.path.join(base_path, runs_folder)))

folders = []

# validation set
for folder in all_folders:
    for validation in validation_set:
        if validation in folder:
            folders.append(folder)

df_val = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(
        base_path, runs_folder, folder, filename), sep=',')

    df_locations['timestamp'] = runs_folder+folder + \
        pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        row_df = pd.DataFrame([row])
        df_val = pd.concat([df_val, row_df], ignore_index=True)

print("Number of validation submaps: "+str(len(df_val['file'])))
construct_query_dict(df_val, save_path, overlap_matrix_file_path, overlap_threshold)





