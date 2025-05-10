import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle




def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, overlap_matrix_file_path, db):
    database_trees = []
    for folder in folders:
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_locations = pd.read_csv(os.path.join(
            base_path, runs_folder, folder, filename), sep=',')
        for index, row in df_locations.iterrows():
            if folder in db:
                row_df = pd.DataFrame([row])
                df_database = pd.concat([df_database, row_df], ignore_index=True)
        if folder in db:
            database_tree = KDTree(df_database[['northing', 'easting']])
            database_trees.append(database_tree)
        
    test_sets = []
    database_sets = []
    bin_idx = 0
    for folder in folders:
        database = {}
        test = {}

        df_locations = pd.read_csv(os.path.join(
            base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder+folder + \
            pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():            
            if folder in db:
                database[len(database.keys())] = {
                    'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'idx' : bin_idx}
            test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting'], 'idx' : bin_idx}
            
            bin_idx = bin_idx + 1
        if folder in db:
            database_sets.append(database)
        test_sets.append(test)

    overlap = np.loadtxt(overlap_matrix_file_path)

    for k in range(len(database_sets)):
        tree = database_trees[k]
        for i in range(len(test_sets)):
            for key in range(len(test_sets[i].keys())):
                index = []
                
                for j in range(len(database_sets[k])):
                    if(float(overlap[test_sets[i][key]['idx'], database_sets[k][j]['idx']]) > 0.5):
                        index.append(j)

                test_sets[i][key][k] = index

    for i in range(len(database_sets)):
        for key in range(len(database_sets[i].keys())):
            del database_sets[i][key]['idx']
    for i in range(len(test_sets)):
        for key in range(len(test_sets[i].keys())):
            del test_sets[i][key]['idx']

    output_to_file(database_sets, base_path + environment + '_evaluation_db.pickle')
    output_to_file(test_sets, base_path + environment + '_evaluation_query.pickle')


base_path = "../../data/"
runs_folder = "validation/"
filename = "trajectory.csv"
pointcloud_fols = "/LiDAR/"
environment = "Roundabout"
overlap_matrix_file_path = base_path + "overlap_matrix_" + environment + ".txt"
db = ["Roundabout01-Ouster"]

folders = []
all_folders = sorted(os.listdir(
    os.path.join(base_path, runs_folder)))

for folder in all_folders:
    if environment in folder:
        folders.append(folder)

print("query: ", folders)
print("db: ", db)
construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols,
                                  filename, overlap_matrix_file_path, db)

