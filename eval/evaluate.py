from typing import List, Dict, Tuple, Optional
import os
import argparse
import pickle
import numpy as np
import torch
import MinkowskiEngine as ME
import tqdm
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import sys

launch_dir = os.getcwd()
if launch_dir not in sys.path:
    sys.path.append(launch_dir)
    sys.path.append(launch_dir + "/docker/thirdparty/SparseTransformer")

from models.model_factory import model_factory
from misc.utils import TrainingParams
from datasets.pnv_raw import PNVPointCloudLoader


DATASET_FOLDER = "data/"

def evaluate(
    model: torch.nn.Module,
    device: str,
    params: TrainingParams,
    log: bool = False,
    show_progress: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on all specified evaluation datasets.

    Args:
        model: The neural network model to evaluate.
        device: Device to run the model on ('cuda' or 'cpu').
        params: Training parameters containing file paths and settings.
        log: Whether to log detailed results.
        show_progress: Whether to display progress bars.

    Returns:
        Dictionary mapping location names to evaluation statistics.
    """
    eval_database_files = [params.eval_db_file]
    eval_query_files = [params.eval_query_file]
    assert len(eval_database_files) == len(eval_query_files), "Mismatch in database and query file counts"

    stats = {}
    for db_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract and validate location name
        location_name = db_file.split('_')[0]
        assert location_name == query_file.split('_')[0], \
            f"Database location {db_file} does not match query location {query_file}"

        # Load database and query sets
        db_path = os.path.join(DATASET_FOLDER, db_file)
        with open(db_path, 'rb') as f:
            database_sets = pickle.load(f)

        query_path = os.path.join(DATASET_FOLDER, query_file)
        with open(query_path, 'rb') as f:
            query_sets = pickle.load(f)

        # Evaluate dataset and store stats
        dataset_stats = evaluate_dataset(
            model, device, params, database_sets, query_sets, log=log, show_progress=show_progress
        )
        stats[location_name] = dataset_stats

    return stats


def evaluate_dataset(
    model: torch.nn.Module,
    device: str,
    params: TrainingParams,
    database_sets: List[Dict],
    query_sets: List[Dict],
    log: bool = False,
    show_progress: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model on a single dataset pair (database and query sets).

    Args:
        model: The neural network model.
        device: Device to run the model on.
        params: Training parameters.
        database_sets: List of database point cloud metadata.
        query_sets: List of query point cloud metadata.
        log: Whether to log detailed results.
        show_progress: Whether to display progress bars.

    Returns:
        Dictionary of average evaluation metrics (recall, one_percent_recall, f1_score, auc_score).
    """
    model.eval()
    recall, one_percent_recall, f1_score, auc_score = [], [], [], []
    database_embeddings, query_embeddings = [], []
    intra_session_pairs = []

    # Identify intra-session pairs
    for i, db_set in enumerate(database_sets):
        db_idx = db_set[0]['query'].find("LiDAR")
        for j, query_set in enumerate(query_sets):
            query_idx = query_set[0]['query'].find("LiDAR")
            if query_set[0]['query'][:query_idx] == db_set[0]['query'][:db_idx]:
                intra_session_pairs.append((i, j))
                if log:
                    print(f"Intra-session pair: ({i}, {j})")

    # Compute embeddings
    for db_set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        embedding = get_latent_vectors(model, db_set, device, params)
        database_embeddings.append(embedding)

    for query_set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        embedding = get_latent_vectors(model, query_set, device, params)
        query_embeddings.append(embedding)
        if len(query_embeddings) > 4:
            break

    # Perform place recognition for all pairs
    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            pair_recall, pair_opr, f1, auc = perform_place_recognition(
                i, j, database_embeddings, query_embeddings, query_sets, database_sets, intra_session_pairs
            )
            recall.append(pair_recall)
            one_percent_recall.append(pair_opr)
            f1_score.append(f1)
            auc_score.append(auc)

    # Compute average metrics
    return {
        'ave_recall': np.mean(recall),
        'ave_one_percent_recall': np.mean(one_percent_recall),
        'ave_f1_score': np.mean(f1_score),
        'ave_auc_score': np.mean(auc_score)
    }


def get_latent_vectors(
    model: torch.nn.Module,
    dataset: List[Dict],
    device: str,
    params: TrainingParams
) -> np.ndarray:
    """
    Compute latent vectors (embeddings) for a dataset.

    Args:
        model: The neural network model.
        dataset: List of point cloud metadata.
        device: Device to run the model on.
        params: Training parameters.

    Returns:
        Array of embeddings for the dataset.
    """
    if params.debug:
        return np.random.rand(len(dataset), 256)

    pc_loader = PNVPointCloudLoader()
    model.eval()
    embeddings = None

    for i, elem in enumerate(dataset):
        pc_path = os.path.join(DATASET_FOLDER, dataset[elem]["query"])
        point_cloud = pc_loader(pc_path)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        embedding = compute_embedding(model, point_cloud, device, params)

        if embeddings is None:
            embeddings = np.zeros((len(dataset), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings


def compute_embedding(
    model: torch.nn.Module,
    point_cloud: torch.Tensor,
    device: str,
    params: TrainingParams
) -> np.ndarray:
    """
    Compute a single embedding for a point cloud.

    Args:
        model: The neural network model.
        point_cloud: Input point cloud tensor.
        device: Device to run the model on.
        params: Training parameters.

    Returns:
        Embedding vector as a numpy array.
    """
    coords, _ = params.model_params.quantizer(point_cloud)
    with torch.no_grad():
        bcoords = ME.utils.batched_coordinates([coords])
        feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
        output = model(batch)
        embedding = output['global'].detach().cpu().numpy()

    return embedding


def process_query(
    query_idx: int,
    query_vector: np.ndarray,
    query_meta: Dict,
    kd_tree: KDTree,
    num_neighbors: int,
    threshold: int,
    database_output: np.ndarray,
    thresholds: np.ndarray,
    db_set_idx: int,
    database_meta: Optional[List[Dict]] = None,
    is_intra: bool = False,
    init_time: Optional[float] = None
) -> Optional[Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Process a single query for place recognition.

    Args:
        query_idx: Index of the query.
        query_vector: Embedding vector for the query.
        query_meta: Metadata for the query.
        kd_tree: KDTree built from database embeddings.
        num_neighbors: Number of nearest neighbors to retrieve.
        threshold: Number of neighbors for one-percent recall metric.
        database_output: Database embeddings.
        thresholds: Array of distance thresholds for evaluation.
        db_set_idx: Database set index.
        database_meta: Metadata for database entries (required for intra-session).
        is_intra: Whether this is an intra-session query.
        init_time: Initial timestamp for the session (required for intra-session).

    Returns:
        Tuple of (recall_indicator, one_percent_flag, tp, fp, tn, fn, count) or None if invalid.
    """
    true_neighbors = query_meta.get(db_set_idx, [])
    if not true_neighbors:
        return None

    if is_intra:
        if database_meta is None or init_time is None:
            raise ValueError("database_meta and init_time required for intra-session")

        query_timestamp = query_meta['query'].split('/')[-1].split('.')[0]
        query_timestamp = float(query_timestamp) / 1e9

        if query_timestamp - init_time < 90:
            return None

        filtered_neighbors = [
            idx for idx in true_neighbors
            if float(database_meta[idx]['query'].split('/')[-1].split('.')[0]) / 1e9 < query_timestamp - 30
        ]
    else:
        filtered_neighbors = true_neighbors

    if not filtered_neighbors:
        return None

    # Query nearest neighbors
    distances, indices = kd_tree.query([query_vector], k=200)
    indices, distances = indices[0], distances[0]

    # Filter indices for intra-session
    if is_intra:
        filtered_indices = [
            idx for j, idx in enumerate(indices)
            if float(database_meta[idx]['query'].split('/')[-1].split('.')[0]) / 1e9 < query_timestamp - 30
        ]
        filtered_distances = [
            distances[j] for j, idx in enumerate(indices)
            if float(database_meta[idx]['query'].split('/')[-1].split('.')[0]) / 1e9 < query_timestamp - 30
        ]
    else:
        filtered_indices, filtered_distances = indices, distances

    # Compute recall indicator
    recall_indicator = np.zeros(num_neighbors)
    for j, idx in enumerate(filtered_indices):
        if j >= num_neighbors:
            break
        if idx in filtered_neighbors:
            recall_indicator[j] = 1
            break

    # Compute one-percent recall
    valid_indices = filtered_indices[:threshold]
    one_percent_flag = 1 if set(valid_indices).intersection(filtered_neighbors) else 0

    # Compute threshold-based metrics
    first_idx, desc_dist_0 = filtered_indices[0], filtered_distances[0]
    tp, fp, tn, fn = [np.zeros(len(thresholds)) for _ in range(4)]

    for thres_idx, th_val in enumerate(thresholds):
        if desc_dist_0 < th_val:
            if first_idx in filtered_neighbors:
                tp[thres_idx] = 1
            else:
                fp[thres_idx] = 1
        else:
            if first_idx in filtered_neighbors:
                fn[thres_idx] = 1
            else:
                tn[thres_idx] = 1

    return recall_indicator, one_percent_flag, tp, fp, tn, fn, 1


def perform_place_recognition(
    db_idx: int,
    query_idx: int,
    database_vectors: List[np.ndarray],
    query_vectors: List[np.ndarray],
    query_sets: List[Dict],
    database_sets: List[Dict],
    intra_session_pairs: List[Tuple[int, int]]
) -> Tuple[float, float, float, float]:
    """
    Perform place recognition for a database-query pair.

    Args:
        db_idx: Database set index.
        query_idx: Query set index.
        database_vectors: List of database embeddings.
        query_vectors: List of query embeddings.
        query_sets: List of query metadata.
        database_sets: List of database metadata.
        intra_session_pairs: List of intra-session pair indices.

    Returns:
        Tuple of (recall@1, one_percent_recall, max_f1_score, auc_score).
    """
    num_jobs = 20
    database_output = database_vectors[db_idx]
    queries_output = query_vectors[query_idx]
    database_meta = list(database_sets[db_idx].values())
    kd_tree = KDTree(database_output)

    num_neighbors = 30
    threshold = max(int(round(len(database_output) / 1000.0)), 1)
    thresholds = np.linspace(0, 1, 1000)
    is_intra = (db_idx, query_idx) in intra_session_pairs

    init_time = float(database_meta[0]['query'].split('/')[-1].split('.')[0]) / 1e9 if is_intra else None

    # Parallel processing of queries
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_query)(
            i, queries_output[i], query_sets[query_idx][i], kd_tree, num_neighbors,
            threshold, database_output, thresholds, db_idx,
            database_meta=database_meta, is_intra=is_intra, init_time=init_time
        ) for i in range(len(queries_output))
    )

    # Aggregate results
    recall_sum = np.zeros(num_neighbors)
    one_percent_total = 0
    num_thresholds = len(thresholds)
    tp, fp, tn, fn = [np.zeros(num_thresholds) for _ in range(4)]
    num_evaluated = 0

    for res in results:
        if res is None:
            continue
        rec_ind, one_flag, t_pos, f_pos, t_neg, f_neg, count = res
        recall_sum += rec_ind
        one_percent_total += one_flag
        tp += t_pos
        fp += f_pos
        tn += t_neg
        fn += f_neg
        num_evaluated += count

    # Compute metrics
    recall_cumulative = (np.cumsum(recall_sum) / num_evaluated) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)
    one_percent_recall = (one_percent_total / num_evaluated) * 100 if num_evaluated > 0 else 0.0

    precisions = np.divide(
        tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0
    )
    recalls_arr = np.divide(
        tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls_arr) / (precisions + recalls_arr)
        f1_scores[np.isnan(f1_scores)] = 0.0
    max_f1 = np.max(f1_scores)

    sort_indices = np.argsort(recalls_arr)
    sorted_recalls = recalls_arr[sort_indices]
    sorted_precisions = precisions[sort_indices]
    auc_score = np.trapz(sorted_precisions, sorted_recalls)

    final_recall = recall_cumulative[0] if recall_cumulative.size > 0 else 0
    print(f"{db_idx} - {query_idx}:")
    print(f"Recall@1: {final_recall:.4f}%")
    print(f"F1 Score: {max_f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    return final_recall, one_percent_recall, max_f1, auc_score


def print_eval_stats(stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print evaluation statistics for all locations.

    Args:
        stats: Dictionary of evaluation metrics per location.
    """
    for location, metrics in stats.items():
        print(f"Location: {location}")
        print(f"Average recall: {metrics['ave_recall']:.4f}")
        print(f"Average one percent recall: {metrics['ave_one_percent_recall']:.4f}")
        print(f"Average F1 score: {metrics['ave_f1_score']:.4f}")
        print(f"Average AUC score: {metrics['ave_auc_score']:.4f}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model-specific configuration file')
    parser.add_argument('--weights', type=str, help='Trained model weights')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--log', action='store_true', help='Log search results')
    args = parser.parse_args()

    # Print configuration
    print(f"Config path: {args.config}")
    print(f"Model config path: {args.model_config}")
    print(f"Weights: {args.weights or 'RANDOM WEIGHTS'}")
    print(f"Debug mode: {args.debug}")
    print(f"Log search results: {args.log}\n")

    # Initialize parameters and device
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = model_factory(params.model_params)
    if args.weights:
        assert os.path.exists(args.weights), f"Cannot open network weights: {args.weights}"
        print(f"Loading weights: {args.weights}")
        prev_model = torch.load(args.weights, map_location=device)
        model.load_state_dict(prev_model)
    model.to(device)

    # Run evaluation and print results
    stats = evaluate(model, device, params, args.log, show_progress=True)
    print_eval_stats(stats)


if __name__ == "__main__":
    main()
