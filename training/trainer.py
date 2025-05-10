import os
import pathlib
import sys
import numpy as np
import torch
import torch.nn as nn
import tqdm

from datasets.dataset_utils import make_dataloaders
from eval.evaluate import evaluate, print_eval_stats
from misc.utils import TrainingParams
from models.losses.loss import make_losses
from models.model_factory import model_factory

def aggregate_epoch_stats(running_stats: list, phase: str, stats: dict) -> None:
    """
    Aggregate statistics over multiple iterations for an epoch and print them.
    
    Args:
        running_stats: List of tuples, each containing (stats_tr, stats_ps, stats_sn)
        phase: Either "train" or "val"
        stats: Dictionary to store aggregated statistics
    """
    # Initialize epoch_stats with sub-dictionaries for each loss function
    epoch_stats = {
        "global": {
            "truncated": {},
            "triplet_ps": {},
            "triplet_sn": {}
        }
    }
    
    # Define loss function names corresponding to the tuple indices
    loss_names = ["truncated", "triplet_ps", "triplet_sn"]
    
    # Aggregate statistics for each loss function
    for idx, loss_name in enumerate(loss_names):
        for key in running_stats[0][idx]:  # Iterate over keys in stats for this loss
            values = [e[idx][key] for e in running_stats]  # Collect values across iterations
            if isinstance(values[0], dict):
                # Handle nested dictionaries (e.g., recall)
                epoch_stats["global"][loss_name][key] = {
                    k: np.mean([e[k] for e in values]) for k in values[0]
                }
            elif isinstance(values[0], np.ndarray):
                # Handle arrays
                epoch_stats["global"][loss_name][key] = np.mean(np.stack(values), axis=0)
            else:
                # Handle scalars
                epoch_stats["global"][loss_name][key] = np.mean(values)
    
    # Store individual losses in global for printing
    epoch_stats["global"]["loss_truncated"] = epoch_stats["global"]["truncated"]["loss"]
    epoch_stats["global"]["loss_triplet_ps"] = epoch_stats["global"]["triplet_ps"]["loss"]
    epoch_stats["global"]["loss_triplet_sn"] = epoch_stats["global"]["triplet_sn"]["loss"]
    
    # Copy truncated stats to global for backward compatibility with print_stats
    epoch_stats["global"].update(epoch_stats["global"]["truncated"])
    
    # Store and print stats
    stats[phase].append(epoch_stats)
    print_stats(phase, epoch_stats)

def print_global_stats(phase: str, stats: dict) -> None:
    """
    Print formatted training or validation statistics for a phase, including individual losses.
    
    Args:
        phase: Either "train" or "val"
        stats: Dictionary containing aggregated statistics
    """
    output = (
        f"{phase}  loss_truncated: {stats['loss_truncated']:.4f}  "
        f"loss_triplet_ps: {stats['loss_triplet_ps']:.4f}  "
        f"loss_triplet_sn: {stats['loss_triplet_sn']:.4f}  "
        f"embedding norm: {stats['avg_embedding_norm']:.3f}  "
    )
    if "num_triplets" in stats:
        output += (
            f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  "
            f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
        )
    if "positives_per_query" in stats:
        output += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if "best_positive_ranking" in stats:
        output += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if "recall" in stats:
        output += f"Recall@1: {stats['recall'][1]:.4f}   "
    if "ap" in stats:
        output += f"AP: {stats['ap']:.4f}   "
    print(output)


def print_stats(phase: str, stats: dict) -> None:
    """Print global statistics for the given phase."""
    print_global_stats(phase, stats["global"])


def tensors_to_numbers(stats: dict) -> dict:
    """Convert tensor values in stats to numbers."""
    return {
        key: value.item() if torch.is_tensor(value) else value
        for key, value in stats.items()
    }


def compute_embeddings(model: nn.Module, batch: list, device: str) -> torch.Tensor:
    """Compute embeddings for a batch without gradients."""
    embeddings_list = []
    with torch.no_grad():
        for minibatch in batch:
            minibatch = {k: v.to(device) for k, v in minibatch.items()}
            outputs = model(minibatch)
            embeddings = outputs["global"]
            embeddings_list.append(embeddings)
    return torch.cat(embeddings_list, dim=0).to(device)


def compute_losses(
    embeddings: torch.Tensor,
    positives_mask: torch.Tensor,
    negatives_mask: torch.Tensor,
    semi_positive_mask: torch.Tensor,
    overlap: torch.Tensor,
    loss_fn: list,
    loss_weights: list,
    device: str,
) -> tuple:
    """Compute the combined loss and statistics from multiple loss functions."""
    loss_truncated, loss_guided_triplet_ps, loss_guided_triplet_sn = loss_fn
    loss_tr, stats_tr = loss_truncated(embeddings, positives_mask, negatives_mask, overlap, device)
    loss_ps, stats_ps = loss_guided_triplet_ps(embeddings, positives_mask, semi_positive_mask, overlap, "ps")
    loss_sn, stats_sn = loss_guided_triplet_sn(embeddings, semi_positive_mask, negatives_mask, overlap, "sn")
    
    # Combine losses with weights
    loss_tr = loss_weights[0] * loss_tr
    loss_ps = loss_weights[1] * loss_ps
    loss_sn = loss_weights[2] * loss_sn
    total_loss = loss_tr + loss_ps + loss_sn

    stats_tr = tensors_to_numbers(stats_tr)  # Use stats_tr as base, assuming it contains all required keys
    stats_ps = tensors_to_numbers(stats_ps)
    stats_sn = tensors_to_numbers(stats_sn)

    stats = (stats_tr, stats_ps, stats_sn)
    
    return total_loss, stats


def update_model(
    model: nn.Module,
    batch: list,
    embeddings_grad: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    """Update model parameters using computed gradients."""
    optimizer.zero_grad()
    grad_idx = 0
    with torch.set_grad_enabled(True):
        for minibatch in batch:
            minibatch = {k: v.to(device) for k, v in minibatch.items()}
            outputs = model(minibatch)
            embeddings = outputs["global"]
            minibatch_size = len(embeddings)
            embeddings.backward(
                gradient=embeddings_grad[grad_idx : grad_idx + minibatch_size]
            )
            grad_idx += minibatch_size
        optimizer.step()


def multistaged_training_step(
    global_iter,
    model: nn.Module,
    phase: str,
    device: str,
    optimizer: torch.optim.Optimizer,
    loss_fn: list,
    loss_weights: list,
) -> dict:
    """Perform a single training or validation step with multi-stage processing."""
    assert phase in ["train", "val"]
    batch, positives_mask, negatives_mask, semi_positive_mask, overlap = next(global_iter)
    
    # Set model mode
    model.train() if phase == "train" else model.eval()

    # Stage 1: Compute embeddings without gradients
    embeddings = compute_embeddings(model, batch, device)
    torch.cuda.empty_cache()

    # Stage 2: Compute losses and gradients
    stats = {}
    with torch.set_grad_enabled(phase == "train"):
        if phase == "train":
            embeddings.requires_grad_(True)
        
        total_loss, stats = compute_losses(
            embeddings, positives_mask, negatives_mask, semi_positive_mask, overlap, loss_fn, loss_weights, device
        )
        print(total_loss)
        if phase == "train":
            total_loss.backward()
            embeddings_grad = embeddings.grad
        else:
            embeddings_grad = None

    # Stage 3: Update model parameters if training
    if phase == "train":
        update_model(model, batch, embeddings_grad, optimizer, device)
    torch.cuda.empty_cache()

    return stats


def create_weights_folder() -> str:
    """Create a folder to save model weights."""
    script_dir = pathlib.Path(__file__).parent.absolute()
    parent_dir = os.path.dirname(script_dir)
    weights_path = os.path.join(parent_dir, "weights")
    os.makedirs(weights_path, exist_ok=True)
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Cannot create weights folder: {weights_path}")
    return weights_path


def initialize_model_and_optimizer(params: TrainingParams, device: str) -> tuple:
    """Initialize the model, optimizer, and scheduler."""
    model = model_factory(params.model_params)
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, params.scheduler_milestones, gamma=0.1
    )
    return model, optimizer, scheduler


def log_model_info(model: nn.Module, model_name: str) -> None:
    """Log model details and parameter count."""
    print(f"Model name: {model_name}")
    if hasattr(model, "print_info"):
        model.print_info()
    else:
        n_params = sum(param.nelement() for param in model.parameters())
        print(f"Number of model parameters: {n_params}")
    print(f"Model device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")


def do_train(params: TrainingParams) -> None:
    """Train the model using the specified parameters."""
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up model, optimizer, and scheduler
    model, optimizer, scheduler = initialize_model_and_optimizer(params, device)
    log_model_info(model, params.model_name)

    # Set up dataloaders and loss functions
    dataloaders = make_dataloaders(params)
    loss_fn = make_losses(params)
    loss_weights = [params.truncated_weight, params.guided_p_sp_weight, params.guided_sp_n_weight]

    # Create weights folder
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, params.model_name)

    # Initialize training statistics and phases
    stats = {"train": [], "val": []} if "val" in dataloaders else {"train": []}
    phases = ["train", "val"] if "val" in dataloaders else ["train"]
    maximum_ap = 0

    # Training loop
    for epoch in tqdm.tqdm(range(1, params.epochs + 1), desc="Training"):
        for phase in phases:
            running_stats = []
            count_batches = 0
            global_iter = iter(dataloaders[phase])

            while True:
                count_batches += 1
                if params.debug and count_batches > 2:
                    break
                try:
                    # Store the tuple of stats directly
                    batch_stats = multistaged_training_step(
                        global_iter, model, phase, device, optimizer, loss_fn, loss_weights
                    )
                    running_stats.append(batch_stats)
                except StopIteration:
                    break

            # Aggregate statistics using the new function
            aggregate_epoch_stats(running_stats, phase, stats)

            # Save best model based on validation AP
            if phase == "val" and stats[phase][-1]["global"]["ap"] > maximum_ap:
                maximum_ap = stats[phase][-1]["global"]["ap"]
                torch.save(model.state_dict(), f"{model_pathname}_best.pth")

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Save model weights periodically
        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), f"{model_pathname}_{epoch}.pth")

    # Save final model weights
    final_model_path = f"{model_pathname}_final.pth"
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    # Evaluate final model
    eval_stats = evaluate(model, device, params, log=False)
    print_eval_stats(eval_stats)




if __name__ == "__main__":
    # Add the launch directory to sys.path
    launch_dir = os.getcwd()
    if launch_dir not in sys.path:
        sys.path.append(launch_dir)
    print(f"Launch directory added to sys.path: {launch_dir}")
    
    # Example usage: params should be defined before calling do_train
    # params = TrainingParams(...)
    # do_train(params)