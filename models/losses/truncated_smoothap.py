from typing import Dict, Tuple
import numpy as np
import torch

from models.losses.loss_utils import sigmoid, compute_aff


class TruncatedSmoothAP:
    """
    Truncated Smooth Average Precision (AP) loss, considering a fixed number of closest positives.
    Implements a surrogate loss for Recall@k as per the paper "Recall@k Surrogate Loss with Large
    Batches and Similarity Mixup", with only the specified number of closest positives considered.

    Attributes:
        tau1: Temperature parameter for the sigmoid function.
        similarity: Similarity metric ('cosine' or other supported metric).
        positives_per_query: Number of closest positive examples to consider per query.
    """
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4):
        """
        Initialize the Truncated Smooth AP loss.

        Args:
            tau1: Temperature parameter for smoothing the sigmoid function.
            similarity: Similarity metric to use (e.g., 'cosine').
            positives_per_query: Number of closest positive examples to consider per query.
        """
        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query

    def __call__(
        self,
        embeddings: torch.Tensor,
        positives_mask: torch.Tensor,
        negatives_mask: torch.Tensor,
        overlap: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the Truncated Smooth AP loss and associated statistics.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) containing query embeddings.
            positives_mask: Boolean tensor of shape (batch_size, batch_size) indicating positive pairs.
            negatives_mask: Boolean tensor of shape (batch_size, batch_size) indicating negative pairs.
            overlap: Tensor of overlap scores (not used in loss computation but included for compatibility).
            epoch: Current training epoch (not used but included for compatibility).

        Returns:
            Tuple of (loss value, dictionary of evaluation statistics).
        """
        # Ensure tensors are on the same device
        device = embeddings.device
        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)
        overlap = overlap.to(device)

        # Compute pairwise similarity scores
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Identify the closest positives
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(~positives_mask, float('-inf'))
        closest_positives_ndx = torch.topk(
            s_positives, k=self.positives_per_query, dim=1, largest=True, sorted=True
        )[1]

        # Count number of positives per query
        n_positives = positives_mask.sum(dim=1)

        # Compute smoothed ranking differences
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Compute positive rankings
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask
        mask = torch.ones_like(pos_s_sigmoid).scatter_(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask
        r_p = pos_s_sigmoid.sum(dim=2) + 1.  # Rank of positives (batch_size, positives_per_query)

        # Compute denominator (including negatives)
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + neg_s_sigmoid.sum(dim=2)

        # Compute ranking ratio for AP
        r = r_p / r_omega

        # Mask out invalid positives (when fewer positives exist than positives_per_query)
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)

        # Filter queries with at least one positive to avoid division by zero
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]
        n_valid_positives = n_valid_positives[valid_q_mask]

        # Compute Average Precision (AP) and loss
        ap = (masked_r.sum(dim=1) / n_valid_positives).mean()
        loss = 1.0 - ap

        # Compute evaluation statistics
        stats = self._compute_stats(
            n_positives=n_positives,
            s_diff=s_diff,
            negatives_mask=negatives_mask,
            embeddings=embeddings,
            loss=loss,
            ap=ap
        )

        return loss, stats

    def _compute_stats(
        self,
        n_positives: torch.Tensor,
        s_diff: torch.Tensor,
        negatives_mask: torch.Tensor,
        embeddings: torch.Tensor,
        loss: torch.Tensor,
        ap: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation statistics for the batch.

        Args:
            n_positives: Tensor of shape (batch_size,) with number of positives per query.
            s_diff: Tensor of shape (batch_size, positives_per_query, batch_size) with similarity differences.
            negatives_mask: Boolean tensor of shape (batch_size, batch_size) for negative pairs.
            embeddings: Tensor of shape (batch_size, embedding_dim).
            loss: Computed loss value.
            ap: Computed average precision.

        Returns:
            Dictionary containing evaluation metrics.
        """
        stats = {}

        # Mean number of positives per query
        stats['positives_per_query'] = n_positives.float().mean().item()

        # Mean ranking of the best positive example
        temp = s_diff.detach() > 0
        temp = torch.logical_and(temp[:, 0], negatives_mask)  # Consider the best positive
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean().item()

        # Recall@1
        stats['recall'] = {1: (hard_ranking <= 1).float().mean().item()}

        # Loss and AP
        stats['loss'] = loss.item()
        stats['ap'] = ap.item()

        # Average embedding norm
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean().item()

        return stats