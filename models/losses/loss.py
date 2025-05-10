import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from misc.utils import TrainingParams
from models.losses.truncated_smoothap import TruncatedSmoothAP


def make_losses(params: TrainingParams):
    """
    Creates three loss functions for training: Truncated Smooth AP and two triplet losses.
    
    Args:
        params: TrainingParams object containing configuration parameters
    
    Returns:
        tuple: (truncated_smooth_ap_loss, triplet_loss_ps, triplet_loss_sn)
    """
    truncated_smooth_ap = TruncatedSmoothAP(
        tau1=params.tau1,
        similarity=params.similarity,
        positives_per_query=params.positives_per_query
    )
    
    triplet_loss_ps = BatchHardGuidedTripletLossWithMasks(margin=params.p_sp_distance)
    triplet_loss_sn = BatchHardGuidedTripletLossWithMasks(margin=2 * params.sp_n_distance)
    
    return truncated_smooth_ap, triplet_loss_ps, triplet_loss_sn


class GuidedTripletMarginLossWithOverlap(BaseMetricLossFunction):
    """
    Custom triplet margin loss that incorporates overlap scores and relationship types.
    """
    def __init__(self, margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def forward(self, embeddings, labels, indices_tuple, overlap_scores, relationship):
        """
        Computes the triplet loss with overlap scores.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            labels: Tensor of shape (batch_size,)
            indices_tuple: Tuple of (anchor_idx, positive_idx, negative_idx)
            overlap_scores: Tensor containing overlap scores
            relationship: String indicating relationship type ("ps" or "sn")
            
        Returns:
            Tensor: Computed loss value
        """
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple, overlap_scores, relationship)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def compute_loss(self, embeddings, labels, indices_tuple, overlap_scores, relationship):
        """
        Core loss computation logic.
        """
        # Convert indices to triplets
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        
        if len(anchor_idx) == 0:
            return self.zero_losses()

        # Compute distance matrix
        distance_matrix = self.distance(embeddings)
        overlap_scores = overlap_scores.to(embeddings.device)

        # Adjust margin based on relationship type and overlap scores
        if relationship == "ps":
            margin = self._compute_ps_margin(overlap_scores, anchor_idx, positive_idx, negative_idx)
        else:  # sn
            margin = self._compute_sn_margin(overlap_scores, anchor_idx, positive_idx)

        # Compute anchor-positive and anchor-negative distances
        ap_distances = distance_matrix[anchor_idx, positive_idx]
        an_distances = distance_matrix[anchor_idx, negative_idx]

        # Apply swap if enabled
        if self.swap:
            pn_distances = distance_matrix[positive_idx, negative_idx]
            an_distances = self.distance.smallest_dist(an_distances, pn_distances)

        # Compute triplet loss
        current_margins = self.distance.margin(ap_distances, an_distances)
        violation = current_margins + margin
        
        loss = F.softplus(violation) if self.smooth_loss else F.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet"
            }
        }

    def _compute_ps_margin(self, overlap, anchor_idx, positive_idx, negative_idx):
        """Compute margin for positive-semantic (ps) relationship."""
        exp_term = torch.exp(torch.tensor(1.0)) - 1
        pos_term = self.margin * torch.log(exp_term * overlap[anchor_idx, positive_idx] + 1)
        neg_term = self.margin * torch.log(exp_term * overlap[anchor_idx, negative_idx] + 1)
        return pos_term - neg_term

    def _compute_sn_margin(self, overlap, anchor_idx, positive_idx):
        """Compute margin for semantic-negative (sn) relationship."""
        exp_term = torch.exp(torch.tensor(1.0)) - 1
        pos_term = self.margin / 2 * torch.log(exp_term * overlap[anchor_idx, positive_idx] + 1)
        return pos_term + self.margin / 2

    def get_default_reducer(self):
        return reducers.AvgNonZeroReducer()

def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows

class HardTripletMinerWithMasks:
    """
    Mines hard triplets based on embeddings and masks.
    """
    def __init__(self, distance):
        self.distance = distance
        self.stats = {
            'max_pos_pair_dist': None,
            'max_neg_pair_dist': None,
            'mean_pos_pair_dist': None,
            'mean_neg_pair_dist': None,
            'min_pos_pair_dist': None,
            'min_neg_pair_dist': None
        }

    def __call__(self, embeddings, positives_mask, negatives_mask):
        """
        Mines hard triplets from embeddings using positive and negative masks.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            positives_mask: Boolean tensor indicating positive pairs
            negatives_mask: Boolean tensor indicating negative pairs
            
        Returns:
            tuple: (anchor_indices, positive_indices, negative_indices)
        """
        assert embeddings.dim() == 2
        with torch.no_grad():
            return self.mine(embeddings.detach(), positives_mask, negatives_mask)

    def mine(self, embeddings, positives_mask, negatives_mask):
        """
        Core triplet mining logic.
        """
        dist_matrix = self.distance(embeddings)
        
        # Get hardest positive and negative pairs
        (hardest_pos_dist, hardest_pos_indices), pos_keep = get_max_per_row(dist_matrix, positives_mask)
        (hardest_neg_dist, hardest_neg_indices), neg_keep = get_min_per_row(dist_matrix, negatives_mask)
        
        # Filter valid triplets
        valid_idx = torch.where(pos_keep & neg_keep)
        anchors = torch.arange(dist_matrix.size(0)).to(hardest_pos_indices.device)[valid_idx]
        positives = hardest_pos_indices[valid_idx]
        negatives = hardest_neg_indices[valid_idx]

        # Update statistics
        self._update_stats(hardest_pos_dist, hardest_neg_dist, valid_idx)
        
        return anchors, positives, negatives

    def _update_stats(self, pos_dist, neg_dist, valid_idx):
        """Update mining statistics."""
        pos_dist_valid = pos_dist[valid_idx]
        neg_dist_valid = neg_dist[valid_idx]
        
        self.stats.update({
            'max_pos_pair_dist': torch.max(pos_dist_valid).item(),
            'max_neg_pair_dist': torch.max(neg_dist_valid).item(),
            'mean_pos_pair_dist': torch.mean(pos_dist_valid).item(),
            'mean_neg_pair_dist': torch.mean(neg_dist_valid).item(),
            'min_pos_pair_dist': torch.min(pos_dist_valid).item(),
            'min_neg_pair_dist': torch.min(neg_dist_valid).item()
        })


class BatchHardGuidedTripletLossWithMasks:
    """
    Combines hard triplet mining with custom triplet margin loss.
    """
    def __init__(self, margin: float):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        self.loss_fn = GuidedTripletMarginLossWithOverlap(
            margin=self.margin,
            swap=False,
            distance=self.distance,
            reducer=reducers.AvgNonZeroReducer(collect_stats=True),
            collect_stats=True
        )

    def __call__(self, embeddings, positives_mask, negatives_mask, overlap_scores, relationship):
        """
        Computes batch hard triplet loss.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)
            positives_mask: Boolean tensor indicating positive pairs
            negatives_mask: Boolean tensor indicating negative pairs
            overlap_scores: Tensor containing overlap scores
            relationship: String indicating relationship type ("ps" or "sn")
            
        Returns:
            tuple: (loss_value, statistics_dict)
        """
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets, overlap_scores, relationship)
        
        stats = {
            'loss': loss.item(),
            'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
            'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
            'num_triplets': len(hard_triplets[0]),
            **{k: v for k, v in self.miner_fn.stats.items()}
        }
        
        return loss, stats