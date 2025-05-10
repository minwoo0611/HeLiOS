import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import sptr

# From SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform Sinkhorn normalization in log-space for stability."""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# From SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Perform differentiable optimal transport in log-space for stability."""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    couplings = torch.cat([scores, bins], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)

    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # Multiply probabilities by M+N

    return Z

class SALAD(nn.Module):
    """
    Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): Number of input channels (d).
        num_clusters (int): Number of clusters (m).
        cluster_dim (int): Number of channels per cluster (l).
        token_dim (int): Dimension of the global scene token (g).
        dropout (float): Dropout rate.
        use_token (bool): Whether to include the global token in the output.
    """
    def __init__(
        self,
        num_channels=256,
        num_clusters=8,
        cluster_dim=32,
        token_dim=256,
        dropout=0.3,
        use_token=True
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.use_token = use_token
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # MLP for global scene token
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features
        self.cluster_features = nn.Sequential(
            nn.Conv1d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv1d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix
        self.score = nn.Sequential(
            nn.Conv1d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv1d(512, self.num_clusters, 1)
        )
        # Dustbin parameter
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        """
        Args:
            x (tuple): Feature tensor [B, C, N] and token tensor [B, C].

        Returns:
            torch.Tensor: Global descriptor [B, m*l + g].
        """
        x, t = x  # Extract features and token

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        if p.shape[2] < self.num_clusters:
            p = F.pad(p, (0, self.num_clusters - p.shape[2]))
            f = F.pad(f, (0, self.num_clusters - f.shape[2]))

        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 10)
        p = torch.exp(p)
        p = p[:, :-1, :]  # Remove dustbin row

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        if self.use_token:
            f = torch.cat([
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
            ], dim=-1)
        else:
            f = nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)

        return nn.functional.normalize(f, p=2, dim=-1)

class GeM(nn.Module):
    """Generalized Mean (GeM) pooling for sparse tensors."""
    def __init__(self, input_dim, p=3, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        """Apply GeM pooling."""
        temp = ME.SparseTensor(
            x.F.clamp(min=self.eps).pow(self.p),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        temp = self.f(temp)
        return temp.F.pow(1. / self.p)

class HeLiOS(nn.Module):
    """HeLiOS model with backbone, GeM pooling, and SALAD aggregation."""
    def __init__(
        self,
        backbone: nn.Module,
        in_dim,
        num_clusters,
        cluster_dim,
        use_token
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = GeM(input_dim=in_dim)
        self.agg = SALAD(in_dim, num_clusters, cluster_dim, in_dim, use_token)
        self.activation = nn.ReLU()

    def forward(self, batch):
        """
        Args:
            batch (dict): Contains 'features' and 'coords' for SparseTensor.

        Returns:
            dict: {'global': global_descriptor}.
        """
        # Construct SparseTensor
        x_batch = ME.SparseTensor(
            batch['features'],
            coordinates=batch['coords'],
            device=batch['coords'].device
        )
        x = self.backbone(x_batch)

        # Decompose sparse tensor
        x_coord, x_feat = x.decomposed_coordinates_and_features
        x_feat = torch.nn.utils.rnn.pad_sequence(x_feat, batch_first=True)

        # Apply pooling and aggregation
        x_gem = self.pooling(x)
        x = (x_feat.permute(0, 2, 1), x_gem)
        x = self.agg(x)

        return {'global': x}

    def print_info(self):
        """Print model information."""
        print('Model class: HeLiOS')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        print('# output channels : {}'.format(self.pooling.output_dim))