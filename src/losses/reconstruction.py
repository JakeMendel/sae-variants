"""
Loss functions for multi-cluster subspace discovery.

All losses are designed to be differentiable for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class LossFunction(ABC, nn.Module):
    """Base class for loss functions."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss.

        Args:
            x: (batch_size, d) input embeddings
            model: SubspaceModel

        Returns:
            loss: scalar loss value
            metrics: dict of named loss components for logging
        """
        pass


class ReconstructionLoss(LossFunction):
    """
    Full reconstruction loss in original space.

    Loss = ||x - reconstruct(x)||^2
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, _, recon_full = model.get_reconstructions(x)

        sq_errors = (x - recon_full) ** 2

        if self.reduction == "mean":
            loss = sq_errors.mean()
        elif self.reduction == "sum":
            loss = sq_errors.sum()
        elif self.reduction == "none":
            loss = sq_errors
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        metrics = {
            "reconstruction_loss": loss.detach(),
            "per_dim_mse": sq_errors.mean(dim=0).detach(),
        }

        return loss, metrics


class SubspaceReconstructionLoss(LossFunction):
    """
    Reconstruction loss computed separately in each subspace.

    This allows monitoring how well each subspace is being learned.
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight_a: float = 1.0,
        weight_b: float = 1.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_a, x_b, assignments_a, assignments_b = model.forward(x)
        recon_a, recon_b, _ = model.get_reconstructions(x)

        # Reconstruction error in each subspace
        loss_a = ((x_a - recon_a) ** 2).mean()
        loss_b = ((x_b - recon_b) ** 2).mean()

        loss = self.weight_a * loss_a + self.weight_b * loss_b

        metrics = {
            "loss_subspace_a": loss_a.detach(),
            "loss_subspace_b": loss_b.detach(),
            "total_loss": loss.detach(),
        }

        return loss, metrics


class ClusterCompactnessLoss(LossFunction):
    """
    Loss based on within-cluster variance (compactness).

    Encourages tight clusters by minimizing the expected distance
    from points to their assigned centroids.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_a, x_b, assignments_a, assignments_b = model.forward(x)

        centroids_a = model.clusterer_a.get_centroids()
        centroids_b = model.clusterer_b.get_centroids()

        # Compute weighted average distance to centroids
        # For subspace A
        dists_a = ((x_a.unsqueeze(1) - centroids_a.unsqueeze(0)) ** 2).sum(dim=2)
        weighted_dist_a = (assignments_a * dists_a).sum(dim=1).mean()

        # For subspace B
        dists_b = ((x_b.unsqueeze(1) - centroids_b.unsqueeze(0)) ** 2).sum(dim=2)
        weighted_dist_b = (assignments_b * dists_b).sum(dim=1).mean()

        loss = weighted_dist_a + weighted_dist_b

        metrics = {
            "compactness_a": weighted_dist_a.detach(),
            "compactness_b": weighted_dist_b.detach(),
            "compactness_total": loss.detach(),
        }

        return loss, metrics


class EntropyRegularization(LossFunction):
    """
    Entropy regularization for cluster assignments.

    Can encourage either:
    - Low entropy (confident assignments) with positive weight
    - High entropy (soft assignments) with negative weight
    """

    def __init__(self, weight: float = 0.1, target: str = "low"):
        super().__init__()
        self.weight = weight
        self.target = target

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, _, assignments_a, assignments_b = model.forward(x)

        # Compute entropy of assignments
        entropy_a = -(assignments_a * torch.log(assignments_a + 1e-10)).sum(dim=1).mean()
        entropy_b = -(assignments_b * torch.log(assignments_b + 1e-10)).sum(dim=1).mean()

        entropy = entropy_a + entropy_b

        if self.target == "low":
            loss = self.weight * entropy
        else:
            loss = -self.weight * entropy

        metrics = {
            "entropy_a": entropy_a.detach(),
            "entropy_b": entropy_b.detach(),
            "entropy_total": entropy.detach(),
        }

        return loss, metrics


class CombinedLoss(LossFunction):
    """Combine multiple loss functions with weights."""

    def __init__(self, losses: Dict[str, Tuple[LossFunction, float]]):
        """
        Args:
            losses: Dict mapping name -> (loss_function, weight)
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = 0.0
        all_metrics = {}

        for name, loss_fn in self.losses.items():
            loss, metrics = loss_fn(x, model)
            weighted_loss = self.weights[name] * loss
            total_loss = total_loss + weighted_loss

            # Prefix metrics with loss name
            for metric_name, value in metrics.items():
                all_metrics[f"{name}/{metric_name}"] = value

        all_metrics["total_loss"] = total_loss.detach() if torch.is_tensor(total_loss) else total_loss

        return total_loss, all_metrics


class IndependenceLoss(LossFunction):
    """
    Encourages cluster assignments in subspace A to be independent of assignments in subspace B.

    This is what makes the TRUE subspace special: in the correct decomposition,
    knowing a point's A-cluster tells you nothing about its B-cluster.

    Uses mutual information approximation via correlation of soft assignments.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, _, assignments_a, assignments_b = model.forward(x)

        # Compute correlation between assignment distributions
        # If independent, correlation should be zero

        # Center the assignments
        a_centered = assignments_a - assignments_a.mean(dim=0, keepdim=True)
        b_centered = assignments_b - assignments_b.mean(dim=0, keepdim=True)

        # Cross-correlation matrix between A and B assignments
        # Shape: (n_clusters_a, n_clusters_b)
        cross_corr = (a_centered.T @ b_centered) / (x.shape[0] - 1)

        # Penalize correlation (want it to be zero for independence)
        # Using Frobenius norm of cross-correlation
        independence_violation = (cross_corr ** 2).sum()

        loss = self.weight * independence_violation

        metrics = {
            "independence_violation": independence_violation.detach(),
            "cross_corr_max": cross_corr.abs().max().detach(),
        }

        return loss, metrics


class SeparationLoss(LossFunction):
    """
    Encourages cluster centroids to be well-separated within each subspace.

    Maximizes minimum distance between centroids (or penalizes small distances).
    """

    def __init__(self, margin: float = 0.5, weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        centroids_a = model.clusterer_a.get_centroids()
        centroids_b = model.clusterer_b.get_centroids()

        def pairwise_separation_loss(centroids):
            # Compute pairwise distances
            n = centroids.shape[0]
            diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)  # (n, n, d)
            dists = (diff ** 2).sum(dim=2).sqrt()  # (n, n)

            # Mask diagonal
            mask = 1 - torch.eye(n, device=centroids.device)
            dists = dists * mask

            # Penalize distances below margin (hinge loss style)
            violations = F.relu(self.margin - dists) * mask
            return violations.sum() / (n * (n - 1))

        loss_a = pairwise_separation_loss(centroids_a)
        loss_b = pairwise_separation_loss(centroids_b)

        loss = self.weight * (loss_a + loss_b)

        metrics = {
            "separation_loss_a": loss_a.detach(),
            "separation_loss_b": loss_b.detach(),
        }

        return loss, metrics


def get_loss_function(
    name: str,
    **kwargs,
) -> LossFunction:
    """Factory function to create loss functions."""
    losses = {
        "reconstruction": ReconstructionLoss,
        "subspace_reconstruction": SubspaceReconstructionLoss,
        "compactness": ClusterCompactnessLoss,
        "entropy": EntropyRegularization,
        "independence": IndependenceLoss,
        "separation": SeparationLoss,
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")

    return losses[name](**kwargs)
