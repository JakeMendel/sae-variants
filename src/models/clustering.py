"""
Differentiable clustering algorithms.

All algorithms output soft cluster assignments that can be used for backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


class ClusteringAlgorithm(ABC, nn.Module):
    """Base class for differentiable clustering algorithms."""

    def __init__(self, dim: int, n_clusters: int):
        super().__init__()
        self.dim = dim
        self.n_clusters = n_clusters

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments.

        Args:
            x: (batch_size, dim) data points

        Returns:
            assignments: (batch_size, n_clusters) soft assignment probabilities
        """
        pass

    @abstractmethod
    def get_centroids(self) -> torch.Tensor:
        """
        Get cluster centroids.

        Returns:
            centroids: (n_clusters, dim) cluster centers
        """
        pass


class SoftKMeans(ClusteringAlgorithm):
    """
    Soft K-Means clustering with learnable centroids.

    Assignments are computed as softmax over negative squared distances,
    with a temperature parameter controlling softness.
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int,
        temperature: float = 1.0,
        learn_temperature: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__(dim, n_clusters)

        # Learnable cluster centroids
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, dim) * init_scale / (dim ** 0.5)
        )

        # Temperature (inverse softness)
        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments based on distance to centroids.

        Args:
            x: (batch_size, dim) data points

        Returns:
            assignments: (batch_size, n_clusters) soft assignment probabilities
        """
        # Compute squared distances: ||x - c||^2
        # = ||x||^2 - 2*x.c + ||c||^2
        x_sq = (x ** 2).sum(dim=1, keepdim=True)  # (batch, 1)
        c_sq = (self.centroids ** 2).sum(dim=1, keepdim=True).T  # (1, n_clusters)
        cross = x @ self.centroids.T  # (batch, n_clusters)

        sq_distances = x_sq - 2 * cross + c_sq  # (batch, n_clusters)

        # Soft assignments via softmax over negative distances
        logits = -sq_distances / self.temperature
        assignments = F.softmax(logits, dim=1)

        return assignments

    def get_centroids(self) -> torch.Tensor:
        return self.centroids


class DifferentiableGMM(ClusteringAlgorithm):
    """
    Differentiable Gaussian Mixture Model.

    Models each cluster as a Gaussian with learnable mean and (optionally) covariance.
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int,
        covariance_type: str = "spherical",  # "spherical", "diagonal", "full"
        init_scale: float = 1.0,
    ):
        super().__init__(dim, n_clusters)
        self.covariance_type = covariance_type

        # Learnable means
        self.means = nn.Parameter(
            torch.randn(n_clusters, dim) * init_scale / (dim ** 0.5)
        )

        # Learnable log-variances (for numerical stability)
        if covariance_type == "spherical":
            self.log_vars = nn.Parameter(torch.zeros(n_clusters, 1))
        elif covariance_type == "diagonal":
            self.log_vars = nn.Parameter(torch.zeros(n_clusters, dim))
        elif covariance_type == "full":
            # Use Cholesky factorization for full covariance
            self.chol_factors = nn.Parameter(
                torch.eye(dim).unsqueeze(0).repeat(n_clusters, 1, 1)
            )
        else:
            raise ValueError(f"Unknown covariance_type: {covariance_type}")

        # Learnable mixture weights (in log space)
        self.log_weights = nn.Parameter(torch.zeros(n_clusters))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior cluster assignments (responsibilities).

        Args:
            x: (batch_size, dim) data points

        Returns:
            responsibilities: (batch_size, n_clusters) posterior probabilities
        """
        batch_size = x.shape[0]

        # Compute log-likelihoods for each cluster
        if self.covariance_type == "spherical":
            vars = torch.exp(self.log_vars)  # (n_clusters, 1)
            diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (batch, n_clusters, dim)
            sq_mahal = (diff ** 2).sum(dim=2) / vars.squeeze(1)  # (batch, n_clusters)
            log_det = self.dim * self.log_vars.squeeze(1)

        elif self.covariance_type == "diagonal":
            vars = torch.exp(self.log_vars)  # (n_clusters, dim)
            diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (batch, n_clusters, dim)
            sq_mahal = ((diff ** 2) / vars.unsqueeze(0)).sum(dim=2)  # (batch, n_clusters)
            log_det = self.log_vars.sum(dim=1)

        elif self.covariance_type == "full":
            # More expensive computation for full covariance
            log_likelihoods = []
            for k in range(self.n_clusters):
                L = self.chol_factors[k]  # Lower triangular
                diff = x - self.means[k]  # (batch, dim)
                # Solve L @ z = diff for z
                z = torch.linalg.solve_triangular(L, diff.T, upper=False).T
                sq_mahal_k = (z ** 2).sum(dim=1)
                log_det_k = 2 * torch.log(torch.diag(L)).sum()
                log_likelihoods.append(-0.5 * (sq_mahal_k + log_det_k))
            log_likelihoods = torch.stack(log_likelihoods, dim=1)
            log_weights = F.log_softmax(self.log_weights, dim=0)
            log_posteriors = log_likelihoods + log_weights
            return F.softmax(log_posteriors, dim=1)

        # For spherical and diagonal cases
        log_likelihoods = -0.5 * (sq_mahal + log_det)  # (batch, n_clusters)
        log_weights = F.log_softmax(self.log_weights, dim=0)
        log_posteriors = log_likelihoods + log_weights

        responsibilities = F.softmax(log_posteriors, dim=1)
        return responsibilities

    def get_centroids(self) -> torch.Tensor:
        return self.means


class AutoClusterDiscovery(ClusteringAlgorithm):
    """
    Clustering with automatic discovery of number of clusters.

    Uses an overcomplete set of centroids with sparsity regularization
    to encourage using only necessary clusters.
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int,  # Maximum number of clusters
        temperature: float = 1.0,
        init_scale: float = 1.0,
    ):
        super().__init__(dim, n_clusters)

        # Overcomplete set of centroids
        self.centroids = nn.Parameter(
            torch.randn(n_clusters, dim) * init_scale / (dim ** 0.5)
        )

        # Learnable cluster "importance" weights (can be regularized to encourage sparsity)
        self.log_importance = nn.Parameter(torch.zeros(n_clusters))

        self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft assignments with importance weighting."""
        # Compute squared distances
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.centroids ** 2).sum(dim=1, keepdim=True).T
        cross = x @ self.centroids.T
        sq_distances = x_sq - 2 * cross + c_sq

        # Weight by cluster importance
        importance = F.softmax(self.log_importance, dim=0)

        # Combine distance and importance
        logits = -sq_distances / self.temperature + torch.log(importance + 1e-10)
        assignments = F.softmax(logits, dim=1)

        return assignments

    def get_centroids(self) -> torch.Tensor:
        return self.centroids

    def get_effective_n_clusters(self, threshold: float = 0.01) -> int:
        """Count clusters with importance above threshold."""
        importance = F.softmax(self.log_importance, dim=0)
        return (importance > threshold).sum().item()

    def get_sparsity_loss(self) -> torch.Tensor:
        """Regularization loss to encourage using fewer clusters."""
        importance = F.softmax(self.log_importance, dim=0)
        # Entropy encourages sparse usage
        entropy = -(importance * torch.log(importance + 1e-10)).sum()
        return entropy


def get_clustering_algorithm(
    name: str,
    dim: int,
    n_clusters: int,
    **kwargs,
) -> ClusteringAlgorithm:
    """Factory function to create clustering algorithms."""
    algorithms = {
        "soft_kmeans": SoftKMeans,
        "gmm": DifferentiableGMM,
        "auto_discovery": AutoClusterDiscovery,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown clustering algorithm: {name}. Available: {list(algorithms.keys())}")

    return algorithms[name](dim=dim, n_clusters=n_clusters, **kwargs)
