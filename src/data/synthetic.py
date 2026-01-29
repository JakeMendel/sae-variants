"""
Synthetic data generation for multi-cluster subspace experiments.

Each data point belongs to exactly one cluster of type A and one cluster of type B.
The embedding is: centroid_A (in subspace 1) + centroid_B (in subspace 2) + noise
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    d: int = 64  # Total embedding dimension
    n_clusters_a: int = 50  # Number of type-A clusters (can exceed d/2)
    n_clusters_b: int = 50  # Number of type-B clusters (can exceed d/2)
    n_samples: int = 10000  # Total number of data points
    noise_std: float = 0.1  # Gaussian noise standard deviation
    centroid_scale: float = 1.0  # Scale of cluster centroids
    seed: Optional[int] = None  # Random seed for reproducibility


class SyntheticMultiClusterDataset(Dataset):
    """
    Dataset where each point belongs to one A-cluster and one B-cluster.

    The ground truth structure:
    - Subspace 1 (first d/2 dims): encodes cluster A membership
    - Subspace 2 (last d/2 dims): encodes cluster B membership
    - Each cluster has a centroid in its respective subspace
    - Data point = centroid_A + centroid_B + noise
    """

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.d = config.d
        self.d_half = config.d // 2

        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Generate cluster centroids
        # Centroids for A clusters live in subspace 1 (first d/2 dims)
        # Centroids for B clusters live in subspace 2 (last d/2 dims)
        self.centroids_a = self._generate_centroids(config.n_clusters_a, self.d_half)
        self.centroids_b = self._generate_centroids(config.n_clusters_b, self.d_half)

        # Assign each data point to clusters
        self.labels_a = torch.randint(0, config.n_clusters_a, (config.n_samples,))
        self.labels_b = torch.randint(0, config.n_clusters_b, (config.n_samples,))

        # Generate embeddings
        self.embeddings = self._generate_embeddings()

        # Store the ground truth subspace basis (identity in this case)
        # The true subspace 1 is spanned by first d/2 standard basis vectors
        # The true subspace 2 is spanned by last d/2 standard basis vectors
        self.true_subspace_a = torch.eye(self.d, self.d_half)[:, :self.d_half]
        self.true_subspace_b = torch.eye(self.d, self.d_half)
        self.true_subspace_b[:self.d_half, :] = 0
        self.true_subspace_b[self.d_half:, :] = torch.eye(self.d_half)

    def _generate_centroids(self, n_clusters: int, dim: int) -> torch.Tensor:
        """
        Generate cluster centroids. When n_clusters > dim, centroids form
        an overcomplete dictionary (not all orthogonal).
        """
        centroids = torch.randn(n_clusters, dim)
        # Normalize to have consistent scale
        centroids = centroids / centroids.norm(dim=1, keepdim=True)
        centroids = centroids * self.config.centroid_scale
        return centroids

    def _generate_embeddings(self) -> torch.Tensor:
        """Generate all embeddings from cluster assignments."""
        embeddings = torch.zeros(self.config.n_samples, self.d)

        # Add centroid_A contribution (in first d/2 dims)
        embeddings[:, :self.d_half] = self.centroids_a[self.labels_a]

        # Add centroid_B contribution (in last d/2 dims)
        embeddings[:, self.d_half:] = self.centroids_b[self.labels_b]

        # Add Gaussian noise
        noise = torch.randn_like(embeddings) * self.config.noise_std
        embeddings = embeddings + noise

        return embeddings

    def __len__(self) -> int:
        return self.config.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Returns (embedding, label_a, label_b)."""
        return self.embeddings[idx], self.labels_a[idx], self.labels_b[idx]

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns all data at once: (embeddings, labels_a, labels_b)."""
        return self.embeddings, self.labels_a, self.labels_b

    def apply_random_rotation(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Apply a random orthogonal transformation to embeddings.
        This makes the subspace discovery problem non-trivial.
        Returns the rotation matrix used.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Generate random orthogonal matrix via QR decomposition
        random_matrix = torch.randn(self.d, self.d)
        Q, _ = torch.linalg.qr(random_matrix)

        # Apply rotation to embeddings
        self.embeddings = self.embeddings @ Q.T

        # Update ground truth subspaces
        self.true_subspace_a = Q @ self.true_subspace_a
        self.true_subspace_b = Q @ self.true_subspace_b

        # Store rotation for reference
        self.rotation_matrix = Q

        return Q


def generate_synthetic_data(
    d: int = 64,
    n_clusters_a: int = 50,
    n_clusters_b: int = 50,
    n_samples: int = 10000,
    noise_std: float = 0.1,
    apply_rotation: bool = True,
    seed: Optional[int] = 42,
) -> SyntheticMultiClusterDataset:
    """
    Convenience function to generate synthetic multi-cluster data.

    Args:
        d: Total embedding dimension
        n_clusters_a: Number of A-type clusters
        n_clusters_b: Number of B-type clusters
        n_samples: Number of data points
        noise_std: Standard deviation of Gaussian noise
        apply_rotation: If True, apply random rotation to make problem harder
        seed: Random seed for reproducibility

    Returns:
        SyntheticMultiClusterDataset with embeddings and ground truth labels
    """
    config = SyntheticDataConfig(
        d=d,
        n_clusters_a=n_clusters_a,
        n_clusters_b=n_clusters_b,
        n_samples=n_samples,
        noise_std=noise_std,
        seed=seed,
    )

    dataset = SyntheticMultiClusterDataset(config)

    if apply_rotation:
        dataset.apply_random_rotation(seed=seed + 1 if seed else None)

    return dataset
