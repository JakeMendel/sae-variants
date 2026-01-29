"""
Subspace parameterization using orthogonal matrices on the Stiefel manifold.

The model learns an orthogonal matrix Q in O(d) that transforms the data
such that the first d/2 dimensions correspond to subspace A and the last
d/2 dimensions correspond to subspace B.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import geoopt


class StiefelParameter(nn.Module):
    """
    A learnable orthogonal matrix on the Stiefel manifold St(d, d).
    This represents a full rotation/reflection of R^d.
    """

    def __init__(self, d: int, init: str = "identity"):
        super().__init__()
        self.d = d

        # Initialize orthogonal matrix
        if init == "identity":
            initial = torch.eye(d)
        elif init == "random":
            random_matrix = torch.randn(d, d)
            initial, _ = torch.linalg.qr(random_matrix)
        else:
            raise ValueError(f"Unknown init: {init}")

        # Use geoopt's Stiefel manifold for proper optimization
        self.manifold = geoopt.Stiefel()
        self.Q = geoopt.ManifoldParameter(initial, manifold=self.manifold)

    def forward(self) -> torch.Tensor:
        """Returns the current orthogonal matrix."""
        return self.Q

    def project_to_subspace_a(self, x: torch.Tensor) -> torch.Tensor:
        """Project data onto learned subspace A (first d/2 dims after rotation)."""
        d_half = self.d // 2
        rotated = x @ self.Q
        return rotated[:, :d_half]

    def project_to_subspace_b(self, x: torch.Tensor) -> torch.Tensor:
        """Project data onto learned subspace B (last d/2 dims after rotation)."""
        d_half = self.d // 2
        rotated = x @ self.Q
        return rotated[:, d_half:]

    def get_subspace_bases(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the basis vectors for each subspace in the original coordinate system.

        Returns:
            basis_a: (d, d/2) matrix whose columns span subspace A
            basis_b: (d, d/2) matrix whose columns span subspace B
        """
        d_half = self.d // 2
        # Q rotates from original to split coordinates
        # Q^T rotates from split coordinates back to original
        # The first d/2 columns of Q^T give the basis for subspace A
        Q_inv = self.Q.T  # For orthogonal matrices, inverse = transpose
        basis_a = Q_inv[:, :d_half]
        basis_b = Q_inv[:, d_half:]
        return basis_a, basis_b


class SubspaceModel(nn.Module):
    """
    Full model for multi-cluster subspace discovery.

    Components:
    1. Learnable orthogonal transformation (Stiefel manifold)
    2. Pluggable clustering algorithm for each subspace
    3. Pluggable loss function
    """

    def __init__(
        self,
        d: int,
        n_clusters_a: int,
        n_clusters_b: int,
        clustering_algorithm: str = "soft_kmeans",
        clustering_kwargs: Optional[dict] = None,
        init: str = "random",
    ):
        super().__init__()
        self.d = d
        self.d_half = d // 2
        self.n_clusters_a = n_clusters_a
        self.n_clusters_b = n_clusters_b

        # Learnable orthogonal matrix
        self.subspace_transform = StiefelParameter(d, init=init)

        # Clustering algorithms for each subspace
        from .clustering import get_clustering_algorithm

        clustering_kwargs = clustering_kwargs or {}
        self.clusterer_a = get_clustering_algorithm(
            clustering_algorithm,
            dim=self.d_half,
            n_clusters=n_clusters_a,
            **clustering_kwargs,
        )
        self.clusterer_b = get_clustering_algorithm(
            clustering_algorithm,
            dim=self.d_half,
            n_clusters=n_clusters_b,
            **clustering_kwargs,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: split data into subspaces and cluster each.

        Args:
            x: (batch_size, d) input embeddings

        Returns:
            x_a: (batch_size, d/2) projections onto subspace A
            x_b: (batch_size, d/2) projections onto subspace B
            assignments_a: (batch_size, n_clusters_a) soft cluster assignments for A
            assignments_b: (batch_size, n_clusters_b) soft cluster assignments for B
        """
        # Project onto subspaces
        x_a = self.subspace_transform.project_to_subspace_a(x)
        x_b = self.subspace_transform.project_to_subspace_b(x)

        # Get soft cluster assignments
        assignments_a = self.clusterer_a(x_a)
        assignments_b = self.clusterer_b(x_b)

        return x_a, x_b, assignments_a, assignments_b

    def get_reconstructions(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct data points from cluster assignments.

        Returns:
            recon_a: reconstruction in subspace A
            recon_b: reconstruction in subspace B
            recon_full: full reconstruction in original space
        """
        x_a, x_b, assignments_a, assignments_b = self.forward(x)

        # Reconstruct in each subspace using cluster centroids
        recon_a = assignments_a @ self.clusterer_a.get_centroids()
        recon_b = assignments_b @ self.clusterer_b.get_centroids()

        # Combine and rotate back to original space
        recon_split = torch.cat([recon_a, recon_b], dim=1)
        Q = self.subspace_transform()
        recon_full = recon_split @ Q.T

        return recon_a, recon_b, recon_full

    def get_hard_assignments(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hard cluster assignments (argmax of soft assignments)."""
        _, _, assignments_a, assignments_b = self.forward(x)
        hard_a = assignments_a.argmax(dim=1)
        hard_b = assignments_b.argmax(dim=1)
        return hard_a, hard_b
