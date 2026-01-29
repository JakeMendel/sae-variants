#!/usr/bin/env python
"""
Multi-subspace experiments:
1. Scale to 500 clusters
2. Variable number of subspaces (K=2, 4, etc.)
3. Baselines for comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import geoopt

from src.metrics import clustering_accuracy, subspace_alignment_score
from src.utils import get_best_device


# ============================================================
# MULTI-SUBSPACE DATA GENERATION
# ============================================================

@dataclass
class MultiSubspaceDataConfig:
    """Configuration for multi-subspace synthetic data."""
    d: int = 128                    # Total dimension
    n_subspaces: int = 4            # Number of orthogonal subspaces
    n_clusters_per_subspace: int = 50  # Clusters in each subspace
    n_samples: int = 10000
    noise_std: float = 0.0
    centroid_scale: float = 1.0
    seed: Optional[int] = 42


class MultiSubspaceDataset:
    """
    Dataset where each point belongs to one cluster in EACH of K subspaces.

    R^d is split into K orthogonal subspaces, each of dimension d/K.
    Each cluster type has centroids in its corresponding subspace.

    Embedding = centroid_1 + centroid_2 + ... + centroid_K + noise
    """

    def __init__(self, config: MultiSubspaceDataConfig):
        self.config = config
        self.d = config.d
        self.n_subspaces = config.n_subspaces
        self.d_per_subspace = config.d // config.n_subspaces

        assert config.d % config.n_subspaces == 0, \
            f"d ({config.d}) must be divisible by n_subspaces ({config.n_subspaces})"

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Generate centroids for each subspace
        self.centroids = []  # List of (n_clusters, d_per_subspace) tensors
        for _ in range(config.n_subspaces):
            c = torch.randn(config.n_clusters_per_subspace, self.d_per_subspace)
            c = F.normalize(c, dim=1) * config.centroid_scale
            self.centroids.append(c)

        # Assign each data point to one cluster in each subspace
        self.labels = []  # List of (n_samples,) tensors
        for _ in range(config.n_subspaces):
            labels = torch.randint(0, config.n_clusters_per_subspace, (config.n_samples,))
            self.labels.append(labels)

        # Generate embeddings
        self.embeddings = self._generate_embeddings()

        # Ground truth: subspace i is spanned by dims [i*d_sub : (i+1)*d_sub]
        # After rotation, we need to track the true bases
        self.true_subspace_bases = []  # List of (d, d_per_subspace) tensors
        for i in range(config.n_subspaces):
            basis = torch.zeros(self.d, self.d_per_subspace)
            start = i * self.d_per_subspace
            end = (i + 1) * self.d_per_subspace
            basis[start:end, :] = torch.eye(self.d_per_subspace)
            self.true_subspace_bases.append(basis)

        self.rotation_matrix = None

    def _generate_embeddings(self):
        """Generate embeddings from cluster assignments."""
        embeddings = torch.zeros(self.config.n_samples, self.d)

        for i in range(self.n_subspaces):
            start = i * self.d_per_subspace
            end = (i + 1) * self.d_per_subspace
            embeddings[:, start:end] = self.centroids[i][self.labels[i]]

        # Add noise
        if self.config.noise_std > 0:
            embeddings = embeddings + torch.randn_like(embeddings) * self.config.noise_std

        return embeddings

    def apply_random_rotation(self, seed: Optional[int] = None):
        """Apply random orthogonal rotation to make problem non-trivial."""
        if seed is not None:
            torch.manual_seed(seed)

        # Generate random orthogonal matrix
        random_matrix = torch.randn(self.d, self.d)
        Q, _ = torch.linalg.qr(random_matrix)

        # Rotate embeddings
        self.embeddings = self.embeddings @ Q.T

        # Update true subspace bases
        for i in range(self.n_subspaces):
            self.true_subspace_bases[i] = Q @ self.true_subspace_bases[i]

        self.rotation_matrix = Q
        return Q


def generate_multi_subspace_data(
    d: int = 128,
    n_subspaces: int = 4,
    n_clusters_per_subspace: int = 50,
    n_samples: int = 10000,
    noise_std: float = 0.0,
    apply_rotation: bool = True,
    seed: int = 42,
) -> MultiSubspaceDataset:
    """Convenience function to generate multi-subspace data."""
    config = MultiSubspaceDataConfig(
        d=d,
        n_subspaces=n_subspaces,
        n_clusters_per_subspace=n_clusters_per_subspace,
        n_samples=n_samples,
        noise_std=noise_std,
        seed=seed,
    )
    dataset = MultiSubspaceDataset(config)
    if apply_rotation:
        dataset.apply_random_rotation(seed=seed + 1 if seed else None)
    return dataset


# ============================================================
# MULTI-SUBSPACE OPTIMIZER (Full K-Means)
# ============================================================

class MultiSubspaceOptimizer:
    """
    Optimizer for K orthogonal subspaces using full k-means.

    Learns an orthogonal matrix Q ∈ O(d) such that:
    - Columns 0:d/K span subspace 1
    - Columns d/K:2d/K span subspace 2
    - etc.
    """

    def __init__(
        self,
        d: int,
        n_subspaces: int,
        n_clusters_per_subspace: int,
        device: str = 'cpu',
        lr: float = 1e-2,
    ):
        self.d = d
        self.n_subspaces = n_subspaces
        self.d_per_subspace = d // n_subspaces
        self.n_clusters = n_clusters_per_subspace
        self.device = device

        # Initialize orthogonal matrix (QR on CPU, then move to device)
        self.manifold = geoopt.Stiefel()
        random_matrix = torch.randn(d, d)
        Q, _ = torch.linalg.qr(random_matrix)
        Q = Q.to(device)
        self.Q = geoopt.ManifoldParameter(Q, manifold=self.manifold)

        self.optimizer = geoopt.optim.RiemannianAdam([self.Q], lr=lr)

        # Cluster info (computed by k-means)
        self.centroids = [None] * n_subspaces  # List of (n_clusters, d_per_subspace)
        self.cluster_labels = [None] * n_subspaces  # List of (n_samples,)

    def project_to_subspace(self, x: torch.Tensor, subspace_idx: int) -> torch.Tensor:
        """Project data onto subspace i."""
        start = subspace_idx * self.d_per_subspace
        end = (subspace_idx + 1) * self.d_per_subspace
        rotated = x @ self.Q
        return rotated[:, start:end]

    def run_kmeans_all_subspaces(self, data: torch.Tensor):
        """Run full k-means in each subspace."""
        for i in range(self.n_subspaces):
            proj = self.project_to_subspace(data, i)
            proj_np = proj.detach().cpu().numpy()

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)
            labels = kmeans.fit_predict(proj_np)

            self.cluster_labels[i] = torch.tensor(labels, device=self.device)
            self.centroids[i] = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )

    def compute_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Compute total reconstruction loss across all subspaces."""
        total_loss = 0.0

        for i in range(self.n_subspaces):
            proj = self.project_to_subspace(data, i)
            recon = self.centroids[i][self.cluster_labels[i]]
            loss = ((proj - recon) ** 2).mean()
            total_loss = total_loss + loss

        return total_loss

    def optimize_subspace_step(self, data: torch.Tensor) -> float:
        """One gradient step to optimize Q."""
        loss = self.compute_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        dataset: MultiSubspaceDataset,
        n_outer_iters: int = 50,
        subspace_steps_per_iter: int = 20,
        verbose: bool = True,
        eval_interval: int = 10,
    ):
        """Full training loop with alternating optimization."""
        data = dataset.embeddings.to(self.device)

        history = []

        for outer_iter in range(1, n_outer_iters + 1):
            # Phase 1: Run full k-means in each subspace
            self.run_kmeans_all_subspaces(data)

            # Phase 2: Optimize Q
            loss = 0
            for _ in range(subspace_steps_per_iter):
                loss = self.optimize_subspace_step(data)

            # Evaluate
            if verbose and outer_iter % eval_interval == 0:
                metrics = self.evaluate(dataset)
                history.append(metrics)
                subspace_str = ", ".join([f"{s:.3f}" for s in metrics['subspace_aligns']])
                ari_str = ", ".join([f"{a:.3f}" for a in metrics['aris']])
                print(f"Iter {outer_iter}: loss={loss:.4f}, "
                      f"subspace=[{subspace_str}], mean={metrics['subspace_mean']:.3f}, "
                      f"ari=[{ari_str}], mean={metrics['ari_mean']:.3f}")

        return history

    def evaluate(self, dataset: MultiSubspaceDataset) -> dict:
        """Evaluate against ground truth."""
        Q_cpu = self.Q.detach().cpu()

        # Get learned bases
        learned_bases = []
        for i in range(self.n_subspaces):
            start = i * self.d_per_subspace
            end = (i + 1) * self.d_per_subspace
            basis = Q_cpu[:, start:end]
            learned_bases.append(basis)

        # Compute subspace alignments (try all permutations for best matching)
        true_bases = dataset.true_subspace_bases

        # Simple greedy matching (could use Hungarian for optimal)
        used_true = set()
        subspace_aligns = []
        matching = []  # (learned_idx, true_idx)

        for i, lb in enumerate(learned_bases):
            best_align = -1
            best_j = -1
            for j, tb in enumerate(true_bases):
                if j in used_true:
                    continue
                align = subspace_alignment_score(lb, tb)
                if align > best_align:
                    best_align = align
                    best_j = j

            subspace_aligns.append(best_align)
            matching.append((i, best_j))
            used_true.add(best_j)

        # Clustering metrics using the matching
        aris = []
        accs = []

        for learned_idx, true_idx in matching:
            pred_labels = self.cluster_labels[learned_idx].cpu()
            true_labels = dataset.labels[true_idx]

            ari = adjusted_rand_score(true_labels.numpy(), pred_labels.numpy())
            acc, _ = clustering_accuracy(pred_labels, true_labels)

            aris.append(ari)
            accs.append(acc)

        return {
            'subspace_aligns': subspace_aligns,
            'subspace_mean': np.mean(subspace_aligns),
            'aris': aris,
            'ari_mean': np.mean(aris),
            'accs': accs,
            'acc_mean': np.mean(accs),
            'matching': matching,
        }


# ============================================================
# BASELINES
# ============================================================

def baseline_kmeans_full_space(dataset: MultiSubspaceDataset) -> dict:
    """Baseline: run k-means on full space."""
    data = dataset.embeddings.numpy()
    total_clusters = dataset.n_subspaces * dataset.config.n_clusters_per_subspace

    kmeans = KMeans(n_clusters=total_clusters, random_state=42, n_init=3)
    pred = kmeans.fit_predict(data)

    # Compare to each ground truth label set
    aris = []
    for i in range(dataset.n_subspaces):
        ari = adjusted_rand_score(dataset.labels[i].numpy(), pred)
        aris.append(ari)

    return {
        'method': 'K-means (full space)',
        'aris': aris,
        'ari_mean': np.mean(aris),
    }


def baseline_oracle(dataset: MultiSubspaceDataset) -> dict:
    """Baseline: k-means with true subspace."""
    aris = []
    accs = []

    for i in range(dataset.n_subspaces):
        # Project to true subspace
        basis = dataset.true_subspace_bases[i]
        proj = (dataset.embeddings @ basis).numpy()

        # Run k-means
        kmeans = KMeans(n_clusters=dataset.config.n_clusters_per_subspace,
                       random_state=42, n_init=10)
        pred = kmeans.fit_predict(proj)

        ari = adjusted_rand_score(dataset.labels[i].numpy(), pred)
        acc, _ = clustering_accuracy(torch.tensor(pred), dataset.labels[i])

        aris.append(ari)
        accs.append(acc)

    return {
        'method': 'Oracle (true subspace + k-means)',
        'aris': aris,
        'ari_mean': np.mean(aris),
        'accs': accs,
        'acc_mean': np.mean(accs),
    }


# ============================================================
# MAIN
# ============================================================

def run_experiment(
    name: str,
    d: int,
    n_subspaces: int,
    n_clusters: int,
    noise_std: float,
    device: str,
    n_iters: int = 50,
):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"d={d}, n_subspaces={n_subspaces}, n_clusters={n_clusters}, noise={noise_std}")
    print('='*70)

    # Generate data
    dataset = generate_multi_subspace_data(
        d=d,
        n_subspaces=n_subspaces,
        n_clusters_per_subspace=n_clusters,
        n_samples=10000,
        noise_std=noise_std,
        seed=42,
    )

    # Baselines
    print("\nBaselines:")
    oracle = baseline_oracle(dataset)
    print(f"  Oracle (true subspace): ARI={oracle['ari_mean']:.4f}, Acc={oracle['acc_mean']:.4f}")

    kmeans_full = baseline_kmeans_full_space(dataset)
    print(f"  K-means (full space): ARI={kmeans_full['ari_mean']:.4f}")

    # Our method
    print("\nTraining multi-subspace optimizer...")
    opt = MultiSubspaceOptimizer(
        d=d,
        n_subspaces=n_subspaces,
        n_clusters_per_subspace=n_clusters,
        device=device,
        lr=1e-2,
    )
    opt.train(dataset, n_outer_iters=n_iters, subspace_steps_per_iter=20, eval_interval=10)

    final = opt.evaluate(dataset)

    print(f"\nFinal results:")
    print(f"  Subspace alignment: {final['subspace_mean']:.4f}")
    print(f"  Clustering ARI: {final['ari_mean']:.4f}")
    print(f"  Clustering accuracy: {final['acc_mean']:.4f}")

    return {
        'name': name,
        'd': d,
        'n_subspaces': n_subspaces,
        'n_clusters': n_clusters,
        'noise_std': noise_std,
        'oracle': oracle,
        'kmeans_full': kmeans_full,
        'final': final,
    }


def main():
    device = get_best_device()
    print(f"Device: {device}")

    results = []

    # ========================================
    # EXPERIMENT 1: Scale to 500 clusters (2 subspaces)
    # ========================================
    print("\n" + "#"*70)
    print("# SCALING TO 500 CLUSTERS")
    print("#"*70)

    for n_clusters in [100, 200, 500]:
        r = run_experiment(
            name=f"K=2, {n_clusters} clusters",
            d=64,
            n_subspaces=2,
            n_clusters=n_clusters,
            noise_std=0.0,
            device=device,
            n_iters=50,
        )
        results.append(r)

    # ========================================
    # EXPERIMENT 2: 4 subspaces
    # ========================================
    print("\n" + "#"*70)
    print("# 4 SUBSPACES")
    print("#"*70)

    for n_clusters in [50, 100]:
        r = run_experiment(
            name=f"K=4, {n_clusters} clusters",
            d=128,  # Larger d for 4 subspaces (32 dims each)
            n_subspaces=4,
            n_clusters=n_clusters,
            noise_std=0.0,
            device=device,
            n_iters=60,
        )
        results.append(r)

    # With noise
    r = run_experiment(
        name="K=4, 50 clusters, noisy",
        d=128,
        n_subspaces=4,
        n_clusters=50,
        noise_std=0.1,
        device=device,
        n_iters=60,
    )
    results.append(r)

    # ========================================
    # EXPERIMENT 3: Compare K=2,3,4,5 subspaces
    # ========================================
    print("\n" + "#"*70)
    print("# VARYING NUMBER OF SUBSPACES")
    print("#"*70)

    for n_subspaces in [2, 3, 4, 5]:
        d = 60 * n_subspaces  # Keep d/K = 60
        r = run_experiment(
            name=f"K={n_subspaces} subspaces",
            d=d,
            n_subspaces=n_subspaces,
            n_clusters=50,
            noise_std=0.0,
            device=device,
            n_iters=50,
        )
        results.append(r)

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Name':<35} {'d':>6} {'K':>4} {'Clust':>6} {'Subspace':>10} {'ARI':>10} {'Oracle ARI':>12}")
    print("-"*80)

    for r in results:
        print(f"{r['name']:<35} {r['d']:>6} {r['n_subspaces']:>4} {r['n_clusters']:>6} "
              f"{r['final']['subspace_mean']:>10.4f} {r['final']['ari_mean']:>10.4f} "
              f"{r['oracle']['ari_mean']:>12.4f}")


if __name__ == "__main__":
    main()
