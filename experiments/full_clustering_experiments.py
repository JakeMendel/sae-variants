#!/usr/bin/env python
"""
Experiments with:
1. Full k-means clustering between gradient steps
2. ReLU SAE baseline
3. Scaling number of clusters (100, 200, 500)
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
import geoopt

from src.data import generate_synthetic_data
from src.metrics import clustering_accuracy, evaluate_full_model, subspace_alignment_score
from src.utils import get_best_device


# ============================================================
# RELU SAE
# ============================================================

class ReLUSAE(nn.Module):
    """Standard ReLU Sparse Autoencoder."""

    def __init__(self, d_input: int, n_latents: int, l1_coef: float = 1e-3):
        super().__init__()
        self.d_input = d_input
        self.n_latents = n_latents
        self.l1_coef = l1_coef

        self.encoder = nn.Linear(d_input, n_latents)
        self.decoder = nn.Linear(n_latents, d_input, bias=True)

        # Initialize decoder with unit norm columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        latents = F.relu(self.encoder(x))
        recon = self.decoder(latents)
        return recon, latents

    def loss(self, x):
        recon, latents = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        l1_loss = latents.abs().mean()
        return recon_loss + self.l1_coef * l1_loss, {'recon': recon_loss, 'l1': l1_loss}


def train_relu_sae(d, n_latents, dataset, n_epochs=100, lr=1e-3, l1_coef=1e-3, device='cpu'):
    """Train ReLU SAE and evaluate clustering by mapping latents to clusters."""
    model = ReLUSAE(d, n_latents, l1_coef).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = dataset.embeddings.to(device)

    for epoch in range(n_epochs):
        loss, _ = model.loss(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder
        with torch.no_grad():
            model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)

    # Get latent activations
    model.eval()
    with torch.no_grad():
        _, latents = model.forward(data)
        latents = latents.cpu()

    # Find top-2 active latents per sample (like TopK SAE)
    topk_vals, topk_idx = torch.topk(latents, k=2, dim=1)

    # Evaluate clustering
    pred_0 = topk_idx[:, 0].numpy()
    pred_1 = topk_idx[:, 1].numpy()

    labels_a = dataset.labels_a.numpy()
    labels_b = dataset.labels_b.numpy()

    # Try both orderings
    ari_0a = adjusted_rand_score(labels_a, pred_0)
    ari_0b = adjusted_rand_score(labels_b, pred_0)
    ari_1a = adjusted_rand_score(labels_a, pred_1)
    ari_1b = adjusted_rand_score(labels_b, pred_1)

    if (ari_0a + ari_1b) >= (ari_0b + ari_1a):
        ari_a, ari_b = ari_0a, ari_1b
    else:
        ari_a, ari_b = ari_1a, ari_0b

    # Reconstruction error
    with torch.no_grad():
        recon, _ = model.forward(data)
        mse = F.mse_loss(recon, data).item()

    return {
        'ari_a': ari_a,
        'ari_b': ari_b,
        'ari_mean': (ari_a + ari_b) / 2,
        'mse': mse,
        'latent_sparsity': (latents > 0).float().mean().item(),
    }


# ============================================================
# FULL CLUSTERING OPTIMIZATION
# ============================================================

class FullClusteringOptimizer:
    """
    Alternating optimization with FULL k-means clustering.

    1. Fix subspace Q, run k-means to convergence in each subspace
    2. Fix cluster assignments, optimize Q via gradient descent
    """

    def __init__(self, d: int, n_clusters_a: int, n_clusters_b: int, device='cpu'):
        self.d = d
        self.d_half = d // 2
        self.n_clusters_a = n_clusters_a
        self.n_clusters_b = n_clusters_b
        self.device = device

        # Initialize orthogonal matrix (QR on CPU, then move to device)
        self.manifold = geoopt.Stiefel()
        random_matrix = torch.randn(d, d)
        Q, _ = torch.linalg.qr(random_matrix)
        Q = Q.to(device)
        self.Q = geoopt.ManifoldParameter(Q, manifold=self.manifold)

        self.optimizer = geoopt.optim.RiemannianAdam([self.Q], lr=1e-2)

        # Cluster centroids (computed by k-means, not learned)
        self.centroids_a = None
        self.centroids_b = None
        self.labels_a = None
        self.labels_b = None

    def project_to_subspaces(self, x: torch.Tensor):
        """Project data onto current subspaces."""
        rotated = x @ self.Q
        return rotated[:, :self.d_half], rotated[:, self.d_half:]

    def run_kmeans(self, data: torch.Tensor):
        """Run full k-means in each subspace."""
        x_a, x_b = self.project_to_subspaces(data)
        x_a_np = x_a.detach().cpu().numpy()
        x_b_np = x_b.detach().cpu().numpy()

        # K-means in subspace A
        kmeans_a = KMeans(n_clusters=self.n_clusters_a, random_state=42, n_init=3)
        self.labels_a = torch.tensor(kmeans_a.fit_predict(x_a_np), device=self.device)
        self.centroids_a = torch.tensor(kmeans_a.cluster_centers_, dtype=torch.float32, device=self.device)

        # K-means in subspace B
        kmeans_b = KMeans(n_clusters=self.n_clusters_b, random_state=42, n_init=3)
        self.labels_b = torch.tensor(kmeans_b.fit_predict(x_b_np), device=self.device)
        self.centroids_b = torch.tensor(kmeans_b.cluster_centers_, dtype=torch.float32, device=self.device)

    def compute_loss(self, data: torch.Tensor):
        """Compute reconstruction loss with fixed cluster assignments."""
        x_a, x_b = self.project_to_subspaces(data)

        # Reconstruct using assigned centroids
        recon_a = self.centroids_a[self.labels_a]
        recon_b = self.centroids_b[self.labels_b]

        # Loss in each subspace
        loss_a = ((x_a - recon_a) ** 2).mean()
        loss_b = ((x_b - recon_b) ** 2).mean()

        return loss_a + loss_b

    def optimize_subspace_step(self, data: torch.Tensor):
        """One gradient step to optimize Q."""
        loss = self.compute_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataset, n_outer_iters=50, subspace_steps_per_iter=10, verbose=True):
        """Full training loop."""
        data = dataset.embeddings.to(self.device)

        history = []

        for outer_iter in range(1, n_outer_iters + 1):
            # Phase 1: Run full k-means with current Q
            self.run_kmeans(data)

            # Phase 2: Optimize Q with fixed clusters
            for _ in range(subspace_steps_per_iter):
                loss = self.optimize_subspace_step(data)

            # Evaluate
            if verbose and outer_iter % 10 == 0:
                metrics = self.evaluate(dataset)
                history.append(metrics)
                print(f"Iter {outer_iter}: loss={loss:.4f}, subspace={metrics['subspace']:.4f}, "
                      f"acc={metrics['acc']:.4f}, ari={metrics['ari']:.4f}")

        return history

    def evaluate(self, dataset):
        """Evaluate against ground truth."""
        # Subspace alignment
        Q_np = self.Q.detach().cpu()
        learned_basis_a = Q_np[:, :self.d_half]
        learned_basis_b = Q_np[:, self.d_half:]

        true_basis_a = dataset.true_subspace_a
        true_basis_b = dataset.true_subspace_b

        align_a = subspace_alignment_score(learned_basis_a, true_basis_a)
        align_b = subspace_alignment_score(learned_basis_b, true_basis_b)
        align_swap_a = subspace_alignment_score(learned_basis_a, true_basis_b)
        align_swap_b = subspace_alignment_score(learned_basis_b, true_basis_a)

        if (align_a + align_b) >= (align_swap_a + align_swap_b):
            subspace_align = (align_a + align_b) / 2
            labels_a_pred = self.labels_a.cpu()
            labels_b_pred = self.labels_b.cpu()
        else:
            subspace_align = (align_swap_a + align_swap_b) / 2
            labels_a_pred = self.labels_b.cpu()
            labels_b_pred = self.labels_a.cpu()

        # Clustering metrics
        true_a = dataset.labels_a
        true_b = dataset.labels_b

        acc_a, _ = clustering_accuracy(labels_a_pred, true_a)
        acc_b, _ = clustering_accuracy(labels_b_pred, true_b)
        ari_a = adjusted_rand_score(true_a.numpy(), labels_a_pred.numpy())
        ari_b = adjusted_rand_score(true_b.numpy(), labels_b_pred.numpy())

        return {
            'subspace': subspace_align,
            'acc': (acc_a + acc_b) / 2,
            'ari': (ari_a + ari_b) / 2,
            'acc_a': acc_a, 'acc_b': acc_b,
            'ari_a': ari_a, 'ari_b': ari_b,
        }


# ============================================================
# MAIN
# ============================================================

def main():
    device = get_best_device()
    print(f"Device: {device}")

    d = 64
    n_samples = 10000

    results = []

    # ========================================
    # EXPERIMENT 1: Full clustering optimization
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Full K-Means Between Gradient Steps")
    print("="*70)

    for noise_std in [0.0, 0.1]:
        for n_clusters in [50]:
            print(f"\n--- noise={noise_std}, n_clusters={n_clusters} ---")

            dataset = generate_synthetic_data(
                d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
                n_samples=n_samples, noise_std=noise_std, seed=42
            )

            opt = FullClusteringOptimizer(d, n_clusters, n_clusters, device=device)
            opt.train(dataset, n_outer_iters=50, subspace_steps_per_iter=20)

            m = opt.evaluate(dataset)
            results.append({
                'name': f'FullKMeans noise={noise_std} k={n_clusters}',
                **m
            })

    # ========================================
    # EXPERIMENT 2: ReLU SAE
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: ReLU SAE")
    print("="*70)

    for noise_std in [0.0, 0.1]:
        for n_clusters in [50]:
            print(f"\n--- noise={noise_std}, n_latents={n_clusters*2} ---")

            dataset = generate_synthetic_data(
                d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
                n_samples=n_samples, noise_std=noise_std, seed=42
            )

            m = train_relu_sae(d, n_clusters * 2, dataset, n_epochs=200,
                              l1_coef=1e-3, device=device)
            print(f"  ARI: {m['ari_mean']:.4f}, MSE: {m['mse']:.6f}, Sparsity: {m['latent_sparsity']:.4f}")

            results.append({
                'name': f'ReLU SAE noise={noise_std} latents={n_clusters*2}',
                'subspace': 'N/A',
                'acc': 'N/A',
                'ari': m['ari_mean'],
                'mse': m['mse'],
            })

    # ========================================
    # EXPERIMENT 3: Scaling clusters
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Scaling Number of Clusters (Zero Noise)")
    print("="*70)

    for n_clusters in [50, 100, 200]:
        print(f"\n--- n_clusters={n_clusters} per subspace ---")

        dataset = generate_synthetic_data(
            d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
            n_samples=n_samples, noise_std=0.0, seed=42
        )

        opt = FullClusteringOptimizer(d, n_clusters, n_clusters, device=device)
        opt.train(dataset, n_outer_iters=50, subspace_steps_per_iter=20)

        m = opt.evaluate(dataset)
        results.append({
            'name': f'FullKMeans k={n_clusters} (zero noise)',
            **m
        })

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Name':<45} {'Subspace':>10} {'Accuracy':>10} {'ARI':>10}")
    print("-"*80)
    for r in results:
        sub = f"{r['subspace']:.4f}" if isinstance(r['subspace'], float) else r['subspace']
        acc = f"{r['acc']:.4f}" if isinstance(r['acc'], float) else r['acc']
        ari = f"{r['ari']:.4f}" if isinstance(r['ari'], float) else f"{r['ari']}"
        print(f"{r['name']:<45} {sub:>10} {acc:>10} {ari:>10}")

    print("\nOracle baselines:")
    print(f"{'True subspace + k-means (noisy)':<45} {'1.0000':>10} {'0.9613':>10} {'0.9650':>10}")
    print(f"{'True subspace + true centroids':<45} {'1.0000':>10} {'1.0000':>10} {'1.0000':>10}")


if __name__ == "__main__":
    main()
