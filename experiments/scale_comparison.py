#!/usr/bin/env python
"""
Scale up experiments and compare to TopK SAE.

Tests:
1. Larger cluster counts (500, 1000, 2000)
2. More subspaces (4, 6, 8)
3. TopK SAE baseline with k = n_subspaces
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
from typing import List
import time
import geoopt

from experiments.multi_subspace_experiments import (
    generate_multi_subspace_data,
    MultiSubspaceDataset,
    MultiSubspaceOptimizer,
    baseline_oracle,
)
from src.metrics import clustering_accuracy, subspace_alignment_score
from src.utils import get_best_device


# ============================================================
# TOPK SAE
# ============================================================

class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder."""

    def __init__(self, d_input: int, n_latents: int, k: int):
        super().__init__()
        self.d_input = d_input
        self.n_latents = n_latents
        self.k = k

        self.encoder = nn.Linear(d_input, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_input, bias=True)

        # Initialize
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor):
        """Returns (sparse_acts, topk_indices)."""
        pre_acts = self.encoder(x)

        # TopK
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=1)
        topk_vals = F.relu(topk_vals)  # Ensure non-negative

        # Create sparse tensor
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(1, topk_idx, topk_vals)

        return acts, topk_idx

    def decode(self, acts: torch.Tensor):
        return self.decoder(acts)

    def forward(self, x: torch.Tensor):
        acts, topk_idx = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, topk_idx

    def loss(self, x: torch.Tensor):
        recon, acts, _ = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        return recon_loss


def train_topk_sae(
    d: int,
    n_latents: int,
    k: int,
    dataset: MultiSubspaceDataset,
    n_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = 'cpu',
    verbose: bool = True,
):
    """Train TopK SAE and evaluate."""
    model = TopKSAE(d, n_latents, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data = dataset.embeddings.to(device)
    n_samples = data.shape[0]

    start_time = time.time()

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            x = data[batch_idx]

            loss = model.loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder columns
            with torch.no_grad():
                model.decoder.weight.data = F.normalize(model.decoder.weight.data, dim=0)

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.6f}")

    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    with torch.no_grad():
        _, topk_idx = model.encode(data)
        topk_idx = topk_idx.cpu()

    # Each sample has k active latents
    # Try to match each of the k active latents to one of the K subspaces
    n_subspaces = dataset.n_subspaces

    # For each active position (0 to k-1), compute ARI against each true label set
    best_aris = []
    for subspace_idx in range(n_subspaces):
        true_labels = dataset.labels[subspace_idx].numpy()

        best_ari = -1
        for pos in range(k):
            pred = topk_idx[:, pos].numpy()
            ari = adjusted_rand_score(true_labels, pred)
            best_ari = max(best_ari, ari)

        best_aris.append(best_ari)

    # Also compute overall metrics
    # Reconstruction error
    with torch.no_grad():
        recon, _, _ = model.forward(data)
        mse = F.mse_loss(recon, data).item()

    return {
        'aris': best_aris,
        'ari_mean': np.mean(best_aris),
        'mse': mse,
        'train_time': train_time,
    }


# ============================================================
# SCALED MULTI-SUBSPACE OPTIMIZER
# ============================================================

class ScaledMultiSubspaceOptimizer(MultiSubspaceOptimizer):
    """Multi-subspace optimizer with MiniBatchKMeans for speed."""

    def run_kmeans_all_subspaces(self, data: torch.Tensor):
        """Run MiniBatchKMeans for speed with large cluster counts."""
        for i in range(self.n_subspaces):
            proj = self.project_to_subspace(data, i)
            proj_np = proj.detach().cpu().numpy()

            # Use MiniBatchKMeans for large cluster counts
            if self.n_clusters > 100:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=3,
                    batch_size=min(1024, len(proj_np)),
                )
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)

            labels = kmeans.fit_predict(proj_np)

            self.cluster_labels[i] = torch.tensor(labels, device=self.device)
            self.centroids[i] = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

def run_comparison(
    name: str,
    d: int,
    n_subspaces: int,
    n_clusters: int,
    noise_std: float,
    device: str,
    n_iters: int = 50,
    sae_epochs: int = 200,
):
    """Run both methods and compare."""
    print(f"\n{'='*80}")
    print(f"Experiment: {name}")
    print(f"d={d}, K={n_subspaces}, clusters={n_clusters}, noise={noise_std}")
    print(f"Total features: {n_subspaces * n_clusters}")
    print('='*80)

    # Generate data
    print("\nGenerating data...")
    dataset = generate_multi_subspace_data(
        d=d,
        n_subspaces=n_subspaces,
        n_clusters_per_subspace=n_clusters,
        n_samples=10000,
        noise_std=noise_std,
        seed=42,
    )

    # Oracle baseline
    print("\nOracle baseline (true subspace + k-means)...")
    oracle = baseline_oracle(dataset)
    print(f"  Oracle ARI: {oracle['ari_mean']:.4f}")

    # Our method
    print("\nOur method (full k-means alternating)...")
    start_time = time.time()
    opt = ScaledMultiSubspaceOptimizer(
        d=d,
        n_subspaces=n_subspaces,
        n_clusters_per_subspace=n_clusters,
        device=device,
        lr=1e-2,
    )
    opt.train(dataset, n_outer_iters=n_iters, subspace_steps_per_iter=20,
              eval_interval=max(1, n_iters // 5))
    our_time = time.time() - start_time
    our_result = opt.evaluate(dataset)
    print(f"  Our method: subspace={our_result['subspace_mean']:.4f}, "
          f"ARI={our_result['ari_mean']:.4f}, time={our_time:.1f}s")

    # TopK SAE
    print(f"\nTopK SAE (k={n_subspaces}, n_latents={n_subspaces * n_clusters})...")
    sae_result = train_topk_sae(
        d=d,
        n_latents=n_subspaces * n_clusters,
        k=n_subspaces,
        dataset=dataset,
        n_epochs=sae_epochs,
        device=device,
        verbose=True,
    )
    print(f"  TopK SAE: ARI={sae_result['ari_mean']:.4f}, "
          f"MSE={sae_result['mse']:.6f}, time={sae_result['train_time']:.1f}s")

    return {
        'name': name,
        'd': d,
        'n_subspaces': n_subspaces,
        'n_clusters': n_clusters,
        'total_features': n_subspaces * n_clusters,
        'noise_std': noise_std,
        'oracle_ari': oracle['ari_mean'],
        'our_subspace': our_result['subspace_mean'],
        'our_ari': our_result['ari_mean'],
        'our_time': our_time,
        'sae_ari': sae_result['ari_mean'],
        'sae_mse': sae_result['mse'],
        'sae_time': sae_result['train_time'],
    }


def main():
    device = get_best_device()
    print(f"Device: {device}")

    results = []

    # ========================================
    # EXPERIMENT 1: Scale clusters (K=2)
    # ========================================
    print("\n" + "#"*80)
    print("# SCALING CLUSTERS (K=2 subspaces)")
    print("#"*80)

    for n_clusters in [100, 500, 1000]:
        r = run_comparison(
            name=f"K=2, {n_clusters} clusters",
            d=64,
            n_subspaces=2,
            n_clusters=n_clusters,
            noise_std=0.0,
            device=device,
            n_iters=50,
            sae_epochs=200,
        )
        results.append(r)

    # ========================================
    # EXPERIMENT 2: Scale clusters (K=4)
    # ========================================
    print("\n" + "#"*80)
    print("# SCALING CLUSTERS (K=4 subspaces)")
    print("#"*80)

    for n_clusters in [100, 250, 500]:
        r = run_comparison(
            name=f"K=4, {n_clusters} clusters",
            d=128,
            n_subspaces=4,
            n_clusters=n_clusters,
            noise_std=0.0,
            device=device,
            n_iters=60,
            sae_epochs=200,
        )
        results.append(r)

    # ========================================
    # EXPERIMENT 3: Scale subspaces
    # ========================================
    print("\n" + "#"*80)
    print("# SCALING SUBSPACES (100 clusters each)")
    print("#"*80)

    for n_subspaces in [2, 4, 6, 8]:
        d = 32 * n_subspaces  # 32 dims per subspace
        r = run_comparison(
            name=f"K={n_subspaces} subspaces",
            d=d,
            n_subspaces=n_subspaces,
            n_clusters=100,
            noise_std=0.0,
            device=device,
            n_iters=60,
            sae_epochs=200,
        )
        results.append(r)

    # ========================================
    # EXPERIMENT 4: With noise
    # ========================================
    print("\n" + "#"*80)
    print("# WITH NOISE")
    print("#"*80)

    for noise_std in [0.1, 0.2]:
        r = run_comparison(
            name=f"K=4, noise={noise_std}",
            d=128,
            n_subspaces=4,
            n_clusters=100,
            noise_std=noise_std,
            device=device,
            n_iters=60,
            sae_epochs=200,
        )
        results.append(r)

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*100)
    print("FINAL SUMMARY: Our Method vs TopK SAE")
    print("="*100)
    print(f"{'Name':<25} {'d':>5} {'K':>3} {'Clust':>6} {'Total':>7} "
          f"{'Our Sub':>8} {'Our ARI':>8} {'SAE ARI':>8} {'Oracle':>8}")
    print("-"*100)

    for r in results:
        print(f"{r['name']:<25} {r['d']:>5} {r['n_subspaces']:>3} {r['n_clusters']:>6} "
              f"{r['total_features']:>7} {r['our_subspace']:>8.4f} {r['our_ari']:>8.4f} "
              f"{r['sae_ari']:>8.4f} {r['oracle_ari']:>8.4f}")

    print("\n" + "="*100)
    print("TIMING COMPARISON")
    print("="*100)
    print(f"{'Name':<25} {'Our Time (s)':>15} {'SAE Time (s)':>15}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['our_time']:>15.1f} {r['sae_time']:>15.1f}")


if __name__ == "__main__":
    main()
