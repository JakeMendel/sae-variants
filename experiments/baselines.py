#!/usr/bin/env python
"""
Baseline comparisons:
1. Standard clustering (k-means) directly on vectors
2. TopK sparse autoencoder with k=2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

from src.data import generate_synthetic_data
from src.metrics import clustering_accuracy


class TopKSparseAutoencoder(nn.Module):
    """
    Standard TopK sparse autoencoder.

    encoder: x -> pre_acts -> topk(pre_acts) -> acts
    decoder: acts @ W_dec -> reconstruction
    """
    def __init__(self, d_input: int, n_latents: int, k: int):
        super().__init__()
        self.d_input = d_input
        self.n_latents = n_latents
        self.k = k

        # Encoder: linear to latent space
        self.W_enc = nn.Parameter(torch.randn(d_input, n_latents) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(n_latents))

        # Decoder: latent back to input
        self.W_dec = nn.Parameter(torch.randn(n_latents, d_input) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        # Normalize decoder columns
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (topk_acts, topk_indices)"""
        pre_acts = x @ self.W_enc + self.b_enc

        # TopK activation
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=1)

        # Create sparse activation (only top k are non-zero)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(1, topk_idx, F.relu(topk_vals))

        return acts, topk_idx

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts, topk_idx = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, topk_idx


def train_topk_sae(model, data, n_epochs=100, lr=1e-3, batch_size=256, device='cpu'):
    """Train TopK SAE with reconstruction loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = data.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            x = data[batch_idx].to(device)

            recon, acts, _ = model(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder
            with torch.no_grad():
                model.W_dec.data = F.normalize(model.W_dec.data, dim=1)

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss/n_batches:.6f}")

    return model


def evaluate_sae_clustering(model, data, labels_a, labels_b, device='cpu'):
    """
    Evaluate how well SAE latents correspond to true clusters.

    With k=2, each point activates exactly 2 latents.
    Ideally, one should correspond to cluster A, one to cluster B.
    """
    model.eval()
    with torch.no_grad():
        acts, topk_idx = model.encode(data.to(device))
        topk_idx = topk_idx.cpu()

    # topk_idx shape: (n_samples, 2) - the two active latents per sample
    # Try to match latent assignments to true cluster assignments

    # For each of the two active latents, compute clustering metrics
    pred_0 = topk_idx[:, 0].numpy()
    pred_1 = topk_idx[:, 1].numpy()

    labels_a_np = labels_a.numpy()
    labels_b_np = labels_b.numpy()

    # Try both orderings (latent 0 -> A, latent 1 -> B) and (latent 0 -> B, latent 1 -> A)
    ari_0a = adjusted_rand_score(labels_a_np, pred_0)
    ari_0b = adjusted_rand_score(labels_b_np, pred_0)
    ari_1a = adjusted_rand_score(labels_a_np, pred_1)
    ari_1b = adjusted_rand_score(labels_b_np, pred_1)

    # Best matching
    option1 = (ari_0a + ari_1b) / 2  # latent 0 -> A, latent 1 -> B
    option2 = (ari_0b + ari_1a) / 2  # latent 0 -> B, latent 1 -> A

    if option1 >= option2:
        ari_a, ari_b = ari_0a, ari_1b
        acc_a, _ = clustering_accuracy(torch.tensor(pred_0), labels_a)
        acc_b, _ = clustering_accuracy(torch.tensor(pred_1), labels_b)
    else:
        ari_a, ari_b = ari_1a, ari_0b
        acc_a, _ = clustering_accuracy(torch.tensor(pred_1), labels_a)
        acc_b, _ = clustering_accuracy(torch.tensor(pred_0), labels_b)

    return {
        'ari_a': ari_a,
        'ari_b': ari_b,
        'ari_mean': (ari_a + ari_b) / 2,
        'acc_a': acc_a,
        'acc_b': acc_b,
        'acc_mean': (acc_a + acc_b) / 2,
    }


def run_kmeans_baseline(data, labels_a, labels_b, n_clusters):
    """Run standard k-means on the full vectors."""
    # K-means with total number of clusters
    total_clusters = n_clusters * 2  # Hoping it finds A and B clusters

    kmeans = KMeans(n_clusters=total_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(data.numpy())

    # This is tricky - k-means gives us one clustering, but we have two ground truths
    # We can only compare against combined (A, B) labels
    combined_labels = labels_a.numpy() * n_clusters + labels_b.numpy()

    ari_combined = adjusted_rand_score(combined_labels, pred)
    nmi_combined = normalized_mutual_info_score(combined_labels, pred)

    # Also try k-means with just n_clusters and see how it matches each
    kmeans_a = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_a = kmeans_a.fit_predict(data.numpy())
    ari_vs_a = adjusted_rand_score(labels_a.numpy(), pred_a)
    ari_vs_b = adjusted_rand_score(labels_b.numpy(), pred_a)

    return {
        'kmeans_ari_combined': ari_combined,
        'kmeans_nmi_combined': nmi_combined,
        'kmeans_ari_vs_a': ari_vs_a,
        'kmeans_ari_vs_b': ari_vs_b,
    }


def main():
    print("="*70)
    print("BASELINE COMPARISONS")
    print("="*70)

    # Use same settings as main experiments
    d = 64
    n_clusters = 50  # Overcomplete
    n_samples = 10000

    print(f"\nSettings: d={d}, {n_clusters} clusters per subspace, {n_samples} samples")

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Generate data
    print("\nGenerating synthetic data...")
    dataset = generate_synthetic_data(
        d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
        n_samples=n_samples, noise_std=0.1, seed=42
    )

    data = dataset.embeddings
    labels_a = dataset.labels_a
    labels_b = dataset.labels_b

    # Baseline 1: K-means directly on vectors
    print("\n" + "-"*70)
    print("BASELINE 1: K-means clustering on raw vectors")
    print("-"*70)
    kmeans_results = run_kmeans_baseline(data, labels_a, labels_b, n_clusters)
    print(f"  K-means (k={n_clusters*2}) ARI vs combined labels: {kmeans_results['kmeans_ari_combined']:.4f}")
    print(f"  K-means (k={n_clusters*2}) NMI vs combined labels: {kmeans_results['kmeans_nmi_combined']:.4f}")
    print(f"  K-means (k={n_clusters}) ARI vs labels_a: {kmeans_results['kmeans_ari_vs_a']:.4f}")
    print(f"  K-means (k={n_clusters}) ARI vs labels_b: {kmeans_results['kmeans_ari_vs_b']:.4f}")

    # Baseline 2: TopK Sparse Autoencoder
    print("\n" + "-"*70)
    print("BASELINE 2: TopK Sparse Autoencoder (k=2)")
    print("-"*70)

    n_latents = n_clusters * 2  # 100 latents for 50+50 clusters
    k = 2  # Each point activates exactly 2 latents

    print(f"  n_latents={n_latents}, k={k}")
    print("  Training SAE...")

    sae = TopKSparseAutoencoder(d_input=d, n_latents=n_latents, k=k)
    sae = train_topk_sae(sae, data, n_epochs=100, lr=1e-3, device=device)

    print("  Evaluating...")
    sae = sae.cpu()
    sae_results = evaluate_sae_clustering(sae, data, labels_a, labels_b, device='cpu')

    print(f"  SAE clustering accuracy A: {sae_results['acc_a']:.4f}")
    print(f"  SAE clustering accuracy B: {sae_results['acc_b']:.4f}")
    print(f"  SAE clustering accuracy mean: {sae_results['acc_mean']:.4f}")
    print(f"  SAE ARI A: {sae_results['ari_a']:.4f}")
    print(f"  SAE ARI B: {sae_results['ari_b']:.4f}")
    print(f"  SAE ARI mean: {sae_results['ari_mean']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'ARI (A)':<12} {'ARI (B)':<12} {'ARI (mean)':<12}")
    print("-"*70)
    print(f"{'K-means (k=n_clusters)':<30} {kmeans_results['kmeans_ari_vs_a']:<12.4f} {kmeans_results['kmeans_ari_vs_b']:<12.4f} {(kmeans_results['kmeans_ari_vs_a']+kmeans_results['kmeans_ari_vs_b'])/2:<12.4f}")
    print(f"{'TopK SAE (k=2)':<30} {sae_results['ari_a']:<12.4f} {sae_results['ari_b']:<12.4f} {sae_results['ari_mean']:<12.4f}")
    print(f"{'Random':<30} {'~0':<12} {'~0':<12} {'~0':<12}")

    return kmeans_results, sae_results


if __name__ == "__main__":
    main()
