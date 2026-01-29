#!/usr/bin/env python
"""
Quick test script to verify the setup works.

Run without Hydra for simpler debugging.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data import generate_synthetic_data
from src.models import SubspaceModel
from src.losses import ReconstructionLoss
from src.metrics import evaluate_full_model
from src.utils import Trainer, TrainingConfig


def main():
    print("=" * 50)
    print("Quick Test: Multi-Cluster Subspace Discovery")
    print("=" * 50)

    # Small-scale test
    d = 32
    n_clusters_a = 20
    n_clusters_b = 20
    n_samples = 2000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Generate data
    print("\nGenerating synthetic data...")
    dataset = generate_synthetic_data(
        d=d,
        n_clusters_a=n_clusters_a,
        n_clusters_b=n_clusters_b,
        n_samples=n_samples,
        noise_std=0.1,
        apply_rotation=True,
        seed=42,
    )
    print(f"  Generated {n_samples} samples in R^{d}")
    print(f"  {n_clusters_a} A-clusters, {n_clusters_b} B-clusters")

    # Build model
    print("\nBuilding model...")
    model = SubspaceModel(
        d=d,
        n_clusters_a=n_clusters_a,
        n_clusters_b=n_clusters_b,
        clustering_algorithm="soft_kmeans",
        clustering_kwargs={"temperature": 1.0},
        init="random",
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = dataset.embeddings[:10].to(device)
    model = model.to(device)
    x_a, x_b, assign_a, assign_b = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Subspace A projection: {x_a.shape}")
    print(f"  Subspace B projection: {x_b.shape}")
    print(f"  Assignments A: {assign_a.shape}")
    print(f"  Assignments B: {assign_b.shape}")

    # Test reconstruction
    print("\nTesting reconstruction...")
    recon_a, recon_b, recon_full = model.get_reconstructions(x)
    print(f"  Reconstruction shape: {recon_full.shape}")

    # Test loss
    print("\nTesting loss computation...")
    loss_fn = ReconstructionLoss()
    loss, metrics = loss_fn(x, model)
    print(f"  Loss: {loss.item():.6f}")

    # Evaluate before training
    print("\nEvaluating before training...")
    model = model.cpu()
    metrics_before = evaluate_full_model(model, dataset, device="cpu")
    print(f"  Subspace alignment: {metrics_before['subspace_best_alignment']:.4f}")
    print(f"  Clustering accuracy: {metrics_before['clustering_accuracy_mean']:.4f}")
    print(f"  ARI: {metrics_before['ari_mean']:.4f}")

    # Quick training
    print("\nTraining for 20 epochs...")
    config = TrainingConfig(
        lr=1e-3,
        lr_subspace=1e-2,
        batch_size=256,
        n_epochs=20,
        log_interval=5,
        eval_interval=10,
        device=device,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        eval_fn=evaluate_full_model,
    )

    history = trainer.train(dataset)

    # Evaluate after training
    print("\nEvaluating after training...")
    metrics_after = trainer.evaluate(dataset)
    print(f"  Subspace alignment: {metrics_after['subspace_best_alignment']:.4f}")
    print(f"  Clustering accuracy: {metrics_after['clustering_accuracy_mean']:.4f}")
    print(f"  ARI: {metrics_after['ari_mean']:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Subspace alignment: {metrics_before['subspace_best_alignment']:.4f} -> {metrics_after['subspace_best_alignment']:.4f}")
    print(f"Clustering accuracy: {metrics_before['clustering_accuracy_mean']:.4f} -> {metrics_after['clustering_accuracy_mean']:.4f}")
    print(f"ARI: {metrics_before['ari_mean']:.4f} -> {metrics_after['ari_mean']:.4f}")

    print("\n✓ All tests passed!")

    return metrics_after


if __name__ == "__main__":
    main()
