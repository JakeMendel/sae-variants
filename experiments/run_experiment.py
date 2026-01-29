#!/usr/bin/env python
"""
Main experiment runner for multi-cluster subspace discovery.

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --config configs/default.yaml
    python experiments/run_experiment.py data.n_clusters_a=100 training.n_epochs=200
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.data import generate_synthetic_data
from src.models import SubspaceModel
from src.losses import get_loss_function, CombinedLoss
from src.metrics import evaluate_full_model
from src.utils import Trainer, TrainingConfig


def setup_device(device_config: str) -> str:
    """Determine device to use."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def build_model(cfg: DictConfig, dataset) -> SubspaceModel:
    """Build model from config."""
    n_clusters_a = cfg.data.n_clusters_a
    n_clusters_b = cfg.data.n_clusters_b

    if not cfg.model.known_n_clusters:
        # Use auto discovery with max clusters
        n_clusters_a = cfg.model.max_clusters
        n_clusters_b = cfg.model.max_clusters
        clustering_algorithm = "auto_discovery"
    else:
        clustering_algorithm = cfg.model.clustering_algorithm

    model = SubspaceModel(
        d=cfg.data.d,
        n_clusters_a=n_clusters_a,
        n_clusters_b=n_clusters_b,
        clustering_algorithm=clustering_algorithm,
        clustering_kwargs=OmegaConf.to_container(cfg.model.clustering_kwargs),
        init=cfg.model.subspace_init,
    )

    return model


def build_loss(cfg: DictConfig):
    """Build loss function from config."""
    losses = {}

    # Primary loss
    primary_loss = get_loss_function(cfg.loss.primary)
    losses["primary"] = (primary_loss, 1.0)

    # Regularization losses
    if cfg.loss.regularization.entropy.enabled:
        entropy_loss = get_loss_function(
            "entropy",
            weight=cfg.loss.regularization.entropy.weight,
            target=cfg.loss.regularization.entropy.target,
        )
        losses["entropy"] = (entropy_loss, 1.0)

    if len(losses) == 1:
        return primary_loss

    return CombinedLoss(losses)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main experiment entry point."""
    print("=" * 60)
    print("Multi-Cluster Subspace Discovery Experiment")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    torch.manual_seed(cfg.experiment.seed)

    # Setup device
    device = setup_device(cfg.experiment.device)
    print(f"\nUsing device: {device}")

    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg),
        )

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    dataset = generate_synthetic_data(
        d=cfg.data.d,
        n_clusters_a=cfg.data.n_clusters_a,
        n_clusters_b=cfg.data.n_clusters_b,
        n_samples=cfg.data.n_samples,
        noise_std=cfg.data.noise_std,
        apply_rotation=cfg.data.apply_rotation,
        seed=cfg.data.seed,
    )
    print(f"  - Dimension: {cfg.data.d}")
    print(f"  - Clusters A: {cfg.data.n_clusters_a}, Clusters B: {cfg.data.n_clusters_b}")
    print(f"  - Samples: {cfg.data.n_samples}")
    print(f"  - Noise std: {cfg.data.noise_std}")
    print(f"  - Rotation applied: {cfg.data.apply_rotation}")

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg, dataset)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {n_params:,}")
    print(f"  - Clustering algorithm: {cfg.model.clustering_algorithm}")

    # Build loss
    print("\nBuilding loss function...")
    loss_fn = build_loss(cfg)
    print(f"  - Primary loss: {cfg.loss.primary}")

    # Build trainer
    training_config = TrainingConfig(
        lr=cfg.training.lr,
        lr_subspace=cfg.training.lr_subspace,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        scheduler=cfg.training.scheduler,
        warmup_epochs=cfg.training.warmup_epochs,
        log_interval=cfg.training.log_interval,
        eval_interval=cfg.training.eval_interval,
        device=device,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=training_config,
        eval_fn=evaluate_full_model,
    )

    # Train
    print("\nStarting training...")
    print("-" * 40)
    history = trainer.train(dataset, eval_dataset=dataset)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_metrics = trainer.evaluate(dataset)

    print(f"\nSubspace Recovery:")
    print(f"  - Alignment (A): {final_metrics['subspace_alignment_a']:.4f}")
    print(f"  - Alignment (B): {final_metrics['subspace_alignment_b']:.4f}")
    print(f"  - Best alignment: {final_metrics['subspace_best_alignment']:.4f}")
    print(f"  - Subspaces swapped: {final_metrics['subspaces_swapped']}")

    print(f"\nClustering Performance:")
    print(f"  - Accuracy (A): {final_metrics['clustering_accuracy_a']:.4f}")
    print(f"  - Accuracy (B): {final_metrics['clustering_accuracy_b']:.4f}")
    print(f"  - Mean accuracy: {final_metrics['clustering_accuracy_mean']:.4f}")
    print(f"  - ARI (A): {final_metrics['ari_a']:.4f}")
    print(f"  - ARI (B): {final_metrics['ari_b']:.4f}")
    print(f"  - Mean ARI: {final_metrics['ari_mean']:.4f}")
    print(f"  - NMI (A): {final_metrics['nmi_a']:.4f}")
    print(f"  - NMI (B): {final_metrics['nmi_b']:.4f}")
    print(f"  - Mean NMI: {final_metrics['nmi_mean']:.4f}")

    if cfg.wandb.enabled:
        wandb.log({"final": final_metrics})
        wandb.finish()

    return final_metrics


if __name__ == "__main__":
    main()
