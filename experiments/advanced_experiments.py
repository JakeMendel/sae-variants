#!/usr/bin/env python
"""
Advanced experiments:
1. Alternating optimization (fix subspace -> optimize clusters -> fix clusters -> optimize subspace)
2. Contrastive/InfoNCE loss
3. Zero noise case
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from src.data import generate_synthetic_data
from src.models import SubspaceModel
from src.losses import get_loss_function, CombinedLoss
from src.metrics import evaluate_full_model
from src.utils import get_best_device
import geoopt


def get_device():
    return get_best_device()


# ============================================================
# CONTRASTIVE / INFONCE LOSS
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss that encourages:
    - Points with same cluster assignment to be close
    - Points with different cluster assignments to be far

    Uses InfoNCE-style formulation within each subspace.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, model: nn.Module):
        x_a, x_b, assignments_a, assignments_b = model.forward(x)

        # Get hard assignments for contrastive pairs
        hard_a = assignments_a.argmax(dim=1)
        hard_b = assignments_b.argmax(dim=1)

        loss_a = self._infonce_loss(x_a, hard_a)
        loss_b = self._infonce_loss(x_b, hard_b)

        loss = loss_a + loss_b

        return loss, {
            'contrastive_a': loss_a.detach(),
            'contrastive_b': loss_b.detach(),
        }

    def _infonce_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss: pull together same-class, push apart different-class."""
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = embeddings @ embeddings.T / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()

        # Remove diagonal (self-similarity)
        pos_mask.fill_diagonal_(0)

        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Compute log-sum-exp over all pairs
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)  # Exclude self

        # Sum of positive similarities
        pos_sim = (exp_sim * pos_mask).sum(dim=1)

        # Sum of all similarities (denominator)
        all_sim = exp_sim.sum(dim=1)

        # Avoid log(0)
        loss = -torch.log((pos_sim + 1e-10) / (all_sim + 1e-10))

        # Only count samples that have positive pairs
        valid_mask = pos_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss


class ContrastiveClusterLoss(nn.Module):
    """
    Softer contrastive loss using soft cluster assignments as similarity weights.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, model: nn.Module):
        x_a, x_b, assignments_a, assignments_b = model.forward(x)

        loss_a = self._soft_contrastive(x_a, assignments_a)
        loss_b = self._soft_contrastive(x_b, assignments_b)

        loss = loss_a + loss_b

        return loss, {
            'soft_contrastive_a': loss_a.detach(),
            'soft_contrastive_b': loss_b.detach(),
        }

    def _soft_contrastive(self, embeddings: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        """Soft contrastive using assignment similarity as target."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Embedding similarity
        embed_sim = embeddings @ embeddings.T / self.temperature

        # Assignment similarity (how likely two points are in same cluster)
        assign_sim = assignments @ assignments.T  # (batch, batch)

        # We want embedding similarity to match assignment similarity
        # Use KL divergence or MSE
        embed_probs = F.softmax(embed_sim, dim=1)
        assign_probs = F.softmax(assign_sim / self.temperature, dim=1)

        # KL divergence: want embed_probs to match assign_probs
        loss = F.kl_div(
            torch.log(embed_probs + 1e-10),
            assign_probs,
            reduction='batchmean'
        )

        return loss


# ============================================================
# ALTERNATING OPTIMIZATION
# ============================================================

class AlternatingOptimizer:
    """
    Alternating optimization:
    1. Fix subspace, optimize cluster centroids
    2. Fix cluster centroids, optimize subspace
    """

    def __init__(
        self,
        model: SubspaceModel,
        loss_fn: nn.Module,
        device: str = 'cpu',
        lr_clusters: float = 1e-2,
        lr_subspace: float = 1e-2,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

        # Separate parameters
        self.subspace_params = [model.subspace_transform.Q]
        self.cluster_params = list(model.clusterer_a.parameters()) + list(model.clusterer_b.parameters())

        # Separate optimizers
        self.opt_subspace = geoopt.optim.RiemannianAdam(self.subspace_params, lr=lr_subspace)
        self.opt_clusters = torch.optim.Adam(self.cluster_params, lr=lr_clusters)

    def train_epoch_clusters(self, dataloader: DataLoader, n_steps: int = None):
        """Train only cluster parameters."""
        self.model.train()

        # Freeze subspace
        for p in self.subspace_params:
            p.requires_grad = False
        for p in self.cluster_params:
            p.requires_grad = True

        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            if n_steps and n_batches >= n_steps:
                break

            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            loss, _ = self.loss_fn(x, self.model)

            self.opt_clusters.zero_grad()
            loss.backward()
            self.opt_clusters.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch_subspace(self, dataloader: DataLoader, n_steps: int = None):
        """Train only subspace parameters."""
        self.model.train()

        # Freeze clusters
        for p in self.subspace_params:
            p.requires_grad = True
        for p in self.cluster_params:
            p.requires_grad = False

        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            if n_steps and n_batches >= n_steps:
                break

            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            loss, _ = self.loss_fn(x, self.model)

            self.opt_subspace.zero_grad()
            loss.backward()
            self.opt_subspace.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        dataset,
        n_outer_iters: int = 50,
        cluster_steps_per_iter: int = 20,
        subspace_steps_per_iter: int = 10,
        batch_size: int = 512,
        eval_fn=None,
        eval_interval: int = 10,
    ):
        """Full alternating training loop."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {'cluster_loss': [], 'subspace_loss': [], 'metrics': []}

        for outer_iter in range(1, n_outer_iters + 1):
            # Phase 1: Optimize clusters (many steps)
            cluster_loss = 0
            for _ in range(cluster_steps_per_iter):
                cluster_loss = self.train_epoch_clusters(dataloader, n_steps=len(dataloader))

            # Phase 2: Optimize subspace (fewer steps)
            subspace_loss = 0
            for _ in range(subspace_steps_per_iter):
                subspace_loss = self.train_epoch_subspace(dataloader, n_steps=len(dataloader))

            history['cluster_loss'].append(cluster_loss)
            history['subspace_loss'].append(subspace_loss)

            if outer_iter % eval_interval == 0:
                if eval_fn:
                    self.model.eval()
                    with torch.no_grad():
                        # Move to CPU for eval
                        model_cpu = self.model.cpu()
                        metrics = eval_fn(model_cpu, dataset, device='cpu')
                        self.model.to(self.device)
                    history['metrics'].append(metrics)
                    print(f"Iter {outer_iter}: cluster_loss={cluster_loss:.4f}, subspace_loss={subspace_loss:.4f}, "
                          f"subspace_align={metrics['subspace_best_alignment']:.4f}, "
                          f"acc={metrics['clustering_accuracy_mean']:.4f}, ari={metrics['ari_mean']:.4f}")
                else:
                    print(f"Iter {outer_iter}: cluster_loss={cluster_loss:.4f}, subspace_loss={subspace_loss:.4f}")

        return history


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

def run_experiment(name, dataset, model_fn, train_fn, device='cpu'):
    """Run a single experiment and return metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print('='*60)

    model = model_fn()

    # Before metrics
    model_cpu = model.cpu()
    m_before = evaluate_full_model(model_cpu, dataset, device='cpu')
    print(f"Before: subspace={m_before['subspace_best_alignment']:.4f}, "
          f"acc={m_before['clustering_accuracy_mean']:.4f}, ari={m_before['ari_mean']:.4f}")

    # Train
    model = model.to(device)
    train_fn(model, dataset, device)

    # After metrics
    model_cpu = model.cpu()
    m_after = evaluate_full_model(model_cpu, dataset, device='cpu')
    print(f"After:  subspace={m_after['subspace_best_alignment']:.4f}, "
          f"acc={m_after['clustering_accuracy_mean']:.4f}, ari={m_after['ari_mean']:.4f}")

    return {
        'name': name,
        'before': m_before,
        'after': m_after,
    }


def main():
    device = get_device()
    print(f"Using device: {device}")

    d = 64
    n_clusters = 50
    n_samples = 10000

    results = []

    # ============================================================
    # EXPERIMENT SET 1: ZERO NOISE
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT SET 1: ZERO NOISE")
    print("="*70)

    dataset_zero_noise = generate_synthetic_data(
        d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
        n_samples=n_samples, noise_std=0.0, seed=42  # ZERO NOISE
    )

    # 1a. Standard training with compactness (zero noise)
    def make_model():
        torch.manual_seed(42)
        return SubspaceModel(d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
                            clustering_algorithm='soft_kmeans',
                            clustering_kwargs={'temperature': 0.1}, init='random')

    def train_standard(model, dataset, device):
        from src.utils import Trainer, TrainingConfig
        loss_fn = get_loss_function('compactness')
        trainer = Trainer(model=model, loss_fn=loss_fn,
                         config=TrainingConfig(lr=1e-2, lr_subspace=5e-2, batch_size=512,
                                              n_epochs=150, log_interval=200, eval_interval=200, device=device),
                         eval_fn=None)
        trainer.train(dataset)

    r = run_experiment("Zero noise + Compactness", dataset_zero_noise, make_model, train_standard, device)
    results.append(r)

    # 1b. Alternating optimization (zero noise)
    def train_alternating(model, dataset, device):
        loss_fn = get_loss_function('compactness')
        alt_opt = AlternatingOptimizer(model, loss_fn, device=device, lr_clusters=1e-2, lr_subspace=5e-2)
        alt_opt.train(dataset, n_outer_iters=30, cluster_steps_per_iter=10, subspace_steps_per_iter=5,
                     eval_fn=evaluate_full_model, eval_interval=10)

    r = run_experiment("Zero noise + Alternating", dataset_zero_noise, make_model, train_alternating, device)
    results.append(r)

    # 1c. Contrastive loss (zero noise)
    def train_contrastive(model, dataset, device):
        from src.utils import Trainer, TrainingConfig
        loss_fn = ContrastiveLoss(temperature=0.1)
        trainer = Trainer(model=model, loss_fn=loss_fn,
                         config=TrainingConfig(lr=1e-2, lr_subspace=5e-2, batch_size=512,
                                              n_epochs=150, log_interval=200, eval_interval=200, device=device),
                         eval_fn=None)
        trainer.train(dataset)

    r = run_experiment("Zero noise + Contrastive", dataset_zero_noise, make_model, train_contrastive, device)
    results.append(r)

    # ============================================================
    # EXPERIMENT SET 2: WITH NOISE (standard setting)
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT SET 2: WITH NOISE (noise_std=0.1)")
    print("="*70)

    dataset_noisy = generate_synthetic_data(
        d=d, n_clusters_a=n_clusters, n_clusters_b=n_clusters,
        n_samples=n_samples, noise_std=0.1, seed=42
    )

    # 2a. Alternating optimization (noisy)
    r = run_experiment("Noisy + Alternating", dataset_noisy, make_model, train_alternating, device)
    results.append(r)

    # 2b. Contrastive loss (noisy)
    r = run_experiment("Noisy + Contrastive", dataset_noisy, make_model, train_contrastive, device)
    results.append(r)

    # 2c. Combined: Compactness + Contrastive (noisy)
    def train_compact_contrastive(model, dataset, device):
        from src.utils import Trainer, TrainingConfig
        loss_fn = CombinedLoss({
            'compact': (get_loss_function('compactness'), 1.0),
            'contrastive': (ContrastiveLoss(temperature=0.1), 0.5),
        })
        trainer = Trainer(model=model, loss_fn=loss_fn,
                         config=TrainingConfig(lr=1e-2, lr_subspace=5e-2, batch_size=512,
                                              n_epochs=150, log_interval=200, eval_interval=200, device=device),
                         eval_fn=None)
        trainer.train(dataset)

    r = run_experiment("Noisy + Compact+Contrastive", dataset_noisy, make_model, train_compact_contrastive, device)
    results.append(r)

    # 2d. Alternating with contrastive loss
    def train_alternating_contrastive(model, dataset, device):
        loss_fn = CombinedLoss({
            'compact': (get_loss_function('compactness'), 1.0),
            'contrastive': (ContrastiveLoss(temperature=0.1), 0.3),
        })
        alt_opt = AlternatingOptimizer(model, loss_fn, device=device, lr_clusters=1e-2, lr_subspace=5e-2)
        alt_opt.train(dataset, n_outer_iters=30, cluster_steps_per_iter=10, subspace_steps_per_iter=5,
                     eval_fn=evaluate_full_model, eval_interval=10)

    r = run_experiment("Noisy + Alt+Compact+Contrastive", dataset_noisy, make_model, train_alternating_contrastive, device)
    results.append(r)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Experiment':<40} {'Subspace':>10} {'Accuracy':>10} {'ARI':>10}")
    print("-"*70)
    for r in results:
        m = r['after']
        print(f"{r['name']:<40} {m['subspace_best_alignment']:>10.4f} {m['clustering_accuracy_mean']:>10.4f} {m['ari_mean']:>10.4f}")

    print("\nBaselines for reference:")
    print(f"{'Oracle (true subspace + k-means)':<40} {'1.0000':>10} {'0.9613':>10} {'0.9650':>10}")
    print(f"{'Previous best (compact t=0.1, noisy)':<40} {'0.5010':>10} {'0.7500':>10} {'0.7630':>10}")
    print(f"{'Random':<40} {'0.5000':>10} {'0.0200':>10} {'0.0000':>10}")


if __name__ == "__main__":
    main()
