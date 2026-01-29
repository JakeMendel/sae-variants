"""
Training utilities for subspace discovery experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from tqdm import tqdm
import geoopt


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    lr: float = 1e-3
    lr_subspace: float = 1e-2  # Separate LR for Stiefel manifold
    weight_decay: float = 0.0
    batch_size: int = 256
    n_epochs: int = 100

    # Learning rate schedule
    scheduler: str = "cosine"  # "cosine", "step", "none"
    warmup_epochs: int = 5

    # Logging
    log_interval: int = 10
    eval_interval: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Trainer for subspace discovery model.

    Handles:
    - Stiefel manifold optimization (via geoopt)
    - Standard parameter optimization
    - Logging and evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        eval_fn: Optional[Callable] = None,
    ):
        self.model = model.to(config.device)
        self.loss_fn = loss_fn
        self.config = config
        self.eval_fn = eval_fn

        # Separate parameters: Stiefel manifold vs Euclidean
        stiefel_params = []
        euclidean_params = []

        for name, param in model.named_parameters():
            if isinstance(param, geoopt.ManifoldParameter):
                stiefel_params.append(param)
            else:
                euclidean_params.append(param)

        # Use geoopt's RiemannianAdam for Stiefel parameters
        self.optimizer_stiefel = geoopt.optim.RiemannianAdam(
            stiefel_params,
            lr=config.lr_subspace,
        ) if stiefel_params else None

        # Standard Adam for other parameters
        self.optimizer_euclidean = torch.optim.Adam(
            euclidean_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        ) if euclidean_params else None

        # Learning rate schedulers
        self.schedulers = []
        if config.scheduler == "cosine":
            if self.optimizer_stiefel:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer_stiefel, T_max=config.n_epochs
                    )
                )
            if self.optimizer_euclidean:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer_euclidean, T_max=config.n_epochs
                    )
                )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_metrics": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics = {}
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.config.device)
            else:
                x = batch.to(self.config.device)

            # Forward pass
            loss, metrics = self.loss_fn(x, self.model)

            # Backward pass
            if self.optimizer_stiefel:
                self.optimizer_stiefel.zero_grad()
            if self.optimizer_euclidean:
                self.optimizer_euclidean.zero_grad()

            loss.backward()

            if self.optimizer_stiefel:
                self.optimizer_stiefel.step()
            if self.optimizer_euclidean:
                self.optimizer_euclidean.step()

            # Accumulate metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v.item() if torch.is_tensor(v) else v
            n_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in all_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_metrics

    def evaluate(self, dataset) -> Dict[str, float]:
        """Evaluate model on dataset."""
        if self.eval_fn is None:
            return {}

        self.model.eval()
        with torch.no_grad():
            metrics = self.eval_fn(self.model, dataset, device=self.config.device)

        return metrics

    def train(
        self,
        dataset,
        eval_dataset=None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            dataset: Training dataset
            eval_dataset: Dataset for evaluation (can be same as train)

        Returns:
            Training history
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        eval_dataset = eval_dataset or dataset

        for epoch in range(1, self.config.n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(dataloader, epoch)
            self.history["train_loss"].append(train_metrics["loss"])

            # Update learning rate
            for scheduler in self.schedulers:
                scheduler.step()

            # Log
            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch}: loss = {train_metrics['loss']:.6f}")

            # Evaluate
            if epoch % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(eval_dataset)
                self.history["eval_metrics"].append(eval_metrics)

                if eval_metrics:
                    print(f"  Subspace alignment: {eval_metrics.get('subspace_best_alignment', 'N/A'):.4f}")
                    print(f"  Clustering accuracy: {eval_metrics.get('clustering_accuracy_mean', 'N/A'):.4f}")
                    print(f"  ARI: {eval_metrics.get('ari_mean', 'N/A'):.4f}")

        return self.history
