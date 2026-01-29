"""
Evaluation metrics for comparing learned subspaces and clusters to ground truth.
"""

import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Optional


def subspace_alignment_score(
    learned_basis: torch.Tensor,
    true_basis: torch.Tensor,
) -> float:
    """
    Compute how well the learned subspace aligns with the true subspace.

    Uses principal angles between subspaces. Perfect alignment = 1.0.

    Args:
        learned_basis: (d, k) orthonormal basis for learned subspace
        true_basis: (d, k) orthonormal basis for true subspace

    Returns:
        Alignment score in [0, 1]
    """
    # Ensure we're working with numpy
    if torch.is_tensor(learned_basis):
        learned_basis = learned_basis.detach().cpu().numpy()
    if torch.is_tensor(true_basis):
        true_basis = true_basis.detach().cpu().numpy()

    # Compute SVD of the product of bases
    # The singular values are cosines of principal angles
    M = learned_basis.T @ true_basis
    _, sigmas, _ = np.linalg.svd(M)

    # Clamp to handle numerical issues
    sigmas = np.clip(sigmas, 0, 1)

    # Average of squared cosines (related to Grassmann distance)
    alignment = np.mean(sigmas ** 2)

    return float(alignment)


def clustering_accuracy(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> Tuple[float, np.ndarray]:
    """
    Compute clustering accuracy using optimal Hungarian matching.

    Args:
        pred_labels: (n,) predicted cluster assignments
        true_labels: (n,) ground truth labels

    Returns:
        accuracy: proportion of correctly assigned points
        mapping: optimal mapping from predicted to true clusters
    """
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()

    pred_labels = pred_labels.astype(np.int64)
    true_labels = true_labels.astype(np.int64)

    n_pred_clusters = pred_labels.max() + 1
    n_true_clusters = true_labels.max() + 1

    # Build confusion matrix
    n = max(n_pred_clusters, n_true_clusters)
    confusion = np.zeros((n, n), dtype=np.int64)

    for pred, true in zip(pred_labels, true_labels):
        confusion[pred, true] += 1

    # Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Compute accuracy
    correct = confusion[row_ind, col_ind].sum()
    accuracy = correct / len(pred_labels)

    # Build mapping
    mapping = np.zeros(n_pred_clusters, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        if r < n_pred_clusters:
            mapping[r] = c

    return float(accuracy), mapping


def adjusted_rand_index(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> float:
    """
    Compute Adjusted Rand Index between predicted and true clusterings.

    ARI = 1.0 indicates perfect clustering, 0.0 indicates random.
    """
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()

    return float(adjusted_rand_score(true_labels, pred_labels))


def normalized_mutual_info(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> float:
    """
    Compute Normalized Mutual Information between clusterings.

    NMI = 1.0 indicates perfect clustering, 0.0 indicates independent.
    """
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()

    return float(normalized_mutual_info_score(true_labels, pred_labels))


def evaluate_full_model(
    model,
    dataset,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the model against ground truth.

    Args:
        model: SubspaceModel
        dataset: SyntheticMultiClusterDataset with ground truth

    Returns:
        Dict of evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        embeddings = dataset.embeddings.to(device)
        true_labels_a = dataset.labels_a
        true_labels_b = dataset.labels_b

        # Get model predictions
        hard_a, hard_b = model.get_hard_assignments(embeddings)
        hard_a = hard_a.cpu()
        hard_b = hard_b.cpu()

        # Get learned subspace bases
        learned_basis_a, learned_basis_b = model.subspace_transform.get_subspace_bases()

        # Get ground truth subspace bases
        true_basis_a = dataset.true_subspace_a
        true_basis_b = dataset.true_subspace_b

    metrics = {}

    # Subspace alignment
    metrics["subspace_alignment_a"] = subspace_alignment_score(learned_basis_a, true_basis_a)
    metrics["subspace_alignment_b"] = subspace_alignment_score(learned_basis_b, true_basis_b)
    metrics["subspace_alignment_mean"] = (
        metrics["subspace_alignment_a"] + metrics["subspace_alignment_b"]
    ) / 2

    # Also check if subspaces are swapped (A learned as B, B learned as A)
    swap_align_a = subspace_alignment_score(learned_basis_a, true_basis_b)
    swap_align_b = subspace_alignment_score(learned_basis_b, true_basis_a)
    metrics["subspace_alignment_swapped"] = (swap_align_a + swap_align_b) / 2

    # Use best alignment (accounting for potential swap)
    if metrics["subspace_alignment_swapped"] > metrics["subspace_alignment_mean"]:
        metrics["subspace_best_alignment"] = metrics["subspace_alignment_swapped"]
        metrics["subspaces_swapped"] = True
        # Swap labels for clustering metrics
        hard_a, hard_b = hard_b, hard_a
    else:
        metrics["subspace_best_alignment"] = metrics["subspace_alignment_mean"]
        metrics["subspaces_swapped"] = False

    # Clustering metrics for subspace A
    metrics["clustering_accuracy_a"], _ = clustering_accuracy(hard_a, true_labels_a)
    metrics["ari_a"] = adjusted_rand_index(hard_a, true_labels_a)
    metrics["nmi_a"] = normalized_mutual_info(hard_a, true_labels_a)

    # Clustering metrics for subspace B
    metrics["clustering_accuracy_b"], _ = clustering_accuracy(hard_b, true_labels_b)
    metrics["ari_b"] = adjusted_rand_index(hard_b, true_labels_b)
    metrics["nmi_b"] = normalized_mutual_info(hard_b, true_labels_b)

    # Averages
    metrics["clustering_accuracy_mean"] = (
        metrics["clustering_accuracy_a"] + metrics["clustering_accuracy_b"]
    ) / 2
    metrics["ari_mean"] = (metrics["ari_a"] + metrics["ari_b"]) / 2
    metrics["nmi_mean"] = (metrics["nmi_a"] + metrics["nmi_b"]) / 2

    return metrics
