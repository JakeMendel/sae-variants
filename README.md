# SAE Variants: Multi-Cluster Subspace Discovery

This repository implements experiments for discovering orthogonal subspaces in high-dimensional data where each data point belongs to multiple cluster types simultaneously.

## Problem Setup

Each data point `x ∈ R^d` belongs to:
- Exactly one cluster of type A (from k_A possible clusters)
- Exactly one cluster of type B (from k_B possible clusters)

The embedding structure:
```
x = centroid_A + centroid_B + noise
```

Where:
- `centroid_A ∈ R^(d/2)` lives in subspace 1 (first d/2 dimensions)
- `centroid_B ∈ R^(d/2)` lives in subspace 2 (last d/2 dimensions)
- The two subspaces are orthogonal

After applying a random rotation, the algorithm must discover:
1. The orthogonal subspace decomposition
2. The cluster assignments in each subspace

## Approach

The model learns:
1. **Orthogonal transformation** `Q ∈ O(d)` on the Stiefel manifold
2. **Cluster centroids** in each subspace via differentiable clustering

Training uses reconstruction loss with backpropagation through the clustering.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

```bash
# Quick test to verify setup
python experiments/quick_test.py

# Full experiment with default config
python experiments/run_experiment.py

# Override config parameters
python experiments/run_experiment.py \
    data.d=64 \
    data.n_clusters_a=50 \
    data.n_clusters_b=50 \
    training.n_epochs=200
```

## Configuration

See `configs/default.yaml` for all options. Key parameters:

### Data
- `d`: Embedding dimension (default: 64)
- `n_clusters_a/b`: Number of clusters per type (default: 50, can exceed d/2)
- `noise_std`: Gaussian noise level (default: 0.1)

### Model
- `clustering_algorithm`: `"soft_kmeans"`, `"gmm"`, `"auto_discovery"`
- `known_n_clusters`: Whether cluster count is known (default: true)

### Loss
- `primary`: `"reconstruction"`, `"subspace_reconstruction"`, `"compactness"`
- Entropy regularization for confident/soft assignments

## Project Structure

```
sae-variants/
├── configs/
│   └── default.yaml          # Hydra configuration
├── experiments/
│   ├── run_experiment.py     # Main experiment runner
│   └── quick_test.py         # Quick verification
├── src/
│   ├── data/
│   │   └── synthetic.py      # Synthetic data generation
│   ├── models/
│   │   ├── subspace.py       # Stiefel manifold subspace model
│   │   └── clustering.py     # Differentiable clustering algorithms
│   ├── losses/
│   │   └── reconstruction.py # Loss functions
│   ├── metrics/
│   │   └── evaluation.py     # Evaluation against ground truth
│   └── utils/
│       └── training.py       # Training loop with Riemannian optimization
├── requirements.txt
└── pyproject.toml
```

## Key Components

### Clustering Algorithms (`src/models/clustering.py`)

1. **SoftKMeans**: Soft assignments via softmax over negative distances
2. **DifferentiableGMM**: Gaussian mixture with learnable means/covariances
3. **AutoClusterDiscovery**: Discovers number of clusters via sparsity

### Subspace Model (`src/models/subspace.py`)

- Uses `geoopt` for Stiefel manifold optimization
- Learns orthogonal transformation that separates the two subspaces
- Separate clustering in each discovered subspace

### Losses (`src/losses/reconstruction.py`)

- **ReconstructionLoss**: `||x - reconstruct(x)||^2`
- **SubspaceReconstructionLoss**: Per-subspace reconstruction
- **ClusterCompactnessLoss**: Within-cluster variance
- **EntropyRegularization**: Assignment confidence

### Metrics (`src/metrics/evaluation.py`)

- **Subspace alignment**: Principal angle-based alignment score
- **Clustering accuracy**: Hungarian-matched accuracy
- **ARI/NMI**: Standard clustering metrics

## Extending

### Add a new clustering algorithm

1. Subclass `ClusteringAlgorithm` in `src/models/clustering.py`
2. Implement `forward()` (soft assignments) and `get_centroids()`
3. Register in `get_clustering_algorithm()`

### Add a new loss function

1. Subclass `LossFunction` in `src/losses/reconstruction.py`
2. Implement `forward()` returning `(loss, metrics_dict)`
3. Register in `get_loss_function()`

## Citation

If you use this code, please cite the relevant papers on:
- Sparse autoencoders
- Subspace clustering
- Stiefel manifold optimization
