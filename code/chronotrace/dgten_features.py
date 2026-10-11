"""DGTEN embedding extraction for ChronoTrace baseline comparison.

This module trains a DGTEN (Dynamic Graph Trust Evaluation Network) model
on the pre-deletion graph and extracts node embeddings to use as features
for the deleted neighbor inference task.

DGTEN uses a different data format and node ordering than ChronoTrace:
- DGTEN: 0-indexed nodes, space-separated rating.txt files
- ChronoTrace: Original node IDs, CSV files

This module handles the mapping between the two systems.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Put code/ (parent of the vendored mycode/ package) on sys.path for mycode imports
# code/chronotrace/dgten_features.py -> parents[1] == code/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .core import DATASETS, Edge, load_edges, nodes_of


def _build_node_mapping(dataset: str) -> Dict[int, int]:
    """Build mapping from DGTEN 0-indexed nodes to ChronoTrace node IDs.
    
    DGTEN uses 0-indexed nodes (0 to N-1), while ChronoTrace uses original
    node IDs from the CSV. The mapping is positional based on sorted order.
    """
    edges = load_edges(DATASETS[dataset])
    chrono_nodes = sorted(nodes_of(edges))
    return {i: node_id for i, node_id in enumerate(chrono_nodes)}


def _build_reverse_mapping(dataset: str) -> Dict[int, int]:
    """Build mapping from ChronoTrace node IDs to DGTEN 0-indexed nodes."""
    edges = load_edges(DATASETS[dataset])
    chrono_nodes = sorted(nodes_of(edges))
    return {node_id: i for i, node_id in enumerate(chrono_nodes)}


def train_dgten_embeddings(dataset: str, seed: int = 42, epochs: int = 50,
                          layers: Optional[List[int]] = None) -> Dict:
    """Train DGTEN and extract node embeddings.
    
    Args:
        dataset: Dataset name ('bitcoinotc', 'bitcoinalpha', 'epinions')
        seed: Random seed
        epochs: Training epochs
        layers: DGTEN layer dimensions (default: [32, 64, 32])
    
    Returns:
        dict with keys:
            - embeddings: (n_nodes, dim) numpy array
            - dgten_to_chrono: mapping from DGTEN index to ChronoTrace node ID
            - chrono_to_dgten: mapping from ChronoTrace node ID to DGTEN index
            - dim: embedding dimension
            - performance: DGTEN training performance metrics
    """
    from mycode.mainz_protocol import train_oracle_mainz_model
    
    # Train DGTEN
    trained = train_oracle_mainz_model(dataset, seed=seed, epochs=epochs, layers=layers)
    trainer = trained['trainer']
    model = trainer.model
    model.eval()
    
    # Extract embeddings
    import torch
    with torch.no_grad():
        _, temporal_out, _ = model(
            trainer.train_edges_final,
            trainer.y,
            trainer.train_labels_final,
        )
    
    embeddings = temporal_out.cpu().numpy()
    
    # Build node mappings
    dgten_to_chrono = _build_node_mapping(dataset)
    chrono_to_dgten = _build_reverse_mapping(dataset)
    
    return {
        'embeddings': embeddings,
        'dgten_to_chrono': dgten_to_chrono,
        'chrono_to_dgten': chrono_to_dgten,
        'dim': embeddings.shape[1],
        'performance': trained['performance'],
    }


def compute_dgten_features(embeddings: np.ndarray, chrono_to_dgten: Dict[int, int],
                          all_nodes: List[int]) -> np.ndarray:
    """Map DGTEN embedding vectors to ChronoTrace candidate nodes.
    
    Args:
        embeddings: (n_dgten_nodes, dim) array from train_dgten_embeddings
        chrono_to_dgten: mapping from ChronoTrace node ID to DGTEN index
        all_nodes: list of ChronoTrace candidate node IDs
    
    Returns:
        (len(all_nodes), dim) feature array
    """
    dim = embeddings.shape[1]
    features = np.zeros((len(all_nodes), dim))
    for i, node in enumerate(all_nodes):
        if node in chrono_to_dgten:
            dgten_idx = chrono_to_dgten[node]
            if dgten_idx < embeddings.shape[0]:
                features[i] = embeddings[dgten_idx]
    return features


def load_or_train_embeddings(dataset: str, seed: int = 42, epochs: int = 50,
                            layers: Optional[List[int]] = None,
                            cache_dir: Optional[Path] = None) -> Dict:
    """Load cached DGTEN embeddings or train new ones.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        epochs: Training epochs
        layers: DGTEN layer dimensions
        cache_dir: Directory to cache embeddings (default: results/dgten)
    
    Returns:
        dict with embeddings and mappings
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[2] / 'paper' / 'results' / 'dgten'
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{dataset}_seed{seed}_embeddings.npz'
    
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            'embeddings': data['embeddings'],
            'dgten_to_chrono': data['dgten_to_chrono'].item(),
            'chrono_to_dgten': data['chrono_to_dgten'].item(),
            'dim': data['embeddings'].shape[1],
            'performance': data['performance'].item(),
        }
    
    # Train new embeddings
    print(f'  Training DGTEN embeddings for {dataset} (seed={seed})...')
    result = train_dgten_embeddings(dataset, seed=seed, epochs=epochs, layers=layers)
    
    # Cache results
    np.savez(cache_path,
             embeddings=result['embeddings'],
             dgten_to_chrono=result['dgten_to_chrono'],
             chrono_to_dgten=result['chrono_to_dgten'],
             performance=result['performance'])
    
    print(f'  Cached DGTEN embeddings to {cache_path}')
    return result


def get_dgten_feature_names(dim: int) -> List[str]:
    """Return feature names for DGTEN embeddings."""
    return [f'dgten_{i}' for i in range(dim)]
