#!/usr/bin/env python3
"""
NDCG and friends — a deliberate duplicate of the metrics in
`../03_learning_to_rank/ranking_losses.py`.

Copied rather than imported, on purpose. This repository's rule is that a
reader can open ONE folder and run it without the other twenty-three existing
(see CLAUDE.md), and a cross-folder import would break that for the sake of
forty lines. The originals carry the full commentary on why the gain is
exponential and why an empty query scores 0; this copy keeps the short version.

Pure PyTorch. No GPU, no download.
"""

import torch


def dcg(relevance: torch.Tensor, k: int = 10) -> torch.Tensor:
    """DCG of a list ALREADY IN RANKED ORDER. Does not sort."""
    relevance = relevance[..., :k]
    gains = torch.pow(2.0, relevance) - 1.0          # exponential, not linear
    positions = torch.arange(relevance.shape[-1], device=relevance.device,
                             dtype=torch.float32)
    return (gains / torch.log2(positions + 2.0)).sum(dim=-1)


def ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """NDCG@k. A query with nothing relevant scores 0.0, not NaN and not 1.0."""
    order = scores.argsort(dim=-1, descending=True)
    actual = dcg(labels.gather(-1, order), k)
    ideal = dcg(labels.sort(dim=-1, descending=True).values, k)
    return torch.where(ideal > 0, actual / ideal, torch.zeros_like(actual))


def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    The listwise objective, held FIXED across the models in this folder.

    03_learning_to_rank varies the loss and holds the architecture fixed; this
    folder does the opposite. Using one loss for all three scorers is what
    makes the comparison here about context rather than about objectives.
    """
    target = torch.softmax(labels.float(), dim=-1)
    return -(target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


if __name__ == "__main__":
    labels = torch.tensor([[3.0, 2.0, 3.0, 0.0]])
    print(f"  NDCG, perfect order : {ndcg(labels, labels).item():.4f}")
    print(f"  NDCG, reversed      : {ndcg(-labels, labels).item():.4f}")
    print(f"  NDCG, nothing relevant: "
          f"{ndcg(torch.randn(1, 4), torch.zeros(1, 4)).item():.4f}")
