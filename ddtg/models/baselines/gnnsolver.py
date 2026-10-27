"""GNNsolver baseline for full-mesh surface-current prediction.

The model keeps GraphSolver's edge-conditioned ``NNConv`` operator, but wraps
it in the same trainable scaffold used by FaceEGAT: pre-LayerNorm, residual
updates, LayerScale, an FFN, FiLM conditioning, and a tangential-current head.
It consumes the requested 14-D node and 5-D geometric edge features and never
uses Green-function features.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


def _read(data, name: str) -> torch.Tensor:
    if isinstance(data, Mapping):
        return data[name]
    return getattr(data, name)


class FiLM(nn.Module):
    """Identity-initialized feature-wise frequency/direction conditioning."""

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(condition).chunk(2, dim=-1)
        return (1.0 + gamma) * features + beta


class GraphSolverConv(nn.Module):
    """Pre-norm residual NNConv block with LayerScale and a residual FFN."""

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_width: int = 256,
        edge_dim: int = 5,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        layer_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        kernel = nn.Sequential(
            nn.Linear(edge_dim, kernel_width),
            nn.SiLU(),
            nn.Linear(kernel_width, kernel_width),
            nn.SiLU(),
            nn.Linear(kernel_width, kernel_width),
            nn.SiLU(),
            nn.Linear(kernel_width, hidden_dim * hidden_dim),
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.convolution = NNConv(
            hidden_dim,
            hidden_dim,
            kernel,
            aggr="mean",
            root_weight=False,
            bias=False,
        )
        self.message_scale = nn.Parameter(
            torch.full((hidden_dim,), float(layer_scale_init))
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )
        self.ffn_scale = nn.Parameter(
            torch.full((hidden_dim,), float(layer_scale_init))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        update = self.convolution(
            self.message_norm(node_features), edge_index, edge_features
        )
        node_features = node_features + self.dropout(update) * self.message_scale
        update = self.ffn(self.ffn_norm(node_features))
        return node_features + self.dropout(update) * self.ffn_scale


def _tangent_frames(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normals = F.normalize(normals.float(), dim=-1)
    reference = torch.zeros_like(normals)
    reference[:, 0] = 1.0
    near_x = normals[:, 0].abs() > 0.9
    reference[near_x, 0] = 0.0
    reference[near_x, 1] = 1.0
    tangent_1 = F.normalize(
        torch.linalg.cross(normals, reference, dim=-1), dim=-1
    )
    tangent_2 = torch.linalg.cross(normals, tangent_1, dim=-1)
    return tangent_1, tangent_2


class GNNSolver(nn.Module):
    """GraphSolver NNConv operator with stable residual full-mesh learning."""

    OUTPUT_LAYOUT = "block"
    OUTPUT_COMPONENTS = (
        "Jx_real", "Jy_real", "Jz_real",
        "Jx_imag", "Jy_imag", "Jz_imag",
    )

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 13,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_dim = 5
        self.node_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.encoder_film = FiLM(4, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                GraphSolverConv(
                    hidden_dim=hidden_dim,
                    kernel_width=256,
                    edge_dim=5,
                    ffn_mult=4,
                    dropout=dropout,
                    layer_scale_init=0.01,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_films = nn.ModuleList(
            [FiLM(4, hidden_dim) for _ in range(num_layers)]
        )
        decoder_hidden_1 = max(hidden_dim // 2, 16)
        decoder_hidden_2 = max(hidden_dim // 4, 8)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, decoder_hidden_1),
            nn.SiLU(),
            nn.Linear(decoder_hidden_1, decoder_hidden_2),
            nn.SiLU(),
            nn.Linear(decoder_hidden_2, 4),
        )
        nn.init.normal_(self.decoder[-1].weight, std=0.005)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer(
            "output_scale", torch.tensor(0.01, dtype=torch.float32)
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        node_features = _read(batch, "node_features").float()
        edge_index = _read(batch, "edge_index").long()
        edge_features = _read(batch, "edge_features").float()
        condition = _read(batch, "condition").float()
        normals = _read(batch, "normals").float()
        if node_features.ndim != 2 or node_features.shape[1] != 14:
            raise ValueError("node_features must have shape [N, 14]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_features.shape != (edge_index.shape[1], self.edge_dim):
            raise ValueError(f"edge_features must have shape [E, {self.edge_dim}]")

        hidden = self.encoder_film(self.node_encoder(node_features), condition)
        for layer, film in zip(self.graph_layers, self.layer_films):
            hidden = film(layer(hidden, edge_index, edge_features), condition)

        coefficients = self.decoder(hidden) * self.output_scale
        tangent_1, tangent_2 = _tangent_frames(normals)
        real_current = coefficients[:, 0:1] * tangent_1 + coefficients[:, 1:2] * tangent_2
        imaginary_current = coefficients[:, 2:3] * tangent_1 + coefficients[:, 3:4] * tangent_2
        return torch.cat([real_current, imaginary_current], dim=-1)

__all__ = ["GNNSolver"]
