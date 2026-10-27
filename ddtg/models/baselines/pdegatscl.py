"""PdEgatSCL network for complex PEC surface-current prediction.

The model operates only on the triangle-face graph.  It does not consume
patches, face-to-patch assignments, patch tokens, or domain-decomposition
features.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def _read(data, name: str) -> torch.Tensor:
    if isinstance(data, Mapping):
        return data[name]
    if hasattr(data, name):
        return getattr(data, name)
    legacy_names = {
        "node_features": "pdegat_node_feat",
        "edge_index": "pdegat_edge_index",
        "edge_features": "pdegat_edge_attr",
    }
    return getattr(data, legacy_names[name])


def _encoder_mlp(in_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    dims = [in_dim, *hidden_dims]
    layers: list[nn.Module] = []
    for in_width, out_width in zip(dims[:-1], dims[1:]):
        layers.extend((nn.Linear(in_width, out_width), nn.ReLU()))
    return nn.Sequential(*layers)


def _decoder_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for in_width, out_width in zip(dims[:-2], dims[1:-1]):
        layers.extend((nn.Linear(in_width, out_width), nn.ReLU()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    network = nn.Sequential(*layers)
    nn.init.normal_(network[-1].weight, std=0.005)
    nn.init.zeros_(network[-1].bias)
    return network


def _compute_tangent_frames(
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the same deterministic orthonormal face frames used by DDGT."""
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


class FiLM(nn.Module):
    """Identity-initialized feature-wise conditioning.

    The zero initialization makes the initial transform exactly ``h``.  This
    lets frequency/direction conditioning be learned without perturbing the
    unconditioned PdEgatSCL initialization.
    """

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.projection(condition).chunk(2, dim=-1)
        return (1.0 + gamma) * features + beta


class EGATConv(MessagePassing):
    """Edge-featured graph attention layer from PdEgatSCL Algorithm 1.

    ``stabilized=True`` wraps the paper update in the same pre-normalisation
    and residual pattern used by the project's trainable Patch-EGAT.  A
    learned residual projection is used for the first 512->256 layer; later
    256->256 layers use identity residuals.  There is deliberately no
    post-LayerNorm: the next layer performs pre-normalisation, while the raw
    residual stream preserves node-specific amplitude information.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        stabilized: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if out_dim % heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.stabilized = bool(stabilized)
        self.node_input_norm = nn.LayerNorm(in_dim) if stabilized else nn.Identity()
        self.edge_input_norm = nn.LayerNorm(in_dim) if stabilized else nn.Identity()
        if stabilized and in_dim != out_dim:
            self.node_residual = nn.Linear(in_dim, out_dim, bias=False)
            self.edge_residual = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.node_residual = nn.Identity()
            self.edge_residual = nn.Identity()
        self.edge_linear = nn.Linear(3 * in_dim, out_dim)
        self.edge_activation = nn.LeakyReLU(0.2)
        self.attention = nn.Linear(out_dim, heads)
        self.node_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.node_out_projection = (
            nn.Linear(out_dim, out_dim) if stabilized else nn.Identity()
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_input = self.node_input_norm(node_features)
        edge_input = self.edge_input_norm(edge_features)
        source, target = edge_index[0], edge_index[1]
        updated_edges = self.edge_activation(
            self.edge_linear(
                torch.cat(
                    [node_input[target], edge_input, node_input[source]],
                    dim=-1,
                )
            )
        )
        updated_nodes = self.propagate(
            edge_index, x=node_input, updated_edges=updated_edges
        )
        if self.stabilized:
            updated_nodes = self.node_residual(node_features) + self.node_out_projection(
                updated_nodes
            )
            updated_edges = self.edge_residual(edge_features) + updated_edges
        return updated_nodes, updated_edges

    def message(
        self,
        x_j: torch.Tensor,
        updated_edges: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i,
    ) -> torch.Tensor:
        logits = self.attention(updated_edges)
        weights = softmax(logits, index, ptr, size_i)
        values = self.node_linear(x_j).view(-1, self.heads, self.head_dim)
        return (weights.unsqueeze(-1) * values).reshape(-1, self.heads * self.head_dim)


class PdEgatSCLBranch(nn.Module):
    """One independent real- or imaginary-current branch.

    ``tangent_coeff`` predicts only the two physical tangent degrees of
    freedom.  ``cross`` retains the paper-style three-vector followed by
    ``n x raw_current`` for loading and reproducing older checkpoints.
    """

    def __init__(
        self,
        node_input_dim: int = 14,
        edge_input_dim: int = 5,
        m_hidden_dim: int = 512,
        g_hidden_dim: int = 256,
        num_egat_layers: int = 4,
        num_heads: int = 4,
        stabilized_egat: bool = True,
        output_parameterization: str = "cross",
        use_film: bool = False,
        condition_dim: int = 4,
    ) -> None:
        super().__init__()
        if output_parameterization not in {"tangent_coeff", "cross"}:
            raise ValueError(
                "output_parameterization must be 'tangent_coeff' or 'cross', "
                f"got {output_parameterization!r}"
            )
        self.output_parameterization = output_parameterization
        self.use_film = bool(use_film)
        encoder_widths = [
            m_hidden_dim // 4,
            m_hidden_dim // 2,
            (3 * m_hidden_dim) // 4,
            m_hidden_dim,
        ]
        self.node_encoder = _encoder_mlp(node_input_dim, encoder_widths)
        self.edge_encoder = _encoder_mlp(edge_input_dim, encoder_widths)
        self.encoder_film = (
            FiLM(condition_dim, m_hidden_dim) if self.use_film else nn.Identity()
        )

        self.egat_layers = nn.ModuleList()
        layer_input_dim = m_hidden_dim
        for _ in range(num_egat_layers):
            self.egat_layers.append(
                EGATConv(
                    layer_input_dim,
                    g_hidden_dim,
                    heads=num_heads,
                    stabilized=stabilized_egat,
                )
            )
            layer_input_dim = g_hidden_dim
        self.egat_films = nn.ModuleList(
            [FiLM(condition_dim, g_hidden_dim) for _ in range(num_egat_layers)]
            if self.use_film
            else []
        )

        self.decoder_input_norm = (
            nn.LayerNorm(g_hidden_dim) if stabilized_egat else nn.Identity()
        )

        decoder_widths = [
            max(g_hidden_dim // 2, 8),
            max(g_hidden_dim // 4, 4),
            max(g_hidden_dim // 16, 3),
        ]
        decoder_output_dim = 2 if output_parameterization == "tangent_coeff" else 3
        self.decoder = _decoder_mlp(
            g_hidden_dim, decoder_widths, out_dim=decoder_output_dim
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        normals: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        if self.use_film:
            if condition is None:
                raise ValueError("condition is required when use_film=True")
            nodes = self.encoder_film(nodes, condition)
        for layer_index, layer in enumerate(self.egat_layers):
            nodes, edges = layer(nodes, edge_index, edges)
            if self.use_film:
                nodes = self.egat_films[layer_index](nodes, condition)
        decoded_current = self.decoder(self.decoder_input_norm(nodes))
        if self.output_parameterization == "tangent_coeff":
            tangent_1, tangent_2 = _compute_tangent_frames(normals)
            return (
                decoded_current[:, 0:1] * tangent_1
                + decoded_current[:, 1:2] * tangent_2
            )
        return torch.linalg.cross(normals, decoded_current, dim=-1)


class PdEgatSCL_model(nn.Module):
    """Two-branch PdEgatSCL model producing ``[Re Jxyz, Im Jxyz]``."""

    def __init__(
        self,
        m_hidden_dim: int = 512,
        g_hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "output_scale", torch.tensor(1.0, dtype=torch.float32)
        )
        branch_kwargs = dict(
            node_input_dim=14,
            edge_input_dim=5,
            m_hidden_dim=m_hidden_dim,
            g_hidden_dim=g_hidden_dim,
            num_egat_layers=num_layers,
            num_heads=num_heads,
            stabilized_egat=True,
            output_parameterization="cross",
            use_film=False,
            condition_dim=4,
        )
        self.real_branch = PdEgatSCLBranch(**branch_kwargs)
        self.imaginary_branch = PdEgatSCLBranch(**branch_kwargs)

    def forward(self, data) -> torch.Tensor:
        node_features = _read(data, "node_features").float()
        edge_index = _read(data, "edge_index").long()
        edge_features = _read(data, "edge_features").float()
        normals = _read(data, "normals").float()
        condition = None
        if node_features.ndim != 2 or node_features.shape[1] != 14:
            raise ValueError("node_features must have shape [N, 14]")
        if edge_features.ndim != 2 or edge_features.shape[1] != 5:
            raise ValueError("edge_features must have shape [E, 5]")
        real_current = self.real_branch(
            node_features, edge_index, edge_features, normals, condition
        )
        imaginary_current = self.imaginary_branch(
            node_features, edge_index, edge_features, normals, condition
        )
        return torch.cat([real_current, imaginary_current], dim=-1) * self.output_scale


