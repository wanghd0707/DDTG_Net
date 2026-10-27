"""
patch_egat.py — Patch-GAT ablation model

The architecture follows DDTGmodel except that the patch coupling layer is
替换为 PatchGATBlock：

  SparsePatchTransformerBlock:
    K 个 token 对邻居所有 K 个 token 做 cross-attention（K×K 级别）

  PatchGATBlock (本文件):
    将每个 patch 的 K 个 token 均值池化为单一嵌入，
    在 patch 图上做 edge-aware 多头 GAT，
    再将更新量广播回 K 个 token（1 级别）

其余所有组件（MeshGNNEncoder、PatchQueryInit、LocalCrossAttnEncoder、
TokenToFaceGNNDecoder、FiLM 条件化）均与原模型相同。

It uses the same 14-D facet input and paper hyperparameters as the main model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the common building blocks from the public DDTG model.
from ..DDTGmodel import (
    FiLM,
    LocalCrossAttnEncoder,
    MLP,
    MeshGNNEncoder,
    PatchQueryInit,
    TokenToFaceGNNDecoder,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PatchGATLayer — 原始 GAT 注意力（加性注意力 + LeakyReLU，无点积）
# ═══════════════════════════════════════════════════════════════════════════════

class PatchGATLayer(nn.Module):
    """
    原始 GAT（Veličković 2018）多头注意力，在 patch 图上操作。

    注意力分数（加性，非点积）:
        e_ij = LeakyReLU( a_src · Wh_j + a_dst · Wh_i ) + edge_bias_ij
        α_ij = softmax over neighbors j of i

    输入: h [P, d]  — 均值池化后的 patch 嵌入
    输出: h'[P, d]  — 更新后的 patch 嵌入

    与 SparsePatchTransformerBlock 的结构差异:
      Transformer : Q·K^T 点积分数，有 FFN，Pre-LN + LayerScale
      原始 GAT    : a^T[Wh_i||Wh_j] 加性分数，无 FFN，简单残差
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_edge_bias: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads     = num_heads
        self.d_head        = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias

        # 共享特征投影（对应原始 GAT 中的 W）
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 加性注意力向量（分解为 src 和 dst 两部分，等价于 a^T[Wh_i||Wh_j]）
        self.attn_src = nn.Parameter(torch.empty(1, num_heads, self.d_head))
        self.attn_dst = nn.Parameter(torch.empty(1, num_heads, self.d_head))
        nn.init.xavier_uniform_(self.attn_src.view(1, -1).unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.view(1, -1).unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)

        if use_edge_bias:
            self.edge_bias_mlp = MLP(edge_dim, hidden_dim // 2, num_heads, layers=2)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,                   # [P, d]
        patch_edge_index: torch.Tensor,    # [2, E]
        patch_edge_attr: torch.Tensor,     # [E, F_e]
    ) -> torch.Tensor:                     # [P, d]
        P, d    = h.shape
        H, d_h  = self.num_heads, self.d_head
        device, dtype = h.device, h.dtype

        # 特征投影: Wh [P, H, d_h]
        Wh = self.W(h).view(P, H, d_h)

        # 每个节点的加性注意力贡献（标量，每头一个）
        alpha_src = (Wh * self.attn_src).sum(-1)   # [P, H]
        alpha_dst = (Wh * self.attn_dst).sum(-1)   # [P, H]

        E = patch_edge_index.shape[1]
        if E == 0:
            return torch.zeros_like(h)

        src, dst = patch_edge_index[0], patch_edge_index[1]

        # e_ij = LeakyReLU( alpha_src[src] + alpha_dst[dst] )  [E, H]
        e = self.leaky_relu(alpha_src[src] + alpha_dst[dst])

        # 加入边特征 bias
        if self.use_edge_bias:
            e = e + self.edge_bias_mlp(patch_edge_attr.float())  # [E, H]

        # 按 dst 排序并填入 padded 张量，做 masked softmax
        sort_e   = dst.argsort(stable=True)
        dst_s    = dst[sort_e]
        src_s    = src[sort_e]
        e_s      = e[sort_e]

        degree   = torch.bincount(dst_s, minlength=P)
        max_M    = int(degree.max().item())
        e_offs   = torch.zeros(P + 1, dtype=torch.long, device=device)
        e_offs[1:] = degree.cumsum(0)
        within_e = torch.arange(E, device=device) - e_offs[dst_s]

        # e_pad [P, max_M, H]，padding 位置为 -inf
        e_pad = torch.full((P, max_M, H), float('-inf'), device=device, dtype=dtype)
        e_pad[dst_s, within_e] = e_s

        # softmax over 邻居维度 (dim=1)
        alpha = self.attn_drop(
            torch.softmax(e_pad, dim=1).nan_to_num(0.0)
        )                                                     # [P, max_M, H]

        # 填入邻居特征 V = Wh[src]，[P, max_M, H, d_h]
        V_pad = torch.zeros(P, max_M, H, d_h, device=device, dtype=dtype)
        V_pad[dst_s, within_e] = Wh[src_s]

        # 加权聚合: [P, max_M, H, 1] × [P, max_M, H, d_h] → sum → [P, H, d_h]
        agg = (alpha.unsqueeze(-1) * V_pad).sum(1)           # [P, H, d_h]

        out = agg.reshape(P, d)
        return self.out_proj(out)                             # [P, d]


# ═══════════════════════════════════════════════════════════════════════════════
# PatchGATBlock — 原始 GAT 结构（无 FFN，简单 LN + 残差）
# ═══════════════════════════════════════════════════════════════════════════════

class PatchGATBlock(nn.Module):
    """
    原始 GAT 风格的 patch 间耦合块。

    与 SparsePatchTransformerBlock 的结构对比:
      Transformer: Pre-LN → 点积注意力 → LayerScale残差 → Pre-LN → FFN → LayerScale残差
      原始GAT    : LN → 加性注意力(LeakyReLU) → 残差   （无 FFN，无 LayerScale）

    数据流:
        h   = LayerNorm(z).mean(dim=1)    [P, d]   — K token 均值池化
        Δh  = PatchGATLayer(h, edges)     [P, d]   — 加性注意力聚合
        z   = z + dropout(Δh).unsqueeze(1)         — 广播残差回 K token
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_edge_bias: bool = True,
        negative_slope: float = 0.2,
        # 以下参数仅为接口兼容，不使用
        ffn_mult: int = 4,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gat  = PatchGATLayer(
            hidden_dim, num_heads, edge_dim,
            dropout, use_edge_bias, negative_slope,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,                   # [P, K, d]
        patch_edge_index: torch.Tensor,    # [2, E]
        patch_edge_attr: torch.Tensor,     # [E, F_e]
        **kwargs,
    ) -> torch.Tensor:                     # [P, K, d]
        h  = self.norm(z).mean(dim=1)                               # [P, d]
        dh = self.drop(self.gat(h, patch_edge_index, patch_edge_attr))  # [P, d]
        z  = z + dh.unsqueeze(1)                                    # [P, K, d]
        return z


# ═══════════════════════════════════════════════════════════════════════════════
# PatchGATModel -- ablation with the same interface as DDTGmodel
# ═══════════════════════════════════════════════════════════════════════════════

class PatchGATModel(nn.Module):
    """
    Patch-GAT ablation model.

    The only architectural change is that interpatch coupling uses
    ``PatchGATBlock`` instead of the Green-aware patch Transformer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tokens: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── face encoder ──────────────────────────────────────────────────
        self.face_encoder = MeshGNNEncoder(
            in_dim=14, hidden_dim=hidden_dim,
            edge_dim=5, num_layers=2, dropout=dropout,
        )

        # ── patch token init + local cross-attention ───────────────────────
        self.patch_query_init = PatchQueryInit(num_tokens, 9, hidden_dim)
        self.local_cross_attn = LocalCrossAttnEncoder(hidden_dim, num_heads)

        # ── patch 间耦合：PatchGATBlock × num_patch_layers ─────────────────
        self.patch_layers = nn.ModuleList([
            PatchGATBlock(
                hidden_dim        = hidden_dim,
                num_heads         = num_heads,
                edge_dim          = 9,
                ffn_mult          = 4,
                dropout           = dropout,
                use_edge_bias     = True,
                layer_scale_init  = 1e-2,
            )
            for _ in range(num_layers)
        ])

        # ── decoder ───────────────────────────────────────────────────────
        self.j_decoder = TokenToFaceGNNDecoder(
            face_feat_dim=14,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            decoder_gnn_layers=2,
            face_edge_dim=5,
        )

        # ── FiLM 条件化 ───────────────────────────────────────────────────
        self.film_face       = FiLM(4, hidden_dim)
        self.film_patch_init = FiLM(4, hidden_dim)
        self.film_gnn = nn.ModuleList([
            FiLM(4, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, data) -> torch.Tensor:
        face_feat     = data.face_feat.float()
        if face_feat.ndim != 2 or face_feat.shape[1] != 14:
            raise ValueError("data.face_feat must have shape [N, 14]")
        patch_feat    = data.patch_feat.float()
        edge_index    = data.patch_edge_index.long()
        edge_attr     = data.patch_edge_attr.float()
        face_patch_id = data.face_patch_id.long()
        cond_face     = data.cond_face.float()
        cond_patch    = data.cond_patch.float()
        P = int(patch_feat.shape[0])

        # 1. Face encoding + FiLM
        h_faces = self.face_encoder(
            face_feat, data.face_edge_index.long(), data.face_edge_attr.float()
        )
        h_faces = self.film_face(h_faces, cond_face)

        # 2. Patch token init
        z = self.patch_query_init(patch_feat)

        # 3. Local cross-attention + FiLM
        z = self.local_cross_attn(z, h_faces, face_patch_id, P)
        z = self.film_patch_init(z, cond_patch.unsqueeze(1))

        # 4. 加 self-loop（让每个 patch 在 GAT 中也关注自身）
        self_idx   = torch.arange(P, device=edge_index.device)
        sl_edges   = torch.stack([self_idx, self_idx], dim=0)
        sl_attr    = torch.zeros(P, edge_attr.shape[1],
                                 device=edge_attr.device, dtype=edge_attr.dtype)
        edge_index = torch.cat([edge_index, sl_edges], dim=1)
        edge_attr  = torch.cat([edge_attr,  sl_attr],  dim=0)

        # 5. Patch 间 GAT 耦合 + FiLM
        for layer, film in zip(self.patch_layers, self.film_gnn):
            z = layer(z, edge_index, edge_attr)
            z = film(z, cond_patch.unsqueeze(1))

        # 6. J 域解码
        return self.j_decoder(
            face_feat, h_faces, z, face_patch_id, P,
            data.face_edge_index.long(), data.face_edge_attr.float(),
        )
