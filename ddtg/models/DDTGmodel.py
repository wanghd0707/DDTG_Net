"""
DDTGmodel: the public implementation of the paper's DDGT-Net.

One-shot mapping: (mesh, incident wave, patch graph) → J_S surface currents.
No MoM impedance matrix Z, no residuals b - Zx, no explicit outer iterations.

Architecture (single forward pass, weights shared):
    MeshGNNEncoder               : [N, F_S] + face graph → [N, d]
    PatchQueryInit               : [F_g] → [P, K, d]
    LocalCrossAttnEncoder        : [P, K, d] ← [N, d]
    DensePatchTransformerBlock × L
    JDecoder                     : [N, F_S] + [P, K, d] → [N, 4]

Output: [N_faces, 4] = [Re α, Im α, Re β, Im β] in local tangent frames
    J_S(r_i) = α_i * t1_i + β_i * t2_i     (automatically satisfies n̂ · J = 0)

Expected data attributes:
    data.face_feat          [N_faces, 14]
    data.face_edge_index    [2, E_f] long
    data.face_edge_attr     [E_f, 5]
    data.face_patch_id      [N_faces] long
    data.patch_feat         [P, F_g]
    data.patch_edge_index   [2, E] long
    data.patch_edge_attr    [E, F_e]
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared MLP building block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        seq: list[nn.Module] = []
        for i in range(len(dims) - 1):
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                seq.append(nn.SiLU())
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# FiLM: Feature-wise Linear Modulation
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Residual FiLM: h ← (1 + γ) ⊙ h + β. Initialized to identity."""
    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


# ---------------------------------------------------------------------------
# Component 1: Mesh GNN encoder (face graph message passing)
# ---------------------------------------------------------------------------

class MeshGNNLayer(nn.Module):
    """Edge-conditioned message passing on the face adjacency graph."""
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return h
        N = h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        msg  = self.message_mlp(torch.cat([h[dst], h[src], edge_attr], dim=-1))
        agg  = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        upd  = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + self.dropout(upd))


class MeshGNNEncoder(nn.Module):
    """Encode face features using the face adjacency graph."""
    def __init__(self, in_dim, hidden_dim, edge_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers  = nn.ModuleList([MeshGNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, face_feat, face_edge_index, face_edge_attr):
        h = self.input_proj(face_feat)
        for layer in self.layers:
            h = layer(h, face_edge_index, face_edge_attr)
        return self.out_norm(h)


# ---------------------------------------------------------------------------
# Component 2: Patch token initialiser
# ---------------------------------------------------------------------------

class PatchQueryInit(nn.Module):
    """K learnable base tokens conditioned on patch geometry."""
    def __init__(self, K: int, patch_feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.K = K
        self.base_tokens = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)
        self.geo_proj    = MLP(patch_feat_dim, hidden_dim, hidden_dim, layers=2)

    def forward(self, patch_feat):
        geo    = self.geo_proj(patch_feat)         # [P, d]
        tokens = self.base_tokens.unsqueeze(0)     # [1, K, d]
        return tokens + geo.unsqueeze(1)           # [P, K, d]


# ---------------------------------------------------------------------------
# Vectorized scatter helpers
# ---------------------------------------------------------------------------

def _pack_faces_to_patches(h, face_patch_id, P):
    device, dtype = h.device, h.dtype
    N, d = h.shape[0], h.shape[1]
    counts   = torch.bincount(face_patch_id, minlength=P)
    max_Np   = int(counts.max().item())
    sort_idx = face_patch_id.argsort(stable=True)
    pid_s    = face_patch_id[sort_idx]
    offsets  = torch.zeros(P + 1, dtype=torch.long, device=device)
    offsets[1:] = counts.cumsum(0)
    within_idx = torch.arange(N, device=device) - offsets[pid_s]
    h_pad    = torch.zeros(P, max_Np, d, device=device, dtype=dtype)
    key_mask = torch.ones(P, max_Np, dtype=torch.bool, device=device)
    h_pad[pid_s, within_idx]    = h[sort_idx]
    key_mask[pid_s, within_idx] = False
    return h_pad, key_mask, sort_idx, within_idx


# ---------------------------------------------------------------------------
# Component 3: Local cross-attention encoder
# ---------------------------------------------------------------------------

class LocalCrossAttnEncoder(nn.Module):
    """Per-patch cross-attention: K tokens attend to local face embeddings."""
    def __init__(self, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_patches, h_faces, face_patch_id, num_patches):
        P  = num_patches
        kv, key_mask, _, _ = _pack_faces_to_patches(h_faces, face_patch_id, P)
        attn_out, _ = self.cross_attn(z_patches, kv, kv, key_padding_mask=key_mask)
        return self.norm(attn_out + z_patches)


# ---------------------------------------------------------------------------
# Component 4a: Sparse patch token attention (no norm / residual)
# ---------------------------------------------------------------------------

class SparsePatchTokenAttention(nn.Module):
    """
    Core sparse patch-level multi-token attention with optional edge bias.

    For each edge q → p (src → dst) in the patch graph, the K tokens of
    patch p attend to the K tokens of patch q via multi-head attention.
    An edge-feature bias is added to attention scores (one scalar per head).

    Returns raw attention output (before any residual or LayerNorm).
    SparsePatchTransformerBlock is responsible for residual connections and
    normalization.

    input:
        z_patch [P, K, H]         patch token tensor
        patch_edge_index [2, E]   (src=q, dst=p)
        patch_edge_attr [E, F_e]  edge features
    output:
        out [P, K, H]             aggregated attention result
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads    = num_heads
        self.d_head       = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_edge_bias:
            self.edge_bias_mlp = MLP(edge_dim, hidden_dim // 2, num_heads, layers=2)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,                # [P, K, d]
        patch_edge_index: torch.Tensor, # [2, E]
        patch_edge_attr: torch.Tensor,  # [E, F_e]
    ) -> torch.Tensor:                  # [P, K, d]
        P, K, d = z.shape
        H, d_h  = self.num_heads, self.d_head
        scale   = d_h ** -0.5
        device, dtype = z.device, z.dtype

        Q  = self.q_proj(z).view(P, K, H, d_h)
        Kk = self.k_proj(z).view(P, K, H, d_h)
        V  = self.v_proj(z).view(P, K, H, d_h)

        E = patch_edge_index.shape[1]
        if E == 0:
            return torch.zeros_like(z)

        src, dst = patch_edge_index[0], patch_edge_index[1]

        # Edge bias [E, H]
        if self.use_edge_bias:
            edge_bias = self.edge_bias_mlp(patch_edge_attr.float())
        else:
            edge_bias = torch.zeros(E, H, device=device, dtype=dtype)

        # Sort edges by dst for contiguous packing
        sort_e  = dst.argsort(stable=True)
        dst_s   = dst[sort_e]
        src_s   = src[sort_e]
        bias_s  = edge_bias[sort_e]

        degree   = torch.bincount(dst_s, minlength=P)
        max_M    = int(degree.max().item())
        e_offs   = torch.zeros(P + 1, dtype=torch.long, device=device)
        e_offs[1:] = degree.cumsum(0)
        within_e = torch.arange(E, device=device) - e_offs[dst_s]

        # Pack neighbor K, V, bias into padded tensors [P, max_M, K, H, d_h]
        Kk_pad = torch.zeros(P, max_M, K, H, d_h, device=device, dtype=dtype)
        V_pad  = torch.zeros(P, max_M, K, H, d_h, device=device, dtype=dtype)
        Kk_pad[dst_s, within_e] = Kk[src_s]
        V_pad[dst_s, within_e]  = V[src_s]

        bias_pad  = torch.zeros(P, max_M, H, device=device, dtype=dtype)
        bias_pad[dst_s, within_e] = bias_s

        key_valid = torch.zeros(P, max_M, dtype=torch.bool, device=device)
        key_valid[dst_s, within_e] = True

        # Batched bmm: [P*H, K, d_h] × [P*H, d_h, max_M*K] → [P*H, K, max_M*K]
        Q_flat  = Q.permute(0, 2, 1, 3).reshape(P * H, K, d_h)
        Kk_flat = Kk_pad.reshape(P, max_M * K, H, d_h).permute(0, 2, 1, 3).reshape(P * H, max_M * K, d_h)
        V_flat  = V_pad.reshape( P, max_M * K, H, d_h).permute(0, 2, 1, 3).reshape(P * H, max_M * K, d_h)

        scores  = torch.bmm(Q_flat, Kk_flat.transpose(1, 2)) * scale  # [P*H, K, max_M*K]

        # Broadcast edge bias over both token axes
        bias_pm = bias_pad.permute(0, 2, 1).reshape(P * H, max_M)
        bias_km = bias_pm.unsqueeze(1).expand(P * H, K, max_M)
        bias_for_scores = bias_km.unsqueeze(-1).expand(P * H, K, max_M, K).reshape(P * H, K, max_M * K)
        scores  = scores + bias_for_scores

        # Mask padding positions
        key_mask = (~key_valid).unsqueeze(-1).expand(P, max_M, K).reshape(P, max_M * K)
        key_mask = key_mask[:, None, None, :].expand(P, H, K, max_M * K).reshape(P * H, K, max_M * K)
        scores   = scores.masked_fill(key_mask, float('-inf'))

        attn = self.attn_drop(torch.softmax(scores, dim=-1).nan_to_num(0.0))
        agg  = torch.bmm(attn, V_flat)                                  # [P*H, K, d_h]

        out = agg.reshape(P, H, K, d_h).permute(0, 2, 1, 3).reshape(P, K, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Component 4b: Sparse Patch Transformer Block (Pre-LN + LayerScale + FFN)
# ---------------------------------------------------------------------------

class SparsePatchTransformerBlock(nn.Module):
    """
    Pre-LN Transformer block operating on sparse patch token graph.

    Structure:
        z = z + dropout(γ_attn ⊙ SparsePatchTokenAttention(LayerNorm(z)))
        z = z + dropout(γ_ffn  ⊙ FFN(LayerNorm(z)))

    LayerScale (γ_attn, γ_ffn initialised to layer_scale_init=1e-2) ensures
    the block starts as a near-identity transformation, allowing stable
    initialisation even with many stacked layers.

    Simulates multi-round inter-subdomain information propagation in a fixed-
    depth forward pass — not an explicit iterative solver.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        use_edge_bias: bool = True,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = SparsePatchTokenAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            use_edge_bias=use_edge_bias,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.gamma_attn = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
        self.gamma_ffn  = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))

    def forward(
        self,
        z: torch.Tensor,                # [P, K, d]
        patch_edge_index: torch.Tensor, # [2, E]
        patch_edge_attr: torch.Tensor,  # [E, F_e]
    ) -> torch.Tensor:                  # [P, K, d]
        # Pre-LN attention with LayerScale residual
        attn_out = self.attn(self.norm1(z), patch_edge_index, patch_edge_attr)
        z = z + self.dropout(self.gamma_attn.view(1, 1, -1) * attn_out)
        # Pre-LN FFN with LayerScale residual
        ffn_out = self.ffn(self.norm2(z))
        z = z + self.dropout(self.gamma_ffn.view(1, 1, -1) * ffn_out)
        return z


# ---------------------------------------------------------------------------
# Component 4c: Dense Patch Transformer Block (full-graph attention)
# ---------------------------------------------------------------------------

class DensePatchTransformerBlock(nn.Module):
    """
    Dense (full-graph) patch transformer block with Green-aware physics bias.

    All P*K tokens attend globally (no sparsity). For edges in the patch graph,
    Green's-function features are scatter-added to the attention logits as a
    physics bias, giving the model both:
      - full global visibility (every patch sees every other patch), and
      - physical inductive bias for electromagnetically coupled pairs.

    Physics bias implementation:
      edge_bias [E, H] = edge_bias_mlp(patch_edge_attr)
      For edge (i→j): scores[b, :, i_token_rows, j_token_cols] += bias[e, :]
      Each patch pair maps to a K×K block of token pairs (constant bias per head).

    Structure (Pre-LN + LayerScale, identical to SparsePatchTransformerBlock):
        z = z + dropout(γ_attn ⊙ DenseMHA_+_PhysBias(LayerNorm(z)))
        z = z + dropout(γ_ffn  ⊙ FFN(LayerNorm(z)))
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        use_edge_bias: bool = True,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads     = num_heads
        self.d_head        = hidden_dim // num_heads
        self.scale         = self.d_head ** -0.5
        self.use_edge_bias = use_edge_bias

        self.norm1     = nn.LayerNorm(hidden_dim)
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        if use_edge_bias:
            self.edge_bias_mlp = MLP(edge_dim, hidden_dim // 2, num_heads, layers=2)

        self.norm2      = nn.LayerNorm(hidden_dim)
        self.ffn        = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )
        self.dropout    = nn.Dropout(dropout)
        self.gamma_attn = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))
        self.gamma_ffn  = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))

    def forward(
        self,
        z: torch.Tensor,                 # [P_total, K, d]
        patch_batch_idx: torch.Tensor,   # [P_total] sample index per patch
        patch_edge_index: torch.Tensor,  # [2, E]  global patch indices
        patch_edge_attr: torch.Tensor,   # [E, F_e]
        **kwargs,
    ) -> torch.Tensor:                   # [P_total, K, d]
        P_total, K, D = z.shape
        H, d_h = self.num_heads, self.d_head
        B = int(patch_batch_idx.max().item()) + 1
        sizes_t = torch.bincount(patch_batch_idx, minlength=B)   # [B] int tensor
        sizes   = sizes_t.tolist()
        max_P   = max(sizes)
        seq     = max_P * K

        # Per-sample patch start offset in the global patch list
        offsets = torch.cat([patch_batch_idx.new_zeros(1), sizes_t.cumsum(0)[:-1]])  # [B]

        # Pack z into [B, seq, D]; True in kp_mask → padded (ignored in attn)
        z_pad   = z.new_zeros(B, seq, D)
        kp_mask = torch.ones(B, seq, dtype=torch.bool, device=z.device)
        offset  = 0
        for b, p in enumerate(sizes):
            z_pad[b, : p * K] = z[offset : offset + p].reshape(p * K, D)
            kp_mask[b, : p * K] = False
            offset += p

        # Pre-LN
        z_n = self.norm1(z_pad)   # [B, seq, D]

        # Q / K / V → [B, H, seq, d_h]
        def _heads(x):
            return x.view(B, seq, H, d_h).permute(0, 2, 1, 3)

        Q  = _heads(self.q_proj(z_n))
        Kk = _heads(self.k_proj(z_n))
        V  = _heads(self.v_proj(z_n))

        # Dense attention logits [B, H, seq, seq]
        scores = torch.matmul(Q, Kk.transpose(-2, -1)) * self.scale

        # Physics bias: scatter Green's-function edge features onto logits
        E = patch_edge_index.shape[1]
        if self.use_edge_bias and E > 0:
            edge_bias = self.edge_bias_mlp(patch_edge_attr)      # [E, H]

            src_g     = patch_edge_index[0]                       # [E] global patch id
            dst_g     = patch_edge_index[1]
            b_e       = patch_batch_idx[src_g]                    # [E] batch index
            local_src = src_g - offsets[b_e]                      # [E] local patch id
            local_dst = dst_g - offsets[b_e]

            # Accumulate per-(b, local_src, local_dst) bias via flat index_add_
            # flat layout: [B * max_P * max_P, H]
            flat_idx   = b_e * (max_P * max_P) + local_src * max_P + local_dst
            patch_bias = z.new_zeros(B * max_P * max_P, H)
            patch_bias.index_add_(0, flat_idx, edge_bias)
            patch_bias = patch_bias.view(B, max_P, max_P, H)

            # Expand each patch-pair bias to cover its K×K token block.
            # [B, max_P, max_P, H]
            # → unsqueeze → [B, max_P, 1, max_P, 1, H]
            # → expand    → [B, max_P, K, max_P, K, H]
            # → view      → [B, seq, seq, H]   (row = p*K+k, col = p*K+k: correct)
            pb = (patch_bias
                  .unsqueeze(2).unsqueeze(4)
                  .expand(B, max_P, K, max_P, K, H)
                  .contiguous()
                  .view(B, seq, seq, H))
            scores = scores + pb.permute(0, 3, 1, 2)             # [B, H, seq, seq]

        # Mask padded key positions
        scores = scores.masked_fill(kp_mask[:, None, None, :], float('-inf'))

        # Softmax + dropout + value aggregation
        attn     = self.attn_drop(torch.softmax(scores, dim=-1).nan_to_num(0.0))
        out      = torch.matmul(attn, V)                          # [B, H, seq, d_h]
        out      = out.permute(0, 2, 1, 3).contiguous().view(B, seq, D)
        attn_out = self.out_proj(out)

        # LayerScale + residual (attention)
        z_pad = z_pad + self.dropout(attn_out * self.gamma_attn)

        # Pre-LN FFN + LayerScale + residual
        z_pad = z_pad + self.dropout(self.ffn(self.norm2(z_pad)) * self.gamma_ffn)

        # Unpack → [P_total, K, D]
        return torch.cat(
            [z_pad[b, : sizes[b] * K].reshape(sizes[b], K, D) for b in range(B)],
            dim=0,
        )


def _filter_same_patch_edges(
    edge_index: torch.Tensor,   # [2, E]
    edge_attr: torch.Tensor,    # [E, F]
    face_patch_id: torch.Tensor # [N]
) -> tuple:                     # ([2, E'], [E', F])
    """Keep only edges (src, dst) where both faces belong to the same patch."""
    if edge_index.shape[1] == 0:
        return edge_index, edge_attr
    src, dst = edge_index[0], edge_index[1]
    mask = face_patch_id[src] == face_patch_id[dst]
    return edge_index[:, mask], edge_attr[mask]


# ---------------------------------------------------------------------------
# Component 5: GNN-refined token-to-face decoder
# ---------------------------------------------------------------------------

class TokenToFaceGNNDecoder(nn.Module):
    """
    Face cross-attends to patch tokens, then refines within patches via GNN.

    Forward stages:
      1. q_proj      : [face_feat | h_faces] → face query [N, d]
      2. cross_attn  : face queries → K patch tokens → context [N, d]  (per-patch)
      3. ctx_norm    : residual + LayerNorm
      4. dec_proj    : [face_feat | h_faces | ctx] → hidden [N, d]
      5. refine_layers: patch-internal MeshGNNLayer steps on hidden
      6. out_mlp     : [u | h_faces | ctx] → [N, 4] J coefficients
    """
    def __init__(
        self,
        face_feat_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        decoder_gnn_layers: int = 2,
        face_edge_dim: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_proj     = nn.Linear(face_feat_dim + hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ctx_norm   = nn.LayerNorm(hidden_dim)
        self.dec_proj   = MLP(face_feat_dim + hidden_dim * 2, hidden_dim, hidden_dim, layers=2)
        self.refine_layers = nn.ModuleList([
            MeshGNNLayer(hidden_dim, face_edge_dim) for _ in range(decoder_gnn_layers)
        ])
        self.out_mlp = MLP(hidden_dim * 3, hidden_dim, 4, layers=3)
        nn.init.normal_(self.out_mlp.net[-1].weight, std=0.005)
        nn.init.zeros_(self.out_mlp.net[-1].bias)

    def forward(
        self,
        face_feat: torch.Tensor,       # [N, face_feat_dim]
        h_faces: torch.Tensor,         # [N, hidden_dim]   (GNN-encoded + FiLM)
        z_patches: torch.Tensor,       # [P, K, hidden_dim]
        face_patch_id: torch.Tensor,   # [N] long
        num_patches: int,
        face_edge_index: torch.Tensor, # [2, E_f]
        face_edge_attr: torch.Tensor,  # [E_f, face_edge_dim]
    ) -> torch.Tensor:                 # [N, 4]
        P = num_patches
        N = face_feat.shape[0]
        device, dtype = face_feat.device, face_feat.dtype

        # 1. Query projection: face geometry + GNN context → query
        face_q = self.q_proj(torch.cat([face_feat, h_faces], dim=-1))  # [N, d]

        # 2. Pack per-patch, cross-attend to patch tokens
        q_pad, _, sort_idx, within_idx = _pack_faces_to_patches(face_q, face_patch_id, P)
        pid_s = face_patch_id[sort_idx]
        ctx_pad, _ = self.cross_attn(q_pad, z_patches, z_patches)      # [P, max_Np, d]

        # 3. Unpack + residual norm
        ctx = torch.zeros(N, self.hidden_dim, device=device, dtype=dtype)
        ctx[sort_idx] = ctx_pad[pid_s, within_idx]
        ctx = self.ctx_norm(ctx + face_q)                               # [N, d]

        # 4. Fuse and project to hidden
        u = self.dec_proj(torch.cat([face_feat, h_faces, ctx], dim=-1))  # [N, d]

        # 5. Patch-internal GNN refinement
        ei_dec, ea_dec = _filter_same_patch_edges(face_edge_index, face_edge_attr, face_patch_id)
        for layer in self.refine_layers:
            u = layer(u, ei_dec, ea_dec)

        # 6. Output
        return self.out_mlp(torch.cat([u, h_faces, ctx], dim=-1))      # [N, 4]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class _DDTGCore(nn.Module):
    """Internal core shared by the paper model and its controlled ablations."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tokens: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention: str = "dense",
        bias: str = "green",
    ) -> None:
        super().__init__()
        if attention not in {"dense", "sparse"}:
            raise ValueError("attention must be 'dense' or 'sparse'")
        if bias not in {"green", "geometry", "none"}:
            raise ValueError("bias must be 'green', 'geometry', or 'none'")
        self.attention = attention
        self.bias = bias

        # ── face encoder ──────────────────────────────────────────────────
        self.face_encoder = MeshGNNEncoder(
            in_dim=14, hidden_dim=hidden_dim,
            edge_dim=5, num_layers=2, dropout=dropout,
        )

        # ── patch token init + local cross-attention ───────────────────────
        self.patch_query_init = PatchQueryInit(num_tokens, 9, hidden_dim)
        self.local_cross_attn = LocalCrossAttnEncoder(hidden_dim, num_heads)

        # ── patch coupling layers ─────────────────────────────────────────
        block = DensePatchTransformerBlock if attention == "dense" else SparsePatchTransformerBlock
        self.patch_layers = nn.ModuleList([
            block(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_dim=9,
                ffn_mult=4,
                dropout=dropout,
                use_edge_bias=bias != "none",
                layer_scale_init=1e-2,
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

        # ── FiLM conditioning ─────────────────────────────────────────────
        self.film_face       = FiLM(4, hidden_dim)
        self.film_patch_init = FiLM(4, hidden_dim)
        self.film_gnn        = nn.ModuleList([FiLM(4, hidden_dim) for _ in range(num_layers)])

    @staticmethod
    def _build_geometry_only_edge_attr(
        patch_feat: torch.Tensor,
        patch_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 9-D geometry-only pairwise feature vector.

        ``patch_feat`` follows the fixed DDGT layout
        [center(3), mean_normal(3), size, curv_mean, curv_std].  The final
        two zero columns keep the edge-bias MLP identical to the full model,
        so the ablation changes only the physical inputs, not its capacity.
        Self-loop features are set to zero to match the synthetic zero-valued
        self-loop attributes used by the Green-aware path.
        """
        if patch_feat.shape[-1] < 6:
            raise ValueError(
                "geometry_only_edge_bias requires patch features containing "
                "center(3) and mean_normal(3)"
            )

        src, dst = patch_edge_index[0], patch_edge_index[1]
        centers = patch_feat[:, :3]
        normals = patch_feat[:, 3:6]
        normals = normals / torch.linalg.vector_norm(
            normals, dim=-1, keepdim=True
        ).clamp_min(1e-12)

        displacement = centers[dst] - centers[src]
        distance = torch.linalg.vector_norm(
            displacement, dim=-1, keepdim=True
        )
        direction = displacement / distance.clamp_min(1e-12)
        n_dst = normals[dst]
        n_src = normals[src]

        geometry_attr = torch.cat(
            [
                displacement,
                distance,
                (n_dst * n_src).sum(dim=-1, keepdim=True),
                (n_dst * direction).sum(dim=-1, keepdim=True),
                (n_src * direction).sum(dim=-1, keepdim=True),
                torch.zeros(
                    (patch_edge_index.shape[1], 2),
                    device=patch_feat.device,
                    dtype=patch_feat.dtype,
                ),
            ],
            dim=-1,
        )

        non_self = (src != dst).to(dtype=geometry_attr.dtype).unsqueeze(-1)
        return geometry_attr * non_self

    def forward(self, data) -> torch.Tensor:
        face_feat     = data.face_feat.float()
        if face_feat.ndim != 2 or face_feat.shape[1] != 14:
            raise ValueError(
                "data.face_feat must have shape [N, 14] using the paper feature "
                "layout [r-c_patch, normal, area, Re(E_inc), Im(E_inc), k]"
            )
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

        # 4. Prepare patch coupling inputs
        if self.attention == "dense":
            # Dense path: all patches attend to all — no self-loops needed
            patch_batch_idx = data.patch_batch_idx.long()
        else:
            # Sparse path: add self-loops so each patch attends to itself
            self_idx   = torch.arange(P, device=edge_index.device)
            sl_edges   = torch.stack([self_idx, self_idx], dim=0)
            sl_attr    = torch.zeros(P, edge_attr.shape[1], device=edge_attr.device, dtype=edge_attr.dtype)
            edge_index = torch.cat([edge_index, sl_edges], dim=1)
            edge_attr  = torch.cat([edge_attr,  sl_attr],  dim=0)
            patch_batch_idx = None

        if self.bias == "geometry":
            edge_attr = self._build_geometry_only_edge_attr(patch_feat, edge_index)

        # 5. Patch coupling + FiLM after each layer
        if self.attention == "dense":
            for layer, film in zip(self.patch_layers, self.film_gnn):
                z = layer(z, patch_batch_idx=patch_batch_idx,
                          patch_edge_index=edge_index, patch_edge_attr=edge_attr)
                z = film(z, cond_patch.unsqueeze(1))
        else:
            for layer, film in zip(self.patch_layers, self.film_gnn):
                z = layer(z, edge_index, edge_attr)
                z = film(z, cond_patch.unsqueeze(1))

        # 6. Decode local tangential current coefficients
        return self.j_decoder(
            face_feat, h_faces, z, face_patch_id, P,
            data.face_edge_index.long(), data.face_edge_attr.float(),
        )


class DDTGmodel(_DDTGCore):
    """Paper model using fixed 14-D facet features and Green-aware attention.

    Args:
        hidden_dim: Latent width (paper default: 256).
        num_tokens: Tokens per patch (paper default: 6).
        num_layers: Global patch Transformer layers (paper default: 4).
        num_heads: Attention heads (paper default: 8).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tokens: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention="dense",
            bias="green",
        )


__all__ = ["DDTGmodel"]
