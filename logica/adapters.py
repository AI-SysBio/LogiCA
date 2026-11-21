"""Native-head-preserving Gated Cross-Attention, Self-Adapters, and Late-Layer LoRA.

Implements Section 2.4 (Eq. 5) of the LogiCA paper:
  H_c^m = H^m + g(s_m) * phi_out^m(Z_N^m - Z_0^m),   m in {x, y},
along with:
  - Cross-modal adapters supporting ``softmax``, length-invariant ``sigmoid``, and ``topk_softmax``
    attention plus optional intra-modality pocket self-attention;
  - Native-width ``HeterogeneousCrossAttentionStack`` for seamless compatibility with
    ``ydchen/logica-tcr`` checkpoints;
  - ``UnimodalGatedAdapter`` for single-BioFM logit-space fine-tuning when ``partner=None``;
  - Built-in ``LoRALinear`` injection into the top ``k`` transformer layers of any BioFM.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import discover_transformer_layers

AttentionMode = Literal["softmax", "sigmoid", "topk_softmax"]
GateMode = Literal["tanh_zero", "v1_sigmoid", "layerscale_zero", "none"]
AdapterStyle = Literal["shared_bottleneck", "native_additional_layers"]


# =============================================================================
# 1. Late-Layer LoRA Injection (Model-Agnostic Across BioFMs)
# =============================================================================


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen ``nn.Linear`` projection (Hu et al., 2022)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive; got {rank}")
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return base_out + self.scaling * delta


def inject_late_layer_lora(
    backbone_or_module: nn.Module,
    top_k_layers: int = 4,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_linear_suffixes: tuple[str, ...] = (
        "query",
        "key",
        "value",
        "dense",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ),
) -> list[nn.Parameter]:
    """Injects ``LoRALinear`` strictly into the top ``top_k_layers`` transformer blocks of a BioFM."""
    if top_k_layers <= 0:
        return []
    layers = discover_transformer_layers(backbone_or_module)
    if not layers:
        return []

    selected = list(layers)[-top_k_layers:]
    lora_params: list[nn.Parameter] = []
    for block in selected:
        for _, parent_mod in list(block.named_modules()):
            for child_name, child_mod in list(parent_mod.named_children()):
                if isinstance(child_mod, nn.Linear) and child_name.endswith(target_linear_suffixes):
                    wrapped = LoRALinear(child_mod, rank=rank, alpha=alpha, dropout=dropout)
                    wrapped.to(device=child_mod.weight.device, dtype=child_mod.weight.dtype)
                    setattr(parent_mod, child_name, wrapped)
                    lora_params.extend([wrapped.lora_a, wrapped.lora_b])
    return lora_params


# =============================================================================
# 2. Multi-Layer Hidden Mixer
# =============================================================================


class MultiLayerHiddenMixer(nn.Module):
    """Learnable softmax mixture over the last ``num_layers`` BioFM hidden states."""

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        selected = list(hidden_states)[-self.num_layers :]
        if len(selected) == 1:
            return selected[0]
        weights = torch.softmax(self.layer_logits[-len(selected) :], dim=0)
        stacked = torch.stack(selected, dim=0)
        return (weights[:, None, None, None] * stacked).sum(dim=0)


# =============================================================================
# 3. Cross-Attention Blocks & Bidirectional Interaction Towers
# =============================================================================


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-modal attention block (v1 checkpoint-compatible + v2 extensions).

    Attribute names ``norm_q``, ``norm_kv``, ``attn``, ``norm_ff``, ``ff`` match the released
    ``Yale-CompBio/logica`` state dicts identically.
    """

    def __init__(
        self,
        width: int,
        heads: int = 4,
        dropout: float = 0.1,
        attn_mode: AttentionMode = "softmax",
        topk_keys: int = 32,
        use_intra_self_attn: bool = False,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.attn_mode = attn_mode
        self.topk_keys = topk_keys
        self.use_intra_self_attn = use_intra_self_attn

        if use_intra_self_attn:
            self.norm_self = nn.LayerNorm(width)
            self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.self_scale = nn.Parameter(torch.zeros(width))

        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        if attn_mode == "sigmoid":
            self.sigmoid_attn_bias = nn.Parameter(torch.full((1, heads, 1, 1), -math.log(64.0)))
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.dropout = nn.Dropout(dropout)

    def _custom_cross_attention(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = q_in.shape
        kv_len = kv_in.shape[1]
        q_proj, k_proj, v_proj = F._in_projection_packed(
            q_in, kv_in, kv_in, self.attn.in_proj_weight, self.attn.in_proj_bias
        )
        q = q_proj.view(bsz, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = k_proj.view(bsz, kv_len, self.heads, self.head_dim).transpose(1, 2)
        v = v_proj.view(bsz, kv_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pad_mask = context_padding_mask[:, None, None, :].bool()

        if self.attn_mode == "sigmoid":
            scores = scores + self.sigmoid_attn_bias
            weights = torch.sigmoid(scores).masked_fill(pad_mask, 0.0)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0)
        elif self.attn_mode == "topk_softmax":
            scores = scores.masked_fill(pad_mask, -torch.inf)
            k_eff = min(self.topk_keys, kv_len)
            if k_eff < kv_len:
                topk_vals, topk_idx = torch.topk(scores, k=k_eff, dim=-1)
                sparse = torch.full_like(scores, -torch.inf)
                sparse.scatter_(-1, topk_idx, topk_vals)
                scores = sparse
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        else:
            raise ValueError(f"Unsupported attn_mode: {self.attn_mode!r}")

        avg_weights = weights.mean(dim=1)
        weights_drop = F.dropout(weights, p=self.attn.dropout, training=self.training)
        out = torch.matmul(weights_drop, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.width)
        return F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias), avg_weights

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor,
        query_padding_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_intra_self_attn:
            q_norm = self.norm_self(query)
            self_out, _ = self.self_attn(
                q_norm,
                q_norm,
                q_norm,
                key_padding_mask=query_padding_mask,
                need_weights=False,
            )
            query = query + self.dropout(self_out) * torch.tanh(self.self_scale)

        q_pre = self.norm_q(query)
        kv_pre = self.norm_kv(context)
        if self.attn_mode == "softmax":
            attended, attn_w = self.attn(
                q_pre,
                kv_pre,
                kv_pre,
                key_padding_mask=context_padding_mask,
                need_weights=return_attn_weights,
                average_attn_weights=True,
            )
        else:
            attended, attn_w = self._custom_cross_attention(q_pre, kv_pre, context_padding_mask)

        hidden = query + self.dropout(attended)
        out = hidden + self.dropout(self.ff(self.norm_ff(hidden)))
        if return_attn_weights:
            assert attn_w is not None
            return out, attn_w
        return out


class BidirectionalCrossAttention(nn.Module):
    """Coupled N-layer bidirectional (``p2d`` and ``d2p``) cross-attention tower."""

    def __init__(
        self,
        width: int = 384,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        p2d_attn_mode: AttentionMode = "softmax",
        d2p_attn_mode: AttentionMode = "softmax",
        d2p_topk_keys: int = 32,
        use_intra_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.p2d = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width,
                    heads,
                    dropout=dropout,
                    attn_mode=p2d_attn_mode,
                    use_intra_self_attn=use_intra_self_attn,
                )
                for _ in range(layers)
            ]
        )
        self.d2p = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width,
                    heads,
                    dropout=dropout,
                    attn_mode=d2p_attn_mode,
                    topk_keys=d2p_topk_keys,
                    use_intra_self_attn=use_intra_self_attn,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        primary: torch.Tensor,
        partner: torch.Tensor,
        primary_padding_mask: torch.Tensor,
        partner_padding_mask: torch.Tensor,
        primary_interface_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary_interface_mask is not None:
            allowed = primary_interface_mask.bool() & (~primary_padding_mask)
            if not bool(allowed.any(dim=1).all()):
                raise ValueError("Every example must have >= 1 unmasked interface token.")
            primary_key_mask = ~allowed
        else:
            allowed = None
            primary_key_mask = primary_padding_mask

        for p_block, d_block in zip(self.p2d, self.d2p, strict=True):
            next_p = p_block(
                primary,
                partner,
                partner_padding_mask,
                query_padding_mask=primary_padding_mask,
            )
            assert isinstance(next_p, torch.Tensor)
            if allowed is not None:
                next_p = torch.where(allowed[:, :, None], next_p, primary)
            next_d = d_block(
                partner,
                primary,
                primary_key_mask,
                query_padding_mask=partner_padding_mask,
            )
            assert isinstance(next_d, torch.Tensor)
            primary, partner = next_p, next_d
        return primary, partner


class HeterogeneousCrossAttentionLayer(nn.Module):
    """Native-dimension ``SelfAttn -> CrossAttn(kdim=peer_dim) -> FFN`` block.

    Matches the state-dict layout of ``AbLang2CrossAttentionLayer`` so both newly trained
    models and released ``ydchen/logica-tcr`` checkpoints run under one unified interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        cross_kv_dim: int | None = None,
    ) -> None:
        super().__init__()
        kv_dim = cross_kv_dim if cross_kv_dim is not None else hidden_size
        if hidden_size % num_heads != 0:
            # Fall back to largest divisor <= 8
            for h in (8, 6, 5, 4, 2, 1):
                if hidden_size % h == 0:
                    num_heads = h
                    break
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=kv_dim,
            vdim=kv_dim,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_padding_mask: torch.Tensor | None = None,
        peer_hidden_states: torch.Tensor | None = None,
        peer_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=self_padding_mask,
            need_weights=False,
        )
        hidden_states = self.norm1(hidden_states + self.dropout(attn_out))
        if peer_hidden_states is not None:
            cross_out, _ = self.cross_attn(
                hidden_states,
                peer_hidden_states,
                peer_hidden_states,
                key_padding_mask=peer_padding_mask,
                need_weights=False,
            )
            hidden_states = self.norm2(hidden_states + self.dropout(cross_out))
        ff_out = self.ff(hidden_states)
        return self.norm3(hidden_states + self.dropout(ff_out))


class UnimodalGatedAdapter(nn.Module):
    """Gated bottleneck self-attention adapter for single-BioFM logit-space fine-tuning.

    When LogiCA is instantiated with a single BioFM (``partner=None``), this module refines
    token hidden states in a shared bottleneck space and returns a gated residual update
    directly into the BioFM's native hidden space prior to ``lm_head``.
    """

    def __init__(
        self,
        hidden_size: int,
        width: int = 384,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        gate_mode: GateMode = "tanh_zero",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.width = width
        self.gate_mode = gate_mode
        self.proj_in = nn.Linear(hidden_size, width, bias=False)
        self.proj_out = nn.Linear(width, hidden_size, bias=False)
        nn.init.orthogonal_(self.proj_in.weight)
        nn.init.orthogonal_(self.proj_out.weight)

        if gate_mode == "v1_sigmoid":
            self.gate = nn.Parameter(torch.tensor([-6.0]))
        elif gate_mode == "layerscale_zero":
            self.gate = nn.Parameter(torch.zeros(hidden_size))
        elif gate_mode == "tanh_zero":
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gate", None)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                    attn_mode="softmax",
                    use_intra_self_attn=False,
                )
                for _ in range(layers)
            ]
        )

    def _scale(self) -> torch.Tensor | float:
        if self.gate_mode == "none" or self.gate is None:
            return 1.0
        if self.gate_mode == "v1_sigmoid":
            return torch.sigmoid(self.gate)
        return torch.tanh(self.gate)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        z0 = self.proj_in(hidden)
        z = z0
        for block in self.blocks:
            out = block(z, z, padding_mask, query_padding_mask=padding_mask)
            assert isinstance(out, torch.Tensor)
            z = out
        return hidden + self._scale() * self.proj_out(z - z0)
