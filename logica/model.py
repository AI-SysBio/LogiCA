"""Unified Model-Agnostic LogiCA Architecture (Dual-BioFM & Single-BioFM Modes).

Provides :class:`LogiCA`, a clean PyTorch module that:
  1. Couples **any pair of pretrained BioFMs** (via Hugging Face model IDs, ``ablang2`` IDs,
     or custom ``BioFMBackbone`` objects) through native-head-preserving Gated Cross-Attention,
  2. Supports **single-BioFM logit-space fine-tuning** when ``partner=None`` via a gated
     self-adapter and/or late-layer LoRA while preserving the exact native token head,
  3. Loads official pretrained checkpoints directly from Hugging Face Hub
     (``Yale-CompBio/logica`` for Protein–Ligand and ``ydchen/logica-tcr`` for TCR–Peptide)
     or local ``.pt`` / ``.safetensors`` files,
  4. Exposes high-level ``rank_variants``, ``dependency_map``, ``detect_kl_hotspots``, and
     ``gibbs_sample`` methods on the exact same native-logit interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .adapters import (
    AdapterStyle,
    AttentionMode,
    BidirectionalCrossAttention,
    GateMode,
    HeterogeneousCrossAttentionLayer,
    MultiLayerHiddenMixer,
    UnimodalGatedAdapter,
    inject_late_layer_lora,
)
from .backbones import CANONICAL_AAS, BioFMBackbone, load_biofm

ArchVersion = Literal["v2", "v1_compat"]

YALE_PROTEIN_LIGAND_HF_REPO = "Yale-CompBio/logica"
YALE_PROTEIN_LIGAND_SCALES: dict[str, tuple[str, str]] = {
    "8m": ("facebook/esm2_t6_8M_UR50D", "checkpoints/8m/best.pt"),
    "35m": ("facebook/esm2_t12_35M_UR50D", "checkpoints/35m/best.pt"),
    "150m": ("facebook/esm2_t30_150M_UR50D", "checkpoints/150m/best.pt"),
    "650m": ("facebook/esm2_t33_650M_UR50D", "checkpoints/650m/best.pt"),
}
TCR_LOGICA_HF_REPO = "ydchen/logica-tcr"


@dataclass
class LogiCAOutput:
    """Structured output of a :class:`LogiCA` forward pass in native token logit space."""

    primary_holo_logits: torch.Tensor
    primary_apo_logits: torch.Tensor
    partner_holo_logits: torch.Tensor | None = None
    partner_apo_logits: torch.Tensor | None = None
    primary_contextual_hidden: torch.Tensor | None = None
    partner_contextual_hidden: torch.Tensor | None = None


class LogiCA(nn.Module):
    """Model-agnostic Logit-Space Contrastive Alignment (LogiCA) wrapper.

    Supports both:
      - **Dual-BioFM cross-modal conditioning** (``partner_backbone`` provided): projects two
        arbitrary BioFMs into a shared interaction width ``width`` (or native cross-attention
        layers), exchanges context via bidirectional cross-attention, and injects a gated
        residual back into each BioFM's native hidden space right before its original ``lm_head``.
      - **Single-BioFM logit-space adaptation** (``partner_backbone=None``): applies a gated
        self-adapter and/or late-layer LoRA while keeping the BioFM's native ``lm_head`` intact.

     Note on parameter naming:
      In ``shared_bottleneck`` mode, attribute names ``prot_proj_in``, ``prot_proj_out``,
      ``drug_proj_in``, ``drug_proj_out``, ``prot_gate``, ``drug_gate``, and ``cross_attn`` are
      preserved so official ``Yale-CompBio/logica`` checkpoints load with zero key translation.
    """

    def __init__(
        self,
        primary_backbone: BioFMBackbone,
        partner_backbone: BioFMBackbone | None = None,
        *,
        width: int = 384,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        arch_version: ArchVersion = "v2",
        adapter_style: AdapterStyle = "shared_bottleneck",
        gate_mode: GateMode | None = None,
        p2d_attn_mode: AttentionMode = "softmax",
        d2p_attn_mode: AttentionMode | None = None,
        d2p_topk_keys: int = 32,
        use_intra_self_attn: bool | None = None,
        num_primary_fusion_layers: int | None = None,
        legacy_unidirectional_partner_esm: bool = False,
    ) -> None:
        super().__init__()
        self.primary_backbone = primary_backbone
        self.partner_backbone = partner_backbone
        self.arch_version = arch_version
        self.adapter_style = adapter_style
        self.interaction_width = int(width)
        self.num_adapter_layers = int(layers)
        self.num_adapter_heads = int(heads)
        self._legacy_unidirectional_partner_esm = legacy_unidirectional_partner_esm

        if gate_mode is None:
            gate_mode = "v1_sigmoid" if arch_version == "v1_compat" else "tanh_zero"
        if d2p_attn_mode is None:
            d2p_attn_mode = "softmax" if arch_version == "v1_compat" else "sigmoid"
        if use_intra_self_attn is None:
            use_intra_self_attn = arch_version == "v2" and adapter_style == "shared_bottleneck"
        if num_primary_fusion_layers is None:
            num_primary_fusion_layers = 1

        self.gate_mode = gate_mode
        self.num_primary_fusion_layers = int(num_primary_fusion_layers)
        self.prot_layer_mixer = (
            MultiLayerHiddenMixer(self.num_primary_fusion_layers)
            if self.num_primary_fusion_layers > 1
            else None
        )

        p_dim = self.primary_backbone.hidden_size

        if self.partner_backbone is None:
            # Single-BioFM native-logit fine-tuning mode
            self.unimodal_adapter = UnimodalGatedAdapter(
                hidden_size=p_dim,
                width=width,
                heads=heads,
                layers=layers,
                dropout=dropout,
                gate_mode=gate_mode,
            )
            self.cross_attn = None
            self.cross_layers_a = None
            self.cross_layers_b = None
        elif adapter_style == "native_additional_layers":
            # Direct native-width cross-attention stack (compatible with ydchen/logica-tcr)
            self.unimodal_adapter = None
            self.cross_attn = None
            d_dim = self.partner_backbone.hidden_size
            p_heads = getattr(getattr(self.primary_backbone, "config", None), "num_attention_heads", heads)
            d_heads = getattr(getattr(self.partner_backbone, "config", None), "num_attention_heads", heads)
            self.cross_layers_a = nn.ModuleList(
                [
                    HeterogeneousCrossAttentionLayer(
                        hidden_size=p_dim,
                        num_heads=int(p_heads),
                        dropout=dropout,
                        cross_kv_dim=d_dim,
                    )
                    for _ in range(layers)
                ]
            )
            if legacy_unidirectional_partner_esm and hasattr(self.partner_backbone, "model"):
                from transformers.models.esm.modeling_esm import EsmLayer

                esm_cfg = self.partner_backbone.base_encoder.config
                self._esm_layer_takes_pos_emb = (
                    "position_embeddings" in inspect.signature(EsmLayer.forward).parameters
                )
                self.cross_layers_b = nn.ModuleList([EsmLayer(esm_cfg) for _ in range(layers)])
            else:
                self._esm_layer_takes_pos_emb = False
                self.cross_layers_b = nn.ModuleList(
                    [
                        HeterogeneousCrossAttentionLayer(
                            hidden_size=d_dim,
                            num_heads=int(d_heads),
                            dropout=dropout,
                            cross_kv_dim=p_dim,
                        )
                        for _ in range(layers)
                    ]
                )
        else:
            # Canonical shared-bottleneck Gated Cross-Attention (Eq. 5)
            self.unimodal_adapter = None
            self.cross_layers_a = None
            self.cross_layers_b = None
            d_dim = self.partner_backbone.hidden_size

            self.prot_proj_in = nn.Linear(p_dim, width, bias=False)
            self.prot_proj_out = nn.Linear(width, p_dim, bias=False)
            self.drug_proj_in = nn.Linear(d_dim, width, bias=False)
            self.drug_proj_out = nn.Linear(width, d_dim, bias=False)

            if gate_mode == "v1_sigmoid":
                self.prot_gate = nn.Parameter(torch.tensor([-6.0]))
                self.drug_gate = nn.Parameter(torch.tensor([-6.0]))
                for proj in (
                    self.prot_proj_in,
                    self.prot_proj_out,
                    self.drug_proj_in,
                    self.drug_proj_out,
                ):
                    self._init_partial_identity(proj)
            elif gate_mode == "layerscale_zero":
                self.prot_gate = nn.Parameter(torch.zeros(p_dim))
                self.drug_gate = nn.Parameter(torch.zeros(d_dim))
                for proj in (
                    self.prot_proj_in,
                    self.prot_proj_out,
                    self.drug_proj_in,
                    self.drug_proj_out,
                ):
                    nn.init.orthogonal_(proj.weight)
            elif gate_mode == "tanh_zero":
                self.prot_gate = nn.Parameter(torch.zeros(1))
                self.drug_gate = nn.Parameter(torch.zeros(1))
                for proj in (
                    self.prot_proj_in,
                    self.prot_proj_out,
                    self.drug_proj_in,
                    self.drug_proj_out,
                ):
                    nn.init.orthogonal_(proj.weight)
            elif gate_mode == "none":
                self.register_parameter("prot_gate", None)
                self.register_parameter("drug_gate", None)
                for proj in (
                    self.prot_proj_in,
                    self.prot_proj_out,
                    self.drug_proj_in,
                    self.drug_proj_out,
                ):
                    self._init_partial_identity(proj)
            else:
                raise ValueError(f"Unsupported gate_mode: {gate_mode!r}")

            self.cross_attn = BidirectionalCrossAttention(
                width=width,
                heads=heads,
                layers=layers,
                dropout=dropout,
                p2d_attn_mode=p2d_attn_mode,
                d2p_attn_mode=d2p_attn_mode,
                d2p_topk_keys=d2p_topk_keys,
                use_intra_self_attn=use_intra_self_attn,
            )

    @property
    def is_dual(self) -> bool:
        """True when coupling two BioFMs; False in single-BioFM adaptation mode."""
        return self.partner_backbone is not None

    @staticmethod
    def _init_partial_identity(layer: nn.Linear) -> None:
        with torch.no_grad():
            layer.weight.zero_()
            diag = min(layer.out_features, layer.in_features)
            layer.weight[:diag, :diag] = torch.eye(diag)

    def _gate_scale(self, gate_param: torch.Tensor | None) -> torch.Tensor | float:
        if gate_param is None or self.gate_mode == "none":
            return 1.0
        if self.gate_mode == "v1_sigmoid":
            return torch.sigmoid(gate_param)
        return torch.tanh(gate_param)

    # =========================================================================
    # Factory Constructors (Hugging Face IDs, Custom BioFMs & Pretrained Ckpts)
    # =========================================================================

    @classmethod
    def from_pretrained(
        cls,
        primary: str | BioFMBackbone | nn.Module,
        partner: str | BioFMBackbone | nn.Module | None = None,
        *,
        width: int = 384,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        arch_version: ArchVersion = "v2",
        adapter_style: AdapterStyle = "shared_bottleneck",
        gate_mode: GateMode | None = None,
        d2p_attn_mode: AttentionMode | None = None,
        use_intra_self_attn: bool | None = None,
        num_primary_fusion_layers: int | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        checkpoint: str | Path | None = None,
    ) -> "LogiCA":
        """Constructs a :class:`LogiCA` model from one or two BioFM identifiers or modules.

        Examples::

            # 1. Protein + Small-Molecule Ligand (ESM-2 + SELFormer)
            model = LogiCA.from_pretrained(
                "facebook/esm2_t30_150M_UR50D",
                "HUBioDataLab/SELFormer",
            )

            # 2. TCR + Peptide Epitope (TCRLang + ESM-2)
            model = LogiCA.from_pretrained(
                "tcrlang-paired",
                "facebook/esm2_t12_35M_UR50D",
            )

            # 3. Single-BioFM Native-Logit Fine-Tuning (partner=None)
            model = LogiCA.from_pretrained("facebook/esm2_t12_35M_UR50D")
        """
        primary_bb = load_biofm(primary, cache_dir=cache_dir, local_files_only=local_files_only)
        partner_bb = (
            load_biofm(partner, cache_dir=cache_dir, local_files_only=local_files_only)
            if partner is not None
            else None
        )
        inst = cls(
            primary_bb,
            partner_bb,
            width=width,
            heads=heads,
            layers=layers,
            dropout=dropout,
            arch_version=arch_version,
            adapter_style=adapter_style,
            gate_mode=gate_mode,
            d2p_attn_mode=d2p_attn_mode,
            use_intra_self_attn=use_intra_self_attn,
            num_primary_fusion_layers=num_primary_fusion_layers,
        )
        if checkpoint is not None:
            inst.load_checkpoint(checkpoint)
        return inst

    @classmethod
    def from_checkpoint(
        cls,
        repo_or_path: str | Path = YALE_PROTEIN_LIGAND_HF_REPO,
        *,
        scale: str = "150m",
        subfolder: str | None = None,
        primary: str | BioFMBackbone | None = None,
        partner: str | BioFMBackbone | None = None,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
    ) -> "LogiCA":
        """Loads an official Hugging Face or local pretrained LogiCA checkpoint.

        Supported presets:
          - ``LogiCA.from_checkpoint("Yale-CompBio/logica", scale="150m")``
            (scales: ``"8m"``, ``"35m"``, ``"150m"``, ``"650m"``)
          - ``LogiCA.from_checkpoint("ydchen/logica-tcr", subfolder="dual-anchored-ratio5")``
            (subfolders: ``"tcr-anchored-neg1"``, ``"tcr-anchored-neg5"``,
            ``"pep-anchored-neg1"``, ``"pep-anchored-neg5"``, ``"dual-anchored-ratio5"``)
          - ``LogiCA.from_checkpoint("/path/to/saved_checkpoint.pt")``
        """
        target = str(repo_or_path)
        local_path = Path(target)

        # Case 1: Official Yale-CompBio/logica Protein–Ligand checkpoints
        if target == YALE_PROTEIN_LIGAND_HF_REPO:
            if scale not in YALE_PROTEIN_LIGAND_SCALES:
                raise ValueError(
                    f"Unknown scale {scale!r}; choose from {sorted(YALE_PROTEIN_LIGAND_SCALES)}"
                )
            esm_id, rel_ckpt = YALE_PROTEIN_LIGAND_SCALES[scale]
            from huggingface_hub import hf_hub_download

            ckpt_file = Path(
                hf_hub_download(repo_id=YALE_PROTEIN_LIGAND_HF_REPO, filename=rel_ckpt, cache_dir=cache_dir)
            )
            payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            sd = payload.get("model_state_dict", payload)
            ckpt_width = int(sd["prot_proj_in.weight"].shape[0])
            model = cls.from_pretrained(
                primary or esm_id,
                partner or "HUBioDataLab/SELFormer",
                width=ckpt_width,
                arch_version="v1_compat",
                cache_dir=cache_dir,
            )
            model.load_v1_protein_ligand_state_dict(sd)
            return model.to(device).eval()

        # Case 2: Official ydchen/logica-tcr TCR–Peptide checkpoints
        if target == TCR_LOGICA_HF_REPO or (":" in target and not local_path.exists()):
            if ":" in target and subfolder is None:
                repo_id, sub = target.split(":", 1)
            else:
                repo_id = target
                sub = subfolder or "dual-anchored-ratio5"
            from huggingface_hub import snapshot_download

            snap_root = Path(
                snapshot_download(repo_id=repo_id, allow_patterns=[f"{sub}/*"], cache_dir=cache_dir)
            )
            ckpt_dir = snap_root / sub
            return cls._load_tcr_checkpoint_dir(
                ckpt_dir,
                primary=primary or "tcrlang-paired",
                partner=partner or "facebook/esm2_t12_35M_UR50D",
                cache_dir=cache_dir,
                device=device,
            )

        # Case 3: Local directory or file
        if local_path.is_dir():
            return cls._load_tcr_checkpoint_dir(
                local_path,
                primary=primary or "tcrlang-paired",
                partner=partner or "facebook/esm2_t12_35M_UR50D",
                cache_dir=cache_dir,
                device=device,
            )

        payload = torch.load(local_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "logica_config" in payload:
            cfg = payload["logica_config"]
            model = cls.from_pretrained(
                primary or cfg["primary_id"],
                partner if partner is not None else cfg.get("partner_id"),
                width=int(cfg.get("width", 384)),
                heads=int(cfg.get("heads", 4)),
                layers=int(cfg.get("layers", 2)),
                arch_version=cfg.get("arch_version", "v2"),
                adapter_style=cfg.get("adapter_style", "shared_bottleneck"),
                gate_mode=cfg.get("gate_mode"),
                cache_dir=cache_dir,
            )
            model.load_state_dict(payload["model_state_dict"], strict=False)
            return model.to(device).eval()

        # Fallback: raw v1 protein-ligand checkpoint file
        sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        ckpt_width = int(sd["prot_proj_in.weight"].shape[0])
        esm_id, _ = YALE_PROTEIN_LIGAND_SCALES.get(scale, YALE_PROTEIN_LIGAND_SCALES["150m"])
        model = cls.from_pretrained(
            primary or esm_id,
            partner or "HUBioDataLab/SELFormer",
            width=ckpt_width,
            arch_version="v1_compat",
            cache_dir=cache_dir,
        )
        model.load_v1_protein_ligand_state_dict(sd)
        return model.to(device).eval()

    @classmethod
    def _load_tcr_checkpoint_dir(
        cls,
        ckpt_dir: Path,
        *,
        primary: str | BioFMBackbone,
        partner: str | BioFMBackbone,
        cache_dir: str | None,
        device: str | torch.device,
    ) -> "LogiCA":
        weights_file = None
        for fname in ("pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"):
            cand = ckpt_dir / fname
            if cand.is_file():
                weights_file = cand
                break
        if weights_file is None:
            raise FileNotFoundError(f"No weights file found inside {ckpt_dir}")

        if weights_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            raw_sd = load_file(str(weights_file))
        else:
            raw_sd = torch.load(weights_file, map_location="cpu", weights_only=False)
            if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
                raw_sd = raw_sd["model_state_dict"]

        stripped = {
            (k[len("logicl.") :] if k.startswith("logicl.") else k): v for k, v in raw_sd.items()
        }
        legacy_b = any(k.startswith("cross_layers_b.0.attention.self.") for k in stripped)

        primary_bb = load_biofm(primary, cache_dir=cache_dir)
        partner_bb = load_biofm(partner, cache_dir=cache_dir)
        model = cls(
            primary_bb,
            partner_bb,
            layers=2,
            adapter_style="native_additional_layers",
            gate_mode="none",
            legacy_unidirectional_partner_esm=legacy_b,
        )

        # Map encoder_a / encoder_b / cross_layers_* weights cleanly
        remapped: dict[str, torch.Tensor] = {}
        for k, v in stripped.items():
            if k.startswith("encoder_a.AbLang."):
                remapped["primary_backbone." + k[len("encoder_a.") :]] = v
            elif k.startswith("encoder_a."):
                remapped["primary_backbone.model." + k[len("encoder_a.") :]] = v
            elif k.startswith("encoder_b."):
                remapped["partner_backbone.model." + k[len("encoder_b.") :]] = v
            elif k.startswith(("cross_layers_a.", "cross_layers_b.")):
                remapped[k] = v
        model.load_state_dict(remapped, strict=False)
        return model.to(device).eval()

    def load_v1_protein_ligand_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Loads a ``Yale-CompBio/logica`` state dict into ``shared_bottleneck`` mode."""
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("esm."):
                remapped["primary_backbone.model." + k[len("esm.") :]] = v
            elif k.startswith("drug_encoder."):
                remapped["partner_backbone.model." + k[len("drug_encoder.") :]] = v
            else:
                remapped[k] = v
        incompatible = self.load_state_dict(remapped, strict=False)
        critical = ("prot_gate", "drug_gate", "prot_proj_", "drug_proj_", "cross_attn.")
        missing_crit = [k for k in incompatible.missing_keys if k.startswith(critical)]
        if missing_crit:
            raise RuntimeError(f"Missing critical LogiCA adapter keys: {missing_crit}")

    def save_checkpoint(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> None:
        """Saves model weights and ``logica_config`` metadata for reproducible reloading."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "logica_config": {
                "primary_id": self.primary_backbone.model_id,
                "partner_id": self.partner_backbone.model_id if self.partner_backbone else None,
                "width": self.interaction_width,
                "heads": self.num_adapter_heads,
                "layers": self.num_adapter_layers,
                "arch_version": self.arch_version,
                "adapter_style": self.adapter_style,
                "gate_mode": self.gate_mode,
            },
            "model_state_dict": self.state_dict(),
        }
        if extra_metadata:
            payload["metadata"] = extra_metadata
        torch.save(payload, out_path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads adapter/backbone weights from a local checkpoint file."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        sd = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if any(k.startswith("esm.") or k == "prot_proj_in.weight" for k in sd):
            if "primary_backbone.model.lm_head.dense.weight" not in sd and any(
                k.startswith("prot_proj_in") for k in sd
            ):
                self.load_v1_protein_ligand_state_dict(sd)
                return payload if isinstance(payload, dict) else {}
        self.load_state_dict(sd, strict=False)
        return payload if isinstance(payload, dict) else {}

    # =========================================================================
    # Unimodal Encoding & Cross-Modal / Unimodal Native-Head Forward Passes
    # =========================================================================

    def encode_primary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes primary sequences once, returning ``(last_hidden, adapter_query_hidden)``."""
        need_multi = self.prot_layer_mixer is not None
        last_h, all_h = self.primary_backbone.encode_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_multi,
        )
        if self.prot_layer_mixer is not None and all_h is not None:
            qkv_h = self.prot_layer_mixer(all_h)
        else:
            qkv_h = last_h
        return last_h, qkv_h

    def encode_partner(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes partner context sequences once, returning ``last_hidden``."""
        if self.partner_backbone is None:
            raise RuntimeError("`encode_partner` called on a single-BioFM LogiCA model.")
        last_h, _ = self.partner_backbone.encode_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return last_h

    def _run_native_additional_layers(
        self,
        p_hidden: torch.Tensor,
        p_attn_mask: torch.Tensor,
        d_hidden: torch.Tensor,
        d_attn_mask: torch.Tensor,
        bypass_cross_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cross_layers_a is not None and self.cross_layers_b is not None
        ha, hb = p_hidden, d_hidden
        p_pad = p_attn_mask == 0
        d_pad = d_attn_mask == 0

        for layer_a, layer_b in zip(self.cross_layers_a, self.cross_layers_b, strict=True):
            peer_b = None if bypass_cross_attention else hb
            peer_a = None if bypass_cross_attention else ha
            next_a = layer_a(ha, p_pad, peer_b, None if bypass_cross_attention else d_pad)

            if isinstance(layer_b, HeterogeneousCrossAttentionLayer):
                next_b = layer_b(hb, d_pad, peer_a, None if bypass_cross_attention else p_pad)
            else:
                neg_inf = -1e4 if hb.dtype == torch.float16 else -1e9
                ext_self = (1.0 - d_attn_mask.to(hb.dtype)).unsqueeze(1).unsqueeze(2) * neg_inf
                ext_peer = (
                    (1.0 - p_attn_mask.to(hb.dtype)).unsqueeze(1).unsqueeze(2) * neg_inf
                    if peer_a is not None
                    else None
                )
                kwargs: dict[str, Any] = {}
                if getattr(self, "_esm_layer_takes_pos_emb", False) and hasattr(
                    self.partner_backbone, "base_encoder"
                ):
                    base_enc = self.partner_backbone.base_encoder
                    if hasattr(base_enc, "rotary_embeddings"):
                        pos_ids = torch.arange(hb.shape[1], device=hb.device).unsqueeze(0)
                        kwargs["position_embeddings"] = base_enc.rotary_embeddings(hb, pos_ids)
                out_b = layer_b(
                    hb,
                    ext_self,
                    encoder_hidden_states=peer_a,
                    encoder_attention_mask=ext_peer,
                    **kwargs,
                )
                next_b = out_b[0] if isinstance(out_b, tuple) else out_b
            ha, hb = next_a, next_b
        return ha, hb

    def forward_from_hidden(
        self,
        primary_last_hidden: torch.Tensor,
        primary_attention_mask: torch.Tensor,
        partner_hidden: torch.Tensor | None = None,
        partner_attention_mask: torch.Tensor | None = None,
        *,
        primary_adapter_input: torch.Tensor | None = None,
        primary_interface_mask: torch.Tensor | None = None,
    ) -> LogiCAOutput:
        """Runs the Gated Cross-Attention (or single-BioFM self-adapter) + native LM heads."""
        if self.partner_backbone is None or partner_hidden is None:
            p_apo = self.primary_backbone.logits_from_hidden(primary_last_hidden)
            if self.unimodal_adapter is not None:
                p_ctx = self.unimodal_adapter(primary_last_hidden, primary_attention_mask == 0)
                p_holo = self.primary_backbone.logits_from_hidden(p_ctx)
            else:
                p_ctx = primary_last_hidden
                p_holo = p_apo
            return LogiCAOutput(
                primary_holo_logits=p_holo,
                primary_apo_logits=p_apo,
                primary_contextual_hidden=p_ctx,
            )

        if partner_attention_mask is None:
            partner_attention_mask = torch.ones(
                partner_hidden.shape[:2], dtype=torch.long, device=partner_hidden.device
            )

        if self.adapter_style == "native_additional_layers":
            p_ctx, d_ctx = self._run_native_additional_layers(
                primary_last_hidden,
                primary_attention_mask,
                partner_hidden,
                partner_attention_mask,
                bypass_cross_attention=False,
            )
            p_apo_h, d_apo_h = self._run_native_additional_layers(
                primary_last_hidden,
                primary_attention_mask,
                partner_hidden,
                partner_attention_mask,
                bypass_cross_attention=True,
            )
            return LogiCAOutput(
                primary_holo_logits=self.primary_backbone.logits_from_hidden(p_ctx),
                primary_apo_logits=self.primary_backbone.logits_from_hidden(p_apo_h),
                partner_holo_logits=self.partner_backbone.logits_from_hidden(d_ctx),
                partner_apo_logits=self.partner_backbone.logits_from_hidden(d_apo_h),
                primary_contextual_hidden=p_ctx,
                partner_contextual_hidden=d_ctx,
            )

        assert self.cross_attn is not None
        p_in = primary_last_hidden if primary_adapter_input is None else primary_adapter_input
        z_p0 = self.prot_proj_in(p_in)
        z_d0 = self.drug_proj_in(partner_hidden)
        z_pN, z_dN = self.cross_attn(
            z_p0,
            z_d0,
            primary_attention_mask == 0,
            partner_attention_mask == 0,
            primary_interface_mask=primary_interface_mask,
        )
        p_scale = self._gate_scale(self.prot_gate)
        d_scale = self._gate_scale(self.drug_gate)

        p_ctx = primary_last_hidden + p_scale * self.prot_proj_out(z_pN - z_p0)
        d_ctx = partner_hidden + d_scale * self.drug_proj_out(z_dN - z_d0)

        return LogiCAOutput(
            primary_holo_logits=self.primary_backbone.logits_from_hidden(p_ctx),
            primary_apo_logits=self.primary_backbone.logits_from_hidden(primary_last_hidden),
            partner_holo_logits=self.partner_backbone.logits_from_hidden(d_ctx),
            partner_apo_logits=self.partner_backbone.logits_from_hidden(partner_hidden),
            primary_contextual_hidden=p_ctx,
            partner_contextual_hidden=d_ctx,
        )

    def forward(
        self,
        primary_input_ids: torch.Tensor,
        primary_attention_mask: torch.Tensor | None = None,
        partner_input_ids: torch.Tensor | None = None,
        partner_attention_mask: torch.Tensor | None = None,
    ) -> LogiCAOutput:
        """Full end-to-end pass from token IDs to native vocabulary holo & apo logits."""
        if primary_attention_mask is None:
            primary_attention_mask = (primary_input_ids != self.primary_backbone.pad_token_id).long()
        p_last, p_qkv = self.encode_primary(primary_input_ids, primary_attention_mask)

        if self.partner_backbone is None or partner_input_ids is None:
            return self.forward_from_hidden(
                p_last,
                primary_attention_mask,
                primary_adapter_input=p_qkv,
            )

        if partner_attention_mask is None:
            partner_attention_mask = (partner_input_ids != self.partner_backbone.pad_token_id).long()
        d_last = self.encode_partner(partner_input_ids, partner_attention_mask)
        return self.forward_from_hidden(
            p_last,
            primary_attention_mask,
            d_last,
            partner_attention_mask,
            primary_adapter_input=p_qkv,
        )

    # =========================================================================
    # Differential Optimizer & Late-Layer LoRA Configuration
    # =========================================================================

    def configure_finetuning_optimizer(
        self,
        *,
        base_lora_lr: float = 1e-4,
        adapter_lr_multiplier: float = 5.0,
        weight_decay: float = 1e-2,
        top_k_primary_lora: int = 4,
        top_k_partner_lora: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        extra_modules: Sequence[nn.Module] = (),
    ) -> list[dict[str, Any]]:
        """Freezes early BioFM blocks, injects late-layer LoRA, and builds differential AdamW groups."""
        for param in self.parameters():
            param.requires_grad_(False)

        lora_params: list[nn.Parameter] = []
        lora_params.extend(
            inject_late_layer_lora(
                self.primary_backbone,
                top_k_layers=top_k_primary_lora,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        )
        if self.partner_backbone is not None and top_k_partner_lora > 0:
            lora_params.extend(
                inject_late_layer_lora(
                    self.partner_backbone,
                    top_k_layers=top_k_partner_lora,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
            )

        adapter_modules: list[nn.Module] = []
        if self.unimodal_adapter is not None:
            adapter_modules.append(self.unimodal_adapter)
        if self.cross_attn is not None:
            adapter_modules.extend(
                [
                    self.prot_proj_in,
                    self.prot_proj_out,
                    self.drug_proj_in,
                    self.drug_proj_out,
                    self.cross_attn,
                ]
            )
        if self.cross_layers_a is not None:
            adapter_modules.append(self.cross_layers_a)
        if self.cross_layers_b is not None:
            adapter_modules.append(self.cross_layers_b)
        if self.prot_layer_mixer is not None:
            adapter_modules.append(self.prot_layer_mixer)
        adapter_modules.extend(extra_modules)

        adapter_decay: list[nn.Parameter] = []
        adapter_no_decay: list[nn.Parameter] = []

        for gate_attr in ("prot_gate", "drug_gate"):
            g = getattr(self, gate_attr, None)
            if isinstance(g, nn.Parameter):
                g.requires_grad_(True)
                adapter_no_decay.append(g)

        for mod in adapter_modules:
            for name, param in mod.named_parameters():
                param.requires_grad_(True)
                if param.ndim < 2 or any(tag in name for tag in ("norm", "bias", "gate", "scale")):
                    adapter_no_decay.append(param)
                else:
                    adapter_decay.append(param)

        groups: list[dict[str, Any]] = []
        if adapter_decay:
            groups.append(
                {
                    "params": adapter_decay,
                    "lr": base_lora_lr * adapter_lr_multiplier,
                    "weight_decay": weight_decay,
                    "name": "adapter_decay",
                }
            )
        if adapter_no_decay:
            groups.append(
                {
                    "params": adapter_no_decay,
                    "lr": base_lora_lr * adapter_lr_multiplier,
                    "weight_decay": 0.0,
                    "name": "adapter_gates_and_norms",
                }
            )
        if lora_params:
            groups.append(
                {
                    "params": lora_params,
                    "lr": base_lora_lr,
                    "weight_decay": weight_decay,
                    "name": "late_layer_lora",
                }
            )
        return groups

    # =========================================================================
    # High-Level Convenience Readouts (Variant Ranking, Dependency Map, Gibbs)
    # =========================================================================

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def rank_variants(
        self,
        wild_type: str,
        variants: Sequence[str],
        context: str | None = None,
        **kwargs: Any,
    ):
        """Scores and ranks mutant candidates ``variants`` relative to ``wild_type`` under ``context``."""
        from .ranking import VariantRanker

        return VariantRanker(self).rank_variants(
            wild_type=wild_type, variants=variants, context=context, **kwargs
        )

    def dependency_map(
        self,
        primary_sequence: str,
        partner_sequence: str | None = None,
        **kwargs: Any,
    ):
        """Computes a cross-modal (or intra-sequence when ``partner_sequence=None``) dependency map."""
        from .dependency import DependencyMapper

        mapper = DependencyMapper(self)
        if partner_sequence is None or not self.is_dual:
            return mapper.compute_intra_dependency_map(primary_sequence, **kwargs)
        return mapper.compute_cross_dependency_map(primary_sequence, partner_sequence, **kwargs)

    def detect_kl_hotspots(
        self,
        primary_sequence: str,
        partner_sequence: str,
        top_k: int = 8,
        alphabet: Sequence[str] = CANONICAL_AAS,
    ):
        """Annotation-free zero-shot interface hotspot detection via ``D_KL(pi(.|x,y) || pi(.|x))``."""
        from .dependency import DependencyMapper

        return DependencyMapper(self).detect_kl_hotspots(
            primary_sequence, partner_sequence, top_k=top_k, alphabet=alphabet
        )

    def gibbs_sample(
        self,
        seed_sequence: str,
        context: str | None = None,
        **kwargs: Any,
    ):
        """Runs decoder-free Coordinate Gibbs / MCMC sequence design over the native token head."""
        from .gibbs import GibbsSampler

        return GibbsSampler(self).sample_sequences(
            seed_sequence=seed_sequence, context=context, **kwargs
        )
