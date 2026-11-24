"""Model-agnostic Biological Foundation Model (BioFM) wrappers and loaders.

LogiCA operates directly on the native token distributions of pretrained BioFMs.
Any encoder-only or masked biological language model that maps token IDs to
per-position hidden states ``H in R^{L x D}`` and projects hidden states through a
native vocabulary head ``lm_head: R^D -> R^{|V|}`` can plug into LogiCA without
modifying its vocabulary or tokenizer:

  - **Hugging Face Masked LMs**: Protein LMs (``facebook/esm2_*``, ``facebook/esm1v_*``,
    ``Rostlab/prot_bert``), Chemical LMs (``HUBioDataLab/SELFormer``,
    ``DeepChem/ChemBERTa-*``), DNA/RNA LMs (``InstaDeepAI/nucleotide-transformer-*``), etc.
  - **AbLang2 / TCRLang Repertoire LMs**: ``tcrlang-paired`` / ``ablang2-paired``
    (paired ``CDR3beta|CDR3alpha`` or ``VH|VL`` chains).
  - **Custom PyTorch ``nn.Module`` Backbones**: Any user-supplied encoder + native head
    + tokenizer wrapped via :func:`load_biofm` or :class:`CustomBioFMBackbone`.
"""

from __future__ import annotations

import abc
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch
from torch import nn

CANONICAL_AAS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")


def _smiles_to_selfies_if_needed(text: str) -> str:
    """Converts a SMILES string to SELFIES when needed; leaves existing SELFIES intact."""
    s = str(text).strip()
    if not s or (s.startswith("[") and "]" in s):
        return s
    try:
        import selfies as sf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Converting SMILES strings for SELFormer requires `pip install selfies`."
        ) from exc
    encoded = sf.encoder(s)
    if not encoded:
        raise ValueError(f"Could not encode SMILES string into SELFIES: {s!r}")
    return str(encoded)


class BioFMBackbone(nn.Module, abc.ABC):
    """Abstract model-agnostic interface for any pretrained Biological Foundation Model."""

    model_id: str
    tokenizer: Any

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """Token hidden-state dimensionality ``D``."""

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Native token vocabulary size ``|V|``."""

    @property
    @abc.abstractmethod
    def mask_token_id(self) -> int:
        """Token ID used to mask positions for conditional log-likelihood evaluation."""

    @property
    def pad_token_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", 0)
        return 0 if pad is None else int(pad)

    @property
    def special_token_ids(self) -> set[int]:
        ids: set[int] = set()
        for attr in ("all_special_ids",):
            vals = getattr(self.tokenizer, attr, None)
            if vals is not None:
                ids.update(int(v) for v in vals if v is not None)
        for attr in (
            "cls_token_id",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "sep_token_id",
            "mask_token_id",
        ):
            val = getattr(self.tokenizer, attr, None)
            if val is not None and int(val) >= 0:
                ids.add(int(val))
        return ids

    @abc.abstractmethod
    def tokenize(
        self,
        sequences: str | Sequence[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenizes raw biological sequence(s) into ``input_ids``, ``attention_mask``, and ``special_tokens_mask``."""

    @abc.abstractmethod
    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Runs the pretrained encoder and returns ``(last_hidden_state, all_hidden_states)``."""

    @abc.abstractmethod
    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects contextualized or unimodal hidden states through the native token head."""

    def get_transformer_layers(self) -> Sequence[nn.Module]:
        """Discovers transformer blocks inside the encoder for late-layer LoRA adaptation."""
        return discover_transformer_layers(self)

    def token_to_id(self, token: str) -> int:
        """Maps a vocabulary token string (e.g. amino acid ``'A'``) to its integer token ID."""
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                return tid
        if hasattr(self.tokenizer, "token_to_id"):
            tid = self.tokenizer.token_to_id(token)
            if isinstance(tid, int) and tid >= 0:
                return tid
        raise KeyError(f"Token {token!r} not found in vocabulary of {self.model_id!r}")

    def id_to_token(self, token_id: int) -> str:
        """Maps an integer token ID back to its string symbol."""
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            tok = self.tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(tok, str):
                return tok
        if hasattr(self.tokenizer, "id_to_token"):
            return str(self.tokenizer.id_to_token(int(token_id)))
        return str(token_id)

    def canonical_aa_token_ids(self, alphabet: Sequence[str] = CANONICAL_AAS) -> list[int]:
        """Returns the native vocabulary token IDs corresponding to the 20 canonical amino acids."""
        return [self.token_to_id(aa) for aa in alphabet]

    def build_special_tokens_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns a boolean mask ``(B, L)`` marking special/padding/separator tokens ``True``."""
        spec = torch.zeros_like(input_ids, dtype=torch.bool)
        if attention_mask is not None:
            spec |= attention_mask == 0
        for sp_id in self.special_token_ids:
            if sp_id != self.mask_token_id:
                spec |= input_ids == sp_id
        return spec


def discover_transformer_layers(module: nn.Module) -> Sequence[nn.Module]:
    """Heuristically locates the ordered ``ModuleList`` of transformer blocks in any BioFM."""
    candidate_paths: tuple[tuple[str, ...], ...] = (
        ("model", "esm", "encoder", "layer"),
        ("model", "roberta", "encoder", "layer"),
        ("model", "bert", "encoder", "layer"),
        ("model", "deberta", "encoder", "layer"),
        ("model", "encoder", "layer"),
        ("model", "layers"),
        ("esm", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("AbLang", "AbRep", "encoder_blocks"),
        ("AbLang", "AbRep", "Transformer", "blocks"),
        ("AbLang", "AbRep", "blocks"),
        ("layers",),
        ("blocks",),
    )
    for path in candidate_paths:
        curr: Any = module
        for attr in path:
            curr = getattr(curr, attr, None)
            if curr is None:
                break
        if isinstance(curr, (nn.ModuleList, list, tuple)) and len(curr) > 0:
            return curr  # type: ignore[return-value]
    return ()


class HuggingFaceMaskedLMBackbone(BioFMBackbone):
    """Wraps any Hugging Face ``AutoModelForMaskedLM`` + ``AutoTokenizer`` BioFM."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "hf-masked-lm",
        preprocess_fn: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.preprocess_fn = preprocess_fn

        if not hasattr(self.model, "lm_head"):
            head = self._resolve_lm_head(self.model)
            if head is None:
                raise ValueError(
                    f"Model {model_id!r} does not expose a native masked-LM head "
                    "(`lm_head`, `cls`, or ` get_output_embeddings()`)."
                )
            self._lm_head_module = head
        else:
            self._lm_head_module = self.model.lm_head

    @staticmethod
    def _resolve_lm_head(model: nn.Module) -> nn.Module | None:
        for attr in ("lm_head", "cls", "predictions", "generator_lm_head"):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Module):
                return mod
        if hasattr(model, "get_output_embeddings"):
            out_emb = model.get_output_embeddings()
            if isinstance(out_emb, nn.Module):
                return out_emb
        return None

    @property
    def base_encoder(self) -> nn.Module:
        for attr in ("esm", "roberta", "bert", "deberta", "modernbert", "base_model"):
            sub = getattr(self.model, attr, None)
            if isinstance(sub, nn.Module) and sub is not self.model:
                return sub
        return self.model

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def mask_token_id(self) -> int:
        mid = getattr(self.tokenizer, "mask_token_id", None)
        if mid is None:
            raise ValueError(f"Tokenizer for {self.model_id!r} has no `mask_token_id`.")
        return int(mid)

    def tokenize(
        self,
        sequences: str | Sequence[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        if self.preprocess_fn is not None:
            seq_list = [self.preprocess_fn(s) for s in seq_list]
        kwargs: dict[str, Any] = {
            "padding": padding,
            "truncation": truncation if max_length is not None else False,
            "return_tensors": "pt",
            "return_special_tokens_mask": True,
        }
        if max_length is not None:
            kwargs["max_length"] = int(max_length)
        enc = self.tokenizer(seq_list, **kwargs)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        spec_mask = enc.get("special_tokens_mask")
        if spec_mask is None:
            spec_mask = self.build_special_tokens_mask(input_ids, attention_mask)
        else:
            spec_mask = spec_mask.bool() | (attention_mask == 0)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": spec_mask.bool(),
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        base = self.base_encoder
        if base is not self.model:
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state
            all_hidden = out.hidden_states if output_hidden_states else None
            return last_hidden, all_hidden

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        all_h = out.hidden_states
        return all_h[-1], (all_h if output_hidden_states else None)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._lm_head_module(hidden_states)


class AbLang2TokenizerAdapter:
    """Adapts an ``ablang2`` / ``tcrlang`` tokenizer to standard HuggingFace conventions."""

    def __init__(self, ablang_tokenizer: Any) -> None:
        self.tokenizer = ablang_tokenizer
        vocab: dict[str, int] = dict(getattr(ablang_tokenizer, "aa_to_token", {}))
        self.vocab = vocab
        self.id_to_tok = {int(v): str(k) for k, v in vocab.items()}
        self.pad_token_id = int(getattr(ablang_tokenizer, "pad_token", vocab.get("-", 21)))
        self.mask_token_id = int(getattr(ablang_tokenizer, "mask_token", vocab.get("*", 23)))
        self.cls_token_id = getattr(ablang_tokenizer, "start_token", vocab.get("<", None))
        self.eos_token_id = getattr(ablang_tokenizer, "end_token", vocab.get(">", None))
        self.sep_token_id = getattr(ablang_tokenizer, "sep_token", vocab.get("|", None))
        self.all_special_ids = [
            int(x)
            for x in (
                self.pad_token_id,
                self.mask_token_id,
                self.cls_token_id,
                self.eos_token_id,
                self.sep_token_id,
            )
            if x is not None
        ]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self.vocab:
            return int(self.vocab[token])
        raise KeyError(f"Unknown AbLang2 token: {token!r}")

    def convert_ids_to_tokens(self, idx: int) -> str:
        return self.id_to_tok.get(int(idx), "X")

    def __call__(
        self,
        sequences: Sequence[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        cleaned = [str(s).strip().upper() for s in sequences]
        tok_out = self.tokenizer(cleaned, pad=True, w_extra_tkns=False, device="cpu")
        ids = torch.as_tensor(tok_out, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        if truncation and max_length is not None and ids.size(1) > int(max_length):
            ids = ids[:, : int(max_length)]
        if padding == "max_length" and max_length is not None and ids.size(1) < int(max_length):
            pad_cols = torch.full(
                (ids.size(0), int(max_length) - ids.size(1)),
                self.pad_token_id,
                dtype=torch.long,
            )
            ids = torch.cat([ids, pad_cols], dim=1)

        attn = ids.ne(self.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}


class AbLang2BioFMBackbone(BioFMBackbone):
    """Wraps ``ablang2`` / ``tcrlang-paired`` / ``ablang2-paired`` repertoire foundation models."""

    def __init__(self, model_name: str = "tcrlang-paired", random_init: bool = False) -> None:
        super().__init__()
        try:
            from ablang2.load_model import load_model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"Loading {model_name!r} requires `pip install ablang2`."
            ) from exc

        clean_name = model_name.split(":", 1)[1] if model_name.startswith("ablang2:") else model_name
        ablang, raw_tok, hparams = load_model(clean_name, random_init=random_init)
        self.AbLang = ablang
        self.hparams = hparams
        self.model_id = clean_name
        self.tokenizer = AbLang2TokenizerAdapter(raw_tok)
        self.config = SimpleNamespace(
            hidden_size=int(hparams.hidden_embed_size),
            vocab_size=int(hparams.vocab_size),
            num_attention_heads=int(hparams.n_attn_heads),
            pad_token_id=int(hparams.pad_tkn),
            mask_token_id=int(hparams.mask_tkn),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.mask_token_id)

    def tokenize(
        self,
        sequences: str | Sequence[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        enc = self.tokenizer(
            seq_list, max_length=max_length, padding=padding, truncation=truncation
        )
        spec_mask = self.build_special_tokens_mask(enc["input_ids"], enc["attention_mask"])
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "special_tokens_mask": spec_mask,
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        _ = attention_mask
        reps = self.AbLang.AbRep(input_ids)
        last = reps.last_hidden_states
        return last, ((last,) if output_hidden_states else None)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.AbLang.AbHead(hidden_states)


class CustomBioFMBackbone(BioFMBackbone):
    """Wraps any user-defined PyTorch ``nn.Module`` encoder + native token head + tokenizer.

    Useful both for custom proprietary BioFMs (e.g., custom RNA, codon, or structure-aware
    masked language models) and for fast lightweight unit testing without network access.
    """

    def __init__(
        self,
        encoder: nn.Module,
        lm_head: nn.Module,
        tokenizer: Any,
        hidden_size: int,
        vocab_size: int,
        model_id: str = "custom-biofm",
        preprocess_fn: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self._hidden_size = int(hidden_size)
        self._vocab_size = int(vocab_size)
        self.model_id = model_id
        self.preprocess_fn = preprocess_fn
        self.config = SimpleNamespace(
            hidden_size=self._hidden_size,
            vocab_size=self._vocab_size,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
            mask_token_id=getattr(tokenizer, "mask_token_id", 1),
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.mask_token_id)

    def tokenize(
        self,
        sequences: str | Sequence[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        if self.preprocess_fn is not None:
            seq_list = [self.preprocess_fn(s) for s in seq_list]
        enc = self.tokenizer(
            seq_list,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", (input_ids != self.pad_token_id).long())
        spec_mask = enc.get("special_tokens_mask")
        if spec_mask is None:
            spec_mask = self.build_special_tokens_mask(input_ids, attention_mask)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": spec_mask.bool(),
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, tuple):
            last_h = out[0]
            all_h = out[1] if (output_hidden_states and len(out) > 1) else ((last_h,) if output_hidden_states else None)
            return last_h, all_h
        if hasattr(out, "last_hidden_state"):
            all_h = getattr(out, "hidden_states", None)
            if output_hidden_states and all_h is None:
                all_h = (out.last_hidden_state,)
            return out.last_hidden_state, (all_h if output_hidden_states else None)
        return out, ((out,) if output_hidden_states else None)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def load_biofm(
    model_or_id: str | BioFMBackbone | nn.Module,
    *,
    tokenizer: Any | None = None,
    lm_head: nn.Module | None = None,
    hidden_size: int | None = None,
    vocab_size: int | None = None,
    backbone_type: str = "auto",
    auto_selfies: bool | None = None,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> BioFMBackbone:
    """Universal factory that loads or wraps any pretrained BioFM into a :class:`BioFMBackbone`.

    Args:
      model_or_id: Any of:
        - A Hugging Face Hub ID or local directory (e.g. ``"facebook/esm2_t30_150M_UR50D"``,
          ``"HUBioDataLab/SELFormer"``, ``"InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"``),
        - An AbLang2 / TCRLang repertoire identifier (e.g. ``"tcrlang-paired"``,
          ``"ablang2:tcrlang-paired"``, ``"ablang2-paired"``),
        - An already-constructed :class:`BioFMBackbone`,
        - A pre-instantiated Hugging Face ``PreTrainedModel`` or custom ``nn.Module``.
      tokenizer: Optional tokenizer when passing a custom ``nn.Module``.
      lm_head: Optional native token head when passing a custom ``nn.Module`` encoder.
      hidden_size: Required only if ``model_or_id`` is a custom ``nn.Module`` without ``.config``.
      vocab_size: Required only if ``model_or_id`` is a custom ``nn.Module`` without ``.config``.
      backbone_type: ``"auto"``, ``"hf"``, ``"selformer"``, ``"ablang2"``, or ``"custom"``.
      auto_selfies: Automatically convert SMILES inputs to SELFIES (auto-enabled for SELFormer).
      cache_dir: Optional Hugging Face download cache directory.
      trust_remote_code: Passed to ``AutoModelForMaskedLM`` / ``AutoTokenizer`` when needed.
      local_files_only: Restrict Hugging Face loading to local cache.
    """
    if isinstance(model_or_id, BioFMBackbone):
        return model_or_id

    if isinstance(model_or_id, nn.Module):
        if lm_head is not None:
            cfg = getattr(model_or_id, "config", None)
            h_dim = hidden_size or int(getattr(cfg, "hidden_size", 0))
            v_dim = vocab_size or int(getattr(cfg, "vocab_size", 0))
            if h_dim <= 0 or v_dim <= 0 or tokenizer is None:
                raise ValueError(
                    "Wrapping a custom `nn.Module` requires `tokenizer`, `hidden_size`, and `vocab_size`."
                )
            return CustomBioFMBackbone(
                encoder=model_or_id,
                lm_head=lm_head,
                tokenizer=tokenizer,
                hidden_size=h_dim,
                vocab_size=v_dim,
            )
        if tokenizer is None:
            raise ValueError("Passing an instantiated HF `PreTrainedModel` requires `tokenizer`.")
        return HuggingFaceMaskedLMBackbone(
            model=model_or_id,
            tokenizer=tokenizer,
            model_id=getattr(getattr(model_or_id, "config", None), "_name_or_path", "hf-model"),
            preprocess_fn=_smiles_to_selfies_if_needed if auto_selfies else None,
        )

    spec = str(model_or_id).strip()
    is_ablang = (
        backbone_type == "ablang2"
        or spec.startswith("ablang2:")
        or spec in {"tcrlang-paired", "ablang2-paired", "tcrlang-heavy", "tcrlang-light"}
    )
    if is_ablang:
        return AbLang2BioFMBackbone(spec)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    use_selfies = (
        auto_selfies
        if auto_selfies is not None
        else (backbone_type == "selformer" or "selformer" in spec.lower())
    )
    tok = tokenizer or AutoTokenizer.from_pretrained(
        spec,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    hf_model = AutoModelForMaskedLM.from_pretrained(
        spec,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return HuggingFaceMaskedLMBackbone(
        model=hf_model,
        tokenizer=tok,
        model_id=spec,
        preprocess_fn=_smiles_to_selfies_if_needed if use_selfies else None,
    )
