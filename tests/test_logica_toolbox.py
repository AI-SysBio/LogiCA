"""Comprehensive model-agnostic unit test suite for the `logica` library."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import torch
from torch import nn

from logica import (
    CANONICAL_AAS,
    AncestralKLAnchorLoss,
    BradleyTerryLogitLoss,
    CustomBioFMBackbone,
    InfoNCELogitLoss,
    LogiCA,
    PlackettLuceLogitLoss,
    ThermodynamicHeadlessScorer,
    apply_mutations,
    create_anchored_mutation_negatives,
    diff_mutations,
    evaluate_contact_localization,
    evaluate_landscape_gibbs,
    evaluate_ranking_metrics,
    parse_mutation_string,
    plot_dependency_map,
    plot_gibbs_design,
    plot_variant_ranking,
)


class ToyVocabularyTokenizer:
    """Minimal character/bracket tokenizer supporting arbitrary biological alphabets."""

    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.special = ("<pad>", "<mask>", "<cls>", "<eos>")
        self.all_tokens = self.special + tuple(tokens)
        self.tok_to_id = {t: i for i, t in enumerate(self.all_tokens)}
        self.id_to_tok = {i: t for i, t in enumerate(self.all_tokens)}
        self.pad_token_id = 0
        self.mask_token_id = 1
        self.cls_token_id = 2
        self.eos_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.tok_to_id[tok]

    def convert_ids_to_tokens(self, idx: int) -> str:
        return self.id_to_tok[int(idx)]

    def _split(self, seq: str) -> list[str]:
        s = seq.strip()
        if s.startswith("[") and "]" in s:
            return [part + "]" for part in s.split("]") if part]
        return list(s)

    def __call__(
        self,
        sequences: list[str],
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        rows = []
        for s in sequences:
            toks = self._split(s)
            if truncation and max_length is not None:
                toks = toks[: max(1, max_length - 2)]
            ids = [self.cls_token_id] + [self.tok_to_id.get(t, self.pad_token_id) for t in toks] + [self.eos_token_id]
            rows.append(ids)
        max_l = max(len(r) for r in rows) if padding else len(rows[0])
        input_ids = torch.zeros((len(rows), max_l), dtype=torch.long)
        attn = torch.zeros((len(rows), max_l), dtype=torch.long)
        spec = torch.ones((len(rows), max_l), dtype=torch.bool)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = 1
            if len(r) > 2:
                spec[i, 1 : len(r) - 1] = False
        return {"input_ids": input_ids, "attention_mask": attn, "special_tokens_mask": spec}


class ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return self.norm(x + self.dense(torch.matmul(attn, v)))


class ToyBioFMEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([ToyTransformerBlock(dim) for _ in range(num_layers)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        _ = attention_mask
        h = self.embed(input_ids)
        states = [h]
        for blk in self.layers:
            h = blk(h)
            states.append(h)
        return SimpleNamespace(last_hidden_state=h, hidden_states=tuple(states))


def make_heterogeneous_biofms(seed: int = 7) -> tuple[CustomBioFMBackbone, CustomBioFMBackbone]:
    """Creates two BioFMs with completely distinct vocabularies and hidden dimensions."""
    torch.manual_seed(seed)
    prot_tok = ToyVocabularyTokenizer(CANONICAL_AAS)
    chem_tokens = ("[C]", "[O]", "[N]", "[=O]", "[Ring1]", "[Branch1]", "[F]", "[Cl]")
    chem_tok = ToyVocabularyTokenizer(chem_tokens)

    prot_enc = ToyBioFMEncoder(len(prot_tok.all_tokens), dim=48, num_layers=3)
    prot_head = nn.Linear(48, len(prot_tok.all_tokens))
    prot_bb = CustomBioFMBackbone(
        encoder=prot_enc,
        lm_head=prot_head,
        tokenizer=prot_tok,
        hidden_size=48,
        vocab_size=len(prot_tok.all_tokens),
        model_id="toy-protein-fm",
    )

    chem_enc = ToyBioFMEncoder(len(chem_tok.all_tokens), dim=32, num_layers=2)
    chem_head = nn.Linear(32, len(chem_tok.all_tokens))
    chem_bb = CustomBioFMBackbone(
        encoder=chem_enc,
        lm_head=chem_head,
        tokenizer=chem_tok,
        hidden_size=32,
        vocab_size=len(chem_tok.all_tokens),
        model_id="toy-chem-fm",
    )
    return prot_bb, chem_bb


def test_dual_biofm_zero_gate_identity_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    prot_bb, chem_bb = make_heterogeneous_biofms()
    model = LogiCA.from_pretrained(
        prot_bb,
        chem_bb,
        width=32,
        heads=4,
        layers=2,
        gate_mode="tanh_zero",
        d2p_attn_mode="sigmoid",
        num_primary_fusion_layers=2,
    )
    model.eval()

    p_enc = prot_bb.tokenize(["ACDEFGHIKLMNPQRSTV"])
    d_enc = chem_bb.tokenize(["[C][O][C][=O][Ring1][F]"])
    out = model(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])

    # At initialization, tanh(0) == 0 -> holo logits equal apo logits identically!
    assert torch.allclose(out.primary_holo_logits, out.primary_apo_logits, atol=1e-6)
    assert out.partner_holo_logits is not None and out.partner_apo_logits is not None
    assert torch.allclose(out.partner_holo_logits, out.partner_apo_logits, atol=1e-6)

    # Open gates and verify context sensitivity + checkpoint save/load
    with torch.no_grad():
        model.prot_gate.fill_(0.8)
        model.drug_gate.fill_(0.8)

    out_open = model(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])
    assert not torch.allclose(out_open.primary_holo_logits, out_open.primary_apo_logits)

    ckpt_file = tmp_path / "toy_dual.pt"
    model.save_checkpoint(ckpt_file)

    prot_bb2, chem_bb2 = make_heterogeneous_biofms()
    reloaded = LogiCA.from_pretrained(
        prot_bb2, chem_bb2, width=32, heads=4, layers=2, gate_mode="tanh_zero",
        d2p_attn_mode="sigmoid", num_primary_fusion_layers=2, checkpoint=ckpt_file
    )
    reloaded.eval()
    out_rel = reloaded(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])
    assert torch.allclose(out_open.primary_holo_logits, out_rel.primary_holo_logits, atol=1e-6)


def test_single_biofm_lora_and_preference_finetuning() -> None:
    prot_bb, _ = make_heterogeneous_biofms()
    single_model = LogiCA.from_pretrained(prot_bb, partner=None, width=32, heads=4, layers=2)
    assert not single_model.is_dual

    opt_groups = single_model.configure_finetuning_optimizer(
        base_lora_lr=1e-3,
        adapter_lr_multiplier=5.0,
        top_k_primary_lora=2,
        lora_rank=4,
    )
    optimizer = torch.optim.AdamW(opt_groups)
    pl_loss = PlackettLuceLogitLoss(top_m=3)
    bt_loss = BradleyTerryLogitLoss()
    kl_loss = AncestralKLAnchorLoss()

    wt = "ACDEFGHIKLMNPQRSTV"
    edits = [parse_mutation_string(m) for m in ("A1W", "C2Y", "D3K", "E4R")]
    targets = torch.tensor([2.5, 1.2, -0.4, -1.8])

    from logica import VariantRanker

    ranker = VariantRanker(single_model)
    with torch.no_grad():
        _, _, ref_lp = ranker.score_cohort_differentiable(wt, edits[:1], context=None)

    single_model.train()
    initial_loss = None
    final_loss = None
    for _ in range(6):
        scores, holo_lp, _ = ranker.score_cohort_differentiable(wt, edits, context=None)
        loss = 0.5 * pl_loss(scores, targets) + 0.5 * bt_loss(scores, targets) + 0.05 * kl_loss(holo_lp, ref_lp)
        if initial_loss is None:
            initial_loss = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss


def test_infonce_logit_loss_and_zero_wt_collision_negatives() -> None:
    # Verify 0% WT collisions across 300 9-mer negative draws
    epi = "SLLMWITQC"
    negs = create_anchored_mutation_negatives(epi, num_negatives=300, strategy="central_contact")
    assert len(negs) == 300
    assert all(n != epi and len(n) == len(epi) for n in negs)

    prot_bb, chem_bb = make_heterogeneous_biofms()
    model = LogiCA.from_pretrained(prot_bb, chem_bb, width=32, heads=4, layers=2)
    with torch.no_grad():
        model.prot_gate.fill_(0.5)
        model.drug_gate.fill_(0.5)

    opt = torch.optim.AdamW(model.configure_finetuning_optimizer(base_lora_lr=1e-3, top_k_primary_lora=1, top_k_partner_lora=1))
    criterion = InfoNCELogitLoss(temperature=0.1, mlm_aux_weight=0.2)

    p_batch = prot_bb.tokenize(["ACDEFGHIKLMN", "MNPQRSTVWYAC", "GHIKLMNPQRST", "TVWYACDEFGHI"])
    d_batch = chem_bb.tokenize(["[C][O][N]", "[C][=O][F]", "[Ring1][Cl][C]", "[O][Branch1][N]"])

    stats_out = criterion(model, p_batch, d_batch)
    assert torch.isfinite(stats_out["loss"])
    opt.zero_grad()
    stats_out["loss"].backward()
    opt.step()


def test_variant_ranking_dependency_maps_gibbs_and_plots(tmp_path: Path) -> None:
    prot_bb, chem_bb = make_heterogeneous_biofms()
    model = LogiCA.from_pretrained(prot_bb, chem_bb, width=32, heads=4, layers=2)
    with torch.no_grad():
        model.prot_gate.fill_(0.9)
        model.drug_gate.fill_(0.9)

    wt = "ACDEFGHIKLMNPQRSTVWY"
    lig = "[C][O][=O][Ring1][F][Cl]"
    mut_strs = ["A1V", "D3E", "F5Y_G6A", "W19F"]
    mut_seqs = [apply_mutations(wt, m) for m in mut_strs]
    assert len(diff_mutations(wt, mut_seqs[2])) == 2

    for mode in ("wt_marginal", "stride_masked", "masked_sites", "exact_loo"):
        rk = model.rank_variants(wt, mut_strs, context=lig, variants_are_mutation_strings=True, mode=mode)  # type: ignore[arg-type]
        assert rk.scores.shape == (4,)
        assert len(rk.top_k(2)) == 2

    # Dependency map (cross-modal + single-BioFM intra-sequence) & KL hotspots
    dep_cross = model.dependency_map(wt, lig, metric="sym_kl")
    assert dep_cross.matrix.shape == (len(wt), 6)
    kl_hs = model.detect_kl_hotspots(wt, lig, top_k=4)
    assert len(kl_hs["top_indices_0based"]) == 4  # type: ignore[arg-type]

    single = LogiCA.from_pretrained(prot_bb, partner=None, width=32, heads=4, layers=1)
    dep_intra = single.dependency_map(wt)
    assert dep_intra.matrix.shape == (len(wt), len(wt))
    assert np.allclose(np.diag(dep_intra.matrix), 0.0)

    # Coordinate Gibbs sampling (auto KL sites + explicit sites) & landscape Gibbs
    gibbs_holo = model.gibbs_sample(wt, context=lig, auto_kl_top_k=4, num_chains=6, num_steps=5)
    gibbs_apo = single.gibbs_sample(wt, context=None, design_positions_0based=[0, 2, 4, 6], num_chains=6, num_steps=5)
    assert gibbs_holo.score_trajectories.shape == (6, 6)
    assert len(gibbs_holo.top_designs(3)) >= 1

    # Landscape metrics & ThermodynamicHeadlessScorer
    y_exp = np.linspace(-1.5, 2.5, 30)
    s_mod = y_exp + 0.2 * np.sin(np.arange(30))
    m_rk = evaluate_ranking_metrics(y_exp, s_mod)
    assert m_rk["spearman_rho"] > 0.9
    gb_land = evaluate_landscape_gibbs(y_exp, s_mod, {1: list(range(15)), 2: list(range(15, 30))}, num_steps=5)
    assert gb_land["boltzmann_gibbs_top10_enrichment"] > 1.5  # type: ignore[operator]

    thermo = ThermodynamicHeadlessScorer()
    t_scores = thermo(torch.tensor([-0.5, -4.0]), torch.tensor([0.8, -0.3]), assay_polarity=-1.0)
    assert t_scores.shape == (2,)

    contacts = [1 if i in (2, 5, 8) else 0 for i in range(len(wt))]
    c_metrics = evaluate_contact_localization(contacts, dep_cross.primary_profile)
    assert np.isfinite(c_metrics["auroc"])

    # Verify all three visualization builders save PNGs cleanly
    f1 = tmp_path / "dep.png"
    f2 = tmp_path / "gibbs.png"
    f3 = tmp_path / "rank.png"
    plot_dependency_map(dep_cross, contact_residues_1based=[3, 6, 9], save_path=f1)
    plot_gibbs_design(gibbs_holo, baseline_result=gibbs_apo, save_path=f2)
    plot_variant_ranking(y_exp, s_mod, baseline_scores=np.zeros_like(s_mod), save_path=f3)
    assert f1.is_file() and f2.is_file() and f3.is_file()


def test_external_hidden_residual_hook() -> None:
    """Verifies optional extra_hidden residuals inject cleanly before native LM heads."""
    prot_bb, chem_bb = make_heterogeneous_biofms()
    dual = LogiCA.from_pretrained(prot_bb, chem_bb, width=32, heads=4, layers=1, gate_mode="tanh_zero")
    dual.eval()

    p_enc = prot_bb.tokenize(["ACDEFGHIKLMN"])
    d_enc = chem_bb.tokenize(["[C][O][N][F]"])

    base = dual(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])
    zero_p = torch.zeros_like(base.primary_contextual_hidden)
    zero_d = torch.zeros_like(base.partner_contextual_hidden)
    out_zero = dual(
        p_enc["input_ids"],
        p_enc["attention_mask"],
        d_enc["input_ids"],
        d_enc["attention_mask"],
        extra_hidden_primary=zero_p,
        extra_hidden_partner=zero_d,
    )
    assert torch.allclose(out_zero.primary_holo_logits, base.primary_holo_logits, atol=1e-6)
    assert torch.allclose(out_zero.partner_holo_logits, base.partner_holo_logits, atol=1e-6)

    delta_p = torch.randn_like(base.primary_contextual_hidden, requires_grad=True)
    delta_d = torch.randn_like(base.partner_contextual_hidden, requires_grad=True)
    out_ext = dual(
        p_enc["input_ids"],
        p_enc["attention_mask"],
        d_enc["input_ids"],
        d_enc["attention_mask"],
        extra_hidden_primary=delta_p,
        extra_hidden_partner=delta_d,
    )
    assert not torch.allclose(out_ext.primary_holo_logits, base.primary_holo_logits)
    assert not torch.allclose(out_ext.partner_holo_logits, base.partner_holo_logits)
    assert torch.allclose(out_ext.primary_apo_logits, base.primary_apo_logits, atol=1e-6)
    (out_ext.primary_holo_logits.sum() + out_ext.partner_holo_logits.sum()).backward()
    assert delta_p.grad is not None and delta_d.grad is not None

    # Single-BioFM mode also supports extra_hidden_primary
    single = LogiCA.from_pretrained(prot_bb, partner=None, width=32, heads=4, layers=1)
    single.eval()
    s_base = single(p_enc["input_ids"], p_enc["attention_mask"])
    s_ext = single(
        p_enc["input_ids"],
        p_enc["attention_mask"],
        extra_hidden_primary=torch.ones_like(s_base.primary_contextual_hidden) * 0.25,
    )
    assert not torch.allclose(s_ext.primary_holo_logits, s_base.primary_holo_logits)
    assert torch.allclose(s_ext.primary_apo_logits, s_base.primary_apo_logits, atol=1e-6)


def test_paired_pipe_delimited_gibbs_preserves_separator() -> None:
    """Verifies paired `VH|VL` / `CDR3b|CDR3a` Gibbs sampling mutates both chains without corrupting `|`."""

    class PipeAwareTokenizer(ToyVocabularyTokenizer):
        def __init__(self, tokens: tuple[str, ...]) -> None:
            super().__init__(tokens)
            self.sep_token_id = len(self.all_tokens)
            self.tok_to_id["|"] = self.sep_token_id
            self.id_to_tok[self.sep_token_id] = "|"
            self.all_special_ids = [*self.all_special_ids, self.sep_token_id]

        def __call__(
            self,
            sequences: list[str],
            max_length: int | None = None,
            padding: bool | str = True,
            truncation: bool = True,
            return_tensors: str = "pt",
        ) -> dict[str, torch.Tensor]:
            out = super().__call__(
                sequences,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            out["special_tokens_mask"] = out["special_tokens_mask"] | (out["input_ids"] == self.sep_token_id)
            return out

    torch.manual_seed(11)
    prot_tok = PipeAwareTokenizer(CANONICAL_AAS)
    _, chem_bb = make_heterogeneous_biofms()
    prot_vocab = len(prot_tok.tok_to_id)
    prot_bb = CustomBioFMBackbone(
        encoder=ToyBioFMEncoder(prot_vocab, dim=32, num_layers=2),
        lm_head=nn.Linear(32, prot_vocab),
        tokenizer=prot_tok,
        hidden_size=32,
        vocab_size=prot_vocab,
        model_id="toy-paired-ab-fm",
    )
    model = LogiCA.from_pretrained(prot_bb, chem_bb, width=24, heads=4, layers=1)
    with torch.no_grad():
        model.prot_gate.fill_(0.6)
        model.drug_gate.fill_(0.6)

    paired_wt = "ACDEFG|HIKLMN"
    # Design positions 1 (VH) and 7 (VL, index 1 after `|`) in valid-token space
    res = model.gibbs_sample(
        paired_wt,
        context="[C][O][N][F]",
        design_positions_0based=[1, 7],
        num_chains=6,
        num_steps=8,
        temperature=1.0,
        metropolis_accept=False,
    )
    for seq in res.final_sequences:
        assert len(seq) == len(paired_wt)
        assert seq[6] == "|"
        # Untouched positions on both VH and VL remain unchanged
        assert seq[0] == "A" and seq[2:6] == "DEFG" and seq[7] == "H" and seq[9:] == "KLMN"


def test_lora_checkpoint_roundtrip_merges_weights_exactly(tmp_path: Path) -> None:
    """Verifies saving a LoRA-adapted LogiCA checkpoint and reloading via from_checkpoint merges LoRA accurately."""
    prot_bb, chem_bb = make_heterogeneous_biofms(seed=19)
    model = LogiCA.from_pretrained(prot_bb, chem_bb, width=24, heads=4, layers=1)
    model.configure_finetuning_optimizer(top_k_primary_lora=1, top_k_partner_lora=1, lora_rank=4, lora_alpha=12.0)

    # Set non-zero LoRA weights so merging produces a clear difference from base weights
    torch.manual_seed(23)
    with torch.no_grad():
        model.prot_gate.fill_(0.4)
        model.drug_gate.fill_(0.3)
        for name, param in model.named_parameters():
            if name.endswith("lora_b"):
                param.normal_(0.0, 0.25)

    model.eval()
    p_enc = prot_bb.tokenize(["ACDEFGHIKLMN"])
    d_enc = chem_bb.tokenize(["[C][O][N][F]"])
    with torch.no_grad():
        ref_out = model(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])

    ckpt_path = tmp_path / "lora_roundtrip.pt"
    model.save_checkpoint(ckpt_path)

    # Reload into a fresh non-LoRA LogiCA model (passing custom backbones directly via load_checkpoint)
    fresh_prot, fresh_chem = make_heterogeneous_biofms(seed=19)
    reloaded = LogiCA.from_pretrained(fresh_prot, fresh_chem, width=24, heads=4, layers=1)
    reloaded.load_checkpoint(ckpt_path)
    reloaded.eval()
    with torch.no_grad():
        new_out = reloaded(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])

    assert torch.allclose(ref_out.primary_holo_logits, new_out.primary_holo_logits, atol=1e-5)
    assert torch.allclose(ref_out.primary_apo_logits, new_out.primary_apo_logits, atol=1e-5)
    assert torch.allclose(ref_out.partner_holo_logits, new_out.partner_holo_logits, atol=1e-5)


