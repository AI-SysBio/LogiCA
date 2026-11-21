"""LogiCA: Model-Agnostic Logit-Space Contrastive Alignment for Biological Foundation Models."""

from .adapters import (
    AdapterStyle,
    AttentionMode,
    BidirectionalCrossAttention,
    CrossAttentionBlock,
    GateMode,
    HeterogeneousCrossAttentionLayer,
    LoRALinear,
    MultiLayerHiddenMixer,
    UnimodalGatedAdapter,
    inject_late_layer_lora,
)
from .backbones import (
    CANONICAL_AAS,
    AbLang2BioFMBackbone,
    BioFMBackbone,
    CustomBioFMBackbone,
    HuggingFaceMaskedLMBackbone,
    load_biofm,
)
from .dependency import (
    DependencyMapResult,
    DependencyMapper,
    evaluate_contact_localization,
)
from .gibbs import (
    GibbsDesignResult,
    GibbsSampler,
    evaluate_landscape_gibbs,
)
from .losses import (
    AncestralKLAnchorLoss,
    BradleyTerryLogitLoss,
    InfoNCELogitLoss,
    PlackettLuceLogitLoss,
    ThermodynamicHeadlessScorer,
    create_anchored_mutation_negatives,
    make_masked_inputs,
    mean_selected_log_probability,
)
from .model import (
    TCR_LOGICA_HF_REPO,
    YALE_PROTEIN_LIGAND_HF_REPO,
    LogiCA,
    LogiCAOutput,
)
from .ranking import (
    MutationEdit,
    VariantRanker,
    VariantRankingResult,
    apply_mutations,
    diff_mutations,
    evaluate_ranking_metrics,
    parse_mutation_string,
    top_k_enrichment,
    window_sequence_around_mutations,
)
from .visualization import (
    plot_dependency_map,
    plot_gibbs_design,
    plot_variant_ranking,
)

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_AAS",
    "AbLang2BioFMBackbone",
    "AdapterStyle",
    "AncestralKLAnchorLoss",
    "AttentionMode",
    "BidirectionalCrossAttention",
    "BioFMBackbone",
    "BradleyTerryLogitLoss",
    "CrossAttentionBlock",
    "CustomBioFMBackbone",
    "DependencyMapResult",
    "DependencyMapper",
    "GateMode",
    "GibbsDesignResult",
    "GibbsSampler",
    "HeterogeneousCrossAttentionLayer",
    "HuggingFaceMaskedLMBackbone",
    "InfoNCELogitLoss",
    "LoRALinear",
    "LogiCA",
    "LogiCAOutput",
    "MultiLayerHiddenMixer",
    "MutationEdit",
    "PlackettLuceLogitLoss",
    "TCR_LOGICA_HF_REPO",
    "ThermodynamicHeadlessScorer",
    "UnimodalGatedAdapter",
    "VariantRanker",
    "VariantRankingResult",
    "YALE_PROTEIN_LIGAND_HF_REPO",
    "apply_mutations",
    "create_anchored_mutation_negatives",
    "diff_mutations",
    "evaluate_contact_localization",
    "evaluate_landscape_gibbs",
    "evaluate_ranking_metrics",
    "inject_late_layer_lora",
    "load_biofm",
    "make_masked_inputs",
    "mean_selected_log_probability",
    "parse_mutation_string",
    "plot_dependency_map",
    "plot_gibbs_design",
    "plot_variant_ranking",
    "top_k_enrichment",
    "window_sequence_around_mutations",
]
