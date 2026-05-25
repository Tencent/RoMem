import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=8192,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    temporal_passage_node_weight: float = field(
        default=0.0,
        metadata={"help": "Passage node weight for explicit-time queries (enable_tkge_tunnel=True)."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    openie_max_workers: Union[None, int] = field(
        default=None,
        metadata={"help": "Max worker threads for OpenIE requests. If None, use default."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    use_llm_fact_filter: bool = field(
        default=True,
        metadata={"help": "Use the LLM-based DSPy reranker to filter candidate facts before final retrieval."},
    )

    # Retrieval (RoMem extensions) - TKGE structural tunnel
    enable_tkge_tunnel: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable/disable the TKGE temporal tunnel."},
    )
    tkge_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for combining TKGE structural scores with text-based fact scores."},
    )
    tkge_entity_top_k: int = field(
        default=5,
        metadata={"help": "Top-k entities to link from query in the TKGE tunnel."},
    )
    tkge_candidate_fact_top_k: int = field(
        default=200,
        metadata={"help": "Max candidate facts to score structurally in the TKGE tunnel."},
    )
    fact_candidate_k_for_kge: int = field(
        default=10,
        metadata={"help": "If >0 and TKGE tunnel is enabled, expand candidate facts to this size before final top-k."},
    )
    tkge_debug_top_k: int = field(
        default=3,
        metadata={"help": "Top-k TKGE candidates to include in retrieval debug output when verbose>=2."},
    )
    tkge_debug_shift_days: int = field(
        default=30,
        metadata={"help": "Day shift for TKGE rotation debug (query time +/- shift) when verbose>=2."},
    )
    tkge_steps_per_update: int = field(
        default=200,
        metadata={"help": "TKGE training epochs per incremental update."},
    )
    tkge_learning_rate: float = field(
        default=0.1,
        metadata={"help": "TKGE learning rate for incremental updates."},
    )
    tkge_embedding_dim: int = field(
        default=128,
        metadata={"help": "TKGE base embedding dimension (entity/relation)."},
    )
    tkge_use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for TKGE incremental learning."},
    )
    tkge_lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank for TKGE."},
    )
    tkge_triple_margin: float = field(
        default=0.5,
        metadata={"help": "Margin used in the TKGE triple log-sigmoid loss."},
    )
    tkge_time_margin: float = field(
        default=0.5,
        metadata={"help": "Margin used in the TKGE time-contrastive loss."},
    )
    tkge_time_contrastive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the TKGE time-contrastive loss."},
    )
    tkge_time_loss_type: Literal["pairwise", "listwise"] = field(
        default="listwise",
        metadata={"help": "Time-contrastive loss type for TKGE (pairwise margin or listwise distribution)."},
    )
    tkge_num_time_negatives: int = field(
        default=4,
        metadata={"help": "Number of negative times sampled per fact for TKGE time-contrastive loss."},
    )
    tkge_time_sigma_years: float = field(
        default=0.5,
        metadata={"help": "Sigma (years) for temporal distance weighting in TKGE time-contrastive loss."},
    )
    tkge_time_sigma_years_start: Optional[float] = field(
        default=None,
        metadata={"help": "Optional curriculum start sigma (years) for time-contrastive loss."},
    )
    tkge_time_sigma_years_end: Optional[float] = field(
        default=None,
        metadata={"help": "Optional curriculum end sigma (years) for time-contrastive loss."},
    )
    tkge_time_sigma_decay_epochs: int = field(
        default=0,
        metadata={"help": "Epochs over which sigma decays from start to end (0 disables)."},
    )
    tkge_time_neg_jitter_years: float = field(
        default=0.0,
        metadata={"help": "Jitter (years) applied to sampled negative times for TKGE time-contrastive loss."},
    )
    tkge_time_neg_far_days: int = field(
        default=0,
        metadata={"help": "Force one far negative per example by offsetting time by +/- days (0 disables)."},
    )
    tkge_time_neg_min_days_start: float = field(
        default=0.0,
        metadata={"help": "Curriculum start minimum gap (days) for time negatives (0 disables)."},
    )
    tkge_time_neg_min_days_end: float = field(
        default=0.0,
        metadata={"help": "Curriculum end minimum gap (days) for time negatives."},
    )
    tkge_time_neg_min_days_decay_epochs: int = field(
        default=0,
        metadata={"help": "Epochs over which min negative gap decays from start to end (0 disables)."},
    )
    tkge_temporal_warmup_epochs: int = field(
        default=0,
        metadata={"help": "Number of initial TKGE epochs to train without temporal losses."},
    )
    tkge_checkpoint_start_epoch: int = field(
        default=0,
        metadata={"help": "Start epoch for best-loss checkpointing (0 = auto based on curriculum)."},
    )
    tkge_time_gate_stage1_epochs: int = field(
        default=5,
        metadata={"help": "Stage-1 epochs with alpha=1.0 to learn global time scale before learning the gate."},
    )
    tkge_time_gate_freeze_omega_after: bool = field(
        default=True,
        metadata={"help": "Freeze global omega (time scale + base) after stage-1 so alpha learns on a fixed spectrum."},
    )
    tkge_time_gate_checkpoint: str = field(
        default="",
        metadata={"help": "Path to pretrained alpha_r checkpoint (.pt). When set, the gate MLP is loaded "
                  "and frozen, bypassing the two-stage schedule. Only s, omega, and embeddings train online."},
    )
    tkge_verbose: int = field(
        default=0,
        metadata={"help": "TKGE training verbosity (0=silent, 1=per-step logs)."},
    )
    tkge_verbose_epoch_interval: int = field(
        default=5,
        metadata={"help": "Epoch interval for verbose training logs (1=every epoch)."},
    )

    tkge_temporal_mode: Literal["none", "romem"] = field(
        default="romem",
        metadata={"help": "Temporal mode for TKGE: none (standard DistMult) or romem (time rotation)."},
    )
    tkge_temporal_backbone: Literal["distmult", "chronor"] = field(
        default="distmult",
        metadata={"help": "Backbone model for RoMem temporal KGE: distmult or chronor (k-component bilinear)."},
    )
    tkge_chronor_k: int = field(
        default=3,
        metadata={"help": "Number of components (k) for ChronoR backbone. Only used when tkge_temporal_backbone=chronor."},
    )
    tkge_time_source: Literal["happen", "obs", "happen_else_obs"] = field(
        default="happen_else_obs",
        metadata={"help": "Which timestamp to use for TKGE training time: happen, obs, or fallback."},
    )
    tkge_gamma: float = field(
        default=200.0,
        metadata={"help": "Gamma for KGE initialization range: embedding_range = (gamma + 2) / dim."},
    )
    tkge_adversarial_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for self-adversarial negative sampling in TKGE triple loss."},
    )
    tkge_regularization_weight: float = field(
        default=1e-5,
        metadata={"help": "Regularization weight: L3 global (DistMult) or N3 per-batch (ChronoR)."},
    )

    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
