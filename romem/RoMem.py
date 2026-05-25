import json
import os
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from typing import Union, Optional, List, Set, Dict, Tuple
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

from .llm import _get_llm_class
from .llm.base import BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from romem.ingestion import OpenIE
from romem.ingestion import GraphManager
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput, TimedTriple
from .utils.embed_utils import retrieve_knn
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class RoMem:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific RoMem instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Apply temporal config: enable_tkge_tunnel controls all temporal modules.
        tkge_enabled = bool(getattr(self.global_config, "enable_tkge_tunnel", False))
        self.global_config.enable_tkge_tunnel = tkge_enabled
        self.global_config.tkge_temporal_mode = "romem" if tkge_enabled else "none"

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"RoMem init with config:\n  {_print_config}\n")

        # Reduce OpenAI client retry log spam for readability.
        for _logger_name in ("openai", "openai._base_client"):
            logging.getLogger(_logger_name).setLevel(logging.WARNING)

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(
                llm_model=self.llm_model,
                max_workers=self.global_config.openie_max_workers,
            )
        elif self.global_config.openie_mode == 'offline':
            from romem.ingestion import VLLMOfflineOpenIE
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode == 'Transformers-offline':
            from romem.ingestion.openie_transformers_offline import TransformersOfflineOpenIE
            self.openie = TransformersOfflineOpenIE(self.global_config)

        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )
        self.graph_manager = GraphManager(
            pickle_path=self._graph_pickle_filename,
            directed=self.global_config.is_directed_graph,
        )
        self.graph = self.graph_manager.load_or_init(
            force_from_scratch=self.global_config.force_index_from_scratch
        )

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(
                    global_config=self.global_config,
                    embedding_model_name=self.global_config.embedding_model_name)

        # Group embedding stores under a builder for modularity.
        if self.embedding_model is not None:
            from romem.ingestion import GraphBuilder

            builder = GraphBuilder.from_paths(
                embedding_model=self.embedding_model,
                working_dir=self.working_dir,
                batch_size=self.global_config.embedding_batch_size,
            )
            self.graph_builder = builder
            self.chunk_embedding_store = builder.chunk_store
            self.entity_embedding_store = builder.entity_store
            self.fact_embedding_store = builder.fact_store
        else:
            self.graph_builder = None
            self.chunk_embedding_store = None
            self.entity_embedding_store = None
            self.fact_embedding_store = None

        # TKGE structural retrieval tunnel (optional; ablation switch).
        self.tkge_tunnel_enabled: bool = bool(getattr(self.global_config, 'enable_tkge_tunnel', False))
        self.tkge_retriever = None
        self.entity_texts: List[str] = []
        self.fact_tuples: List[Tuple[str, str, str]] = []
        self.entity_to_fact_indices: Dict[str, Set[int]] = {}
        self.last_tkge_debug: Dict[str, Any] | None = None
        self.last_rerank_debug: Dict[str, Any] | None = None
        if self.tkge_tunnel_enabled:
            from .kge.retriever import TKGERetriever
            from .kge.config import TKGEConfig

            cfg = TKGEConfig(
                embedding_dim=self.global_config.tkge_embedding_dim,
                learning_rate=self.global_config.tkge_learning_rate,
                batch_size=self.global_config.embedding_batch_size,
                steps_per_update=self.global_config.tkge_steps_per_update,
                use_lora=self.global_config.tkge_use_lora,
                lora_rank=self.global_config.tkge_lora_rank,
                checkpoint_strategy='best_train_loss',
                triple_margin=self.global_config.tkge_triple_margin,
                gamma=float(getattr(self.global_config, 'tkge_gamma', 200.0)),
                adversarial_temperature=float(getattr(self.global_config, 'tkge_adversarial_temperature', 1.0)),
                regularization_weight=float(getattr(self.global_config, 'tkge_regularization_weight', 1e-5)),
                temporal_mode=self.global_config.tkge_temporal_mode,
                temporal_backbone=getattr(self.global_config, 'tkge_temporal_backbone', 'distmult'),
                chronor_k=int(getattr(self.global_config, 'tkge_chronor_k', 3)),
                time_source=self.global_config.tkge_time_source,
                # Temporal supervision: enable time-contrastive loss when TKGE tunnel is on.
                use_time_contrastive=bool(getattr(self.global_config, "enable_tkge_tunnel", False)),
                time_contrastive_weight=self.global_config.tkge_time_contrastive_weight,
                time_contrastive_margin=self.global_config.tkge_time_margin,
                time_loss_type=self.global_config.tkge_time_loss_type,
                num_time_negatives=self.global_config.tkge_num_time_negatives,
                time_neg_jitter_years=self.global_config.tkge_time_neg_jitter_years,
                time_neg_far_days=self.global_config.tkge_time_neg_far_days,
                time_neg_min_days_start=self.global_config.tkge_time_neg_min_days_start,
                time_neg_min_days_end=self.global_config.tkge_time_neg_min_days_end,
                time_neg_min_days_decay_epochs=self.global_config.tkge_time_neg_min_days_decay_epochs,
                time_sigma_years=self.global_config.tkge_time_sigma_years,
                time_sigma_years_start=self.global_config.tkge_time_sigma_years_start,
                time_sigma_years_end=self.global_config.tkge_time_sigma_years_end,
                time_sigma_decay_epochs=self.global_config.tkge_time_sigma_decay_epochs,
                time_gate_reg_weight=0.0,
                force_time_gate_one=False,
                temporal_warmup_epochs=self.global_config.tkge_temporal_warmup_epochs,
                time_gate_stage1_epochs=self.global_config.tkge_time_gate_stage1_epochs,
                time_gate_freeze_omega_after=self.global_config.tkge_time_gate_freeze_omega_after,
                checkpoint_start_epoch=self.global_config.tkge_checkpoint_start_epoch,
                time_gate_checkpoint=getattr(self.global_config, 'tkge_time_gate_checkpoint', ''),
            )
            self.tkge_retriever = TKGERetriever(
                config=cfg,
                verbose=int(getattr(self.global_config, 'tkge_verbose', 0)),
                verbose_epoch_interval=int(getattr(self.global_config, 'tkge_verbose_epoch_interval', 5)),
                relation_embedder=self._embed_relation_texts,
            )

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)
        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

    def _embed_relation_texts(self, rel_texts: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("Relation embedding requested but embedding_model is not initialized.")
        if isinstance(rel_texts, str):
            rel_texts = [rel_texts]
        return self.embedding_model.batch_encode(rel_texts)

    def pretrain_tkge_time_gate(
        self,
        pretrain_data_dir: str = "outputs/pretrain_gate_data",
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> Dict:
        """
        Debug-stage gate pretraining using unsupervised artifact files.

        This is intentionally separated from normal RoMem training so alpha_r
        can be inspected before automatic chaining.
        """
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        result = self.tkge_retriever.pretrain_time_gate(
            pretrain_data_dir=pretrain_data_dir,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.last_tkge_debug = {
            "stage": "gate_pretrain",
            "result": result,
        }
        return result

    def inspect_tkge_time_gate(
        self,
        relations: Optional[List[str]] = None,
        top_k: int = 10,
        print_report: bool = True,
    ) -> List[Dict]:
        """
        Print/return current alpha_r values for selected relations.
        """
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        report = self.tkge_retriever.inspect_time_gate(
            relations=relations,
            top_k=top_k,
            print_report=print_report,
        )
        self.last_tkge_debug = {
            "stage": "gate_inspect",
            "report_size": len(report),
            "relations": relations or [],
        }
        return report

    def save_tkge_time_gate(self, checkpoint_path: str) -> Dict:
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        return self.tkge_retriever.save_time_gate_checkpoint(checkpoint_path)

    def load_tkge_time_gate(self, checkpoint_path: str, strict_dim: bool = True) -> Dict:
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        return self.tkge_retriever.load_time_gate_checkpoint(
            checkpoint_path=checkpoint_path,
            strict_dim=strict_dim,
        )

    def save_tkge_checkpoint(self, checkpoint_path: str) -> Dict:
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        return self.tkge_retriever.save_full_checkpoint(checkpoint_path)

    def load_tkge_checkpoint(self, checkpoint_path: str) -> Dict:
        if not self.tkge_tunnel_enabled or self.tkge_retriever is None:
            raise RuntimeError("TKGE tunnel is not enabled.")
        return self.tkge_retriever.load_full_checkpoint(checkpoint_path)

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
            # Incremental TKGE update (structural tunnel) on newly extracted triples only.
            if self.tkge_tunnel_enabled and self.tkge_retriever is not None:
                try:
                    new_chunk_ids = list(new_openie_rows.keys())
                    if self.global_config.tkge_temporal_mode == "romem":
                        new_timed = []
                        for cid in new_chunk_ids:
                            new_timed.extend(new_triple_results_dict[cid].timed_triples)
                        self.tkge_retriever.update(new_timed)
                    else:
                        new_chunk_triples = [
                            [text_processing(t) for t in new_triple_results_dict[cid].triples] for cid in new_chunk_ids
                        ]
                        new_facts = flatten_facts(new_chunk_triples)
                        self.tkge_retriever.update(new_facts)
                except Exception as exc:
                    logger.error(f"TKGE incremental update failed: {exc}")
                    raise

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str], observed_time: str | None = None):
        """
        Indexes the given documents based on the RoMem framework (HippoRAG-derived) which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
            observed_time : str, optional
                Override for the ingestion time used by the OpenIE time normalizer and metadata.
                When provided, it is applied to all documents in this call.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        # Store observation time on chunk records (metadata). This does not change embeddings by default.
        obs_now = (observed_time or datetime.now(tz=timezone.utc).isoformat())
        self.chunk_embedding_store.insert_records(
            [{"content": d, "meta": {"observed_time": obs_now}} for d in docs]
        )
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        new_chunk_ids = list(new_openie_rows.keys()) if new_openie_rows else []
        # Incremental TKGE update (structural tunnel) on newly accepted triples only.
        if self.tkge_tunnel_enabled and self.tkge_retriever is not None and new_chunk_ids:
            try:
                if self.global_config.tkge_temporal_mode == "romem":
                    new_timed = []
                    for cid in new_chunk_ids:
                        new_timed.extend(triple_results_dict[cid].timed_triples)
                    self.tkge_retriever.update(new_timed)
                else:
                    new_chunk_triples = [
                        [text_processing(t) for t in triple_results_dict[cid].triples] for cid in new_chunk_ids
                    ]
                    new_facts = flatten_facts(new_chunk_triples)
                    self.tkge_retriever.update(new_facts)
            except Exception as exc:
                logger.error(f"TKGE incremental update failed: {exc}")
                raise

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_timed_triples = [triple_results_dict[chunk_id].timed_triples for chunk_id in chunk_ids]
        chunk_triples = [
            [text_processing(list(tt.triple)) for tt in timed_list] for timed_list in chunk_timed_triples
        ]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.debug("Encoding Entities")
        self.entity_embedding_store.insert_records(
            [{"content": e, "meta": {"last_seen_time": obs_now}} for e in entity_nodes]
        )

        logger.debug("Encoding Facts")
        fact_to_time: Dict[Tuple[str, str, str], str] = {}
        fact_to_chunk_ids: Dict[Tuple[str, str, str], List[str]] = {}
        for timed_list in chunk_timed_triples:
            for tt in timed_list:
                tr = tuple(text_processing(list(tt.triple)))
                if tr not in fact_to_time and tt.happen_time:
                    fact_to_time[tr] = str(tt.happen_time)
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            for triple in triples:
                tr = tuple(triple)
                fact_to_chunk_ids.setdefault(tr, [])
                fact_to_chunk_ids[tr].append(str(chunk_key))
        records = []
        for fact in facts:
            f_tuple = tuple(fact)
            happen = fact_to_time.get(f_tuple, "")
            chunk_ids_for_fact = fact_to_chunk_ids.get(f_tuple, [])
            base = str(fact)
            records.append(
                {
                    "content": base,
                    "embed_text": base,
                    "meta": {"happen_time": happen, "observed_time": obs_now, "chunk_ids": chunk_ids_for_fact},
                }
            )
        self.fact_embedding_store.insert_records(records)

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.debug(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def index_triples(
        self,
        docs: List[str],
        timed_triples: List[TimedTriple],
        observed_time: str | None = None,
        embed_texts: List[str] | None = None,
        *,
        train_tkge: bool = True,
    ):
        """
        Index pre-extracted (subject, relation, object) triples with explicit times, bypassing OpenIE/NER.

        Parameters:
            docs: List[str]
                Passage strings to anchor triples (used for passage embeddings).
            timed_triples: List[TimedTriple]
                Triples with happen_time + system_time already attached.
            observed_time: str, optional
                Observation time used in metadata when provided.
            embed_texts: List[str], optional
                Alternative text used for embedding while keeping `docs` as stable IDs.
        """
        if not docs or not timed_triples:
            return
        if len(docs) != len(timed_triples):
            raise ValueError("docs and timed_triples must be the same length.")

        logger.info("Indexing Structured Triples (bypass OpenIE)")

        obs_now = observed_time or datetime.now(tz=timezone.utc).isoformat()
        records = []
        for idx, doc in enumerate(docs):
            embed_text = embed_texts[idx] if embed_texts is not None else doc
            records.append({"content": doc, "embed_text": embed_text, "meta": {"observed_time": obs_now}})
        self.chunk_embedding_store.insert_records(records)

        chunk_ids = [compute_mdhash_id(doc, prefix="chunk-") for doc in docs]
        triple_results_dict: Dict[str, TripleRawOutput] = {}
        ner_results_dict: Dict[str, NerRawOutput] = {}
        for cid, tt in zip(chunk_ids, timed_triples):
            triple_results_dict[cid] = TripleRawOutput(
                chunk_id=cid,
                response=None,
                metadata={},
                timed_triples=[tt],
            )
            entities = list({str(tt.triple[0]), str(tt.triple[2])})
            ner_results_dict[cid] = NerRawOutput(
                chunk_id=cid,
                response=None,
                unique_entities=entities,
                metadata={},
            )

        if self.tkge_tunnel_enabled and self.tkge_retriever is not None and chunk_ids:
            try:
                if self.global_config.tkge_temporal_mode == "romem":
                    self.tkge_retriever.update(timed_triples, train=train_tkge)
                else:
                    new_facts = flatten_facts([[text_processing(list(tt.triple))] for tt in timed_triples])
                    self.tkge_retriever.update(new_facts, train=train_tkge)
            except Exception as exc:
                logger.error(f"TKGE incremental update failed: {exc}")
                raise

        chunk_timed_triples = [triple_results_dict[cid].timed_triples for cid in chunk_ids]
        chunk_triples = [
            [text_processing(list(tt.triple)) for tt in timed_list] for timed_list in chunk_timed_triples
        ]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info("Encoding Entities")
        self.entity_embedding_store.insert_records(
            [{"content": e, "meta": {"last_seen_time": obs_now}} for e in entity_nodes]
        )

        logger.info("Encoding Facts")
        fact_to_time: Dict[Tuple[str, str, str], str] = {}
        fact_to_chunk_ids: Dict[Tuple[str, str, str], List[str]] = {}
        for timed_list in chunk_timed_triples:
            for tt in timed_list:
                tr = tuple(text_processing(list(tt.triple)))
                if tr not in fact_to_time and tt.happen_time:
                    fact_to_time[tr] = str(tt.happen_time)
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            for triple in triples:
                tr = tuple(triple)
                fact_to_chunk_ids.setdefault(tr, [])
                fact_to_chunk_ids[tr].append(str(chunk_key))

        records = []
        for fact in facts:
            f_tuple = tuple(fact)
            happen = fact_to_time.get(f_tuple, "")
            chunk_ids_for_fact = fact_to_chunk_ids.get(f_tuple, [])
            base = str(fact)
            records.append(
                {
                    "content": base,
                    "embed_text": base,
                    "meta": {"happen_time": happen, "observed_time": obs_now, "chunk_ids": chunk_ids_for_fact},
                }
            )
        self.fact_embedding_store.insert_records(records)

        if not hasattr(self, "proc_triples_to_docs") or self.proc_triples_to_docs is None:
            self.proc_triples_to_docs = {}
        for chunk_key, triples in zip(chunk_ids, chunk_triples):
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union({chunk_key})

        logger.info("Constructing Graph")
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)
        if num_new_chunks > 0:
            logger.debug(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

    def train_tkge(self) -> None:
        """Explicitly trigger TKGE training on all accumulated triples.

        Use after calling ``index_triples(..., train_tkge=False)`` in a loop
        to train only once on the complete graph.
        """
        if self.tkge_tunnel_enabled and self.tkge_retriever is not None:
            encoder = self.tkge_retriever.encoder
            if encoder is not None:
                encoder._maybe_init_model()
                encoder._train()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the RoMem class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.graph_builder.chunk_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [cid for chunk in docs_to_delete if (cid := self.graph_builder.chunk_id_for_text(chunk))]
        )

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set(
            [tid for triple in processed_true_triples_to_delete if (tid := self.graph_builder.fact_id_for_text(str(triple)))]
        )

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [eid for ent in entities_to_delete if (eid := self.graph_builder.entity_id_for_text(ent))]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.debug(
            f"Deleting {len(chunk_ids_to_delete)} chunks, "
            f"{len(triple_ids_to_delete)} triples, "
            f"{len(filtered_ent_ids_to_delete)} entities"
        )

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 query_time_overrides: List[str] | None = None,
                 observed_time_cutoffs: List[str] | None = None,
                 observed_time_bounds: List[tuple[str | None, str | None]] | None = None,
                 ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the RoMem framework (HippoRAG-derived), which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            query_time_overrides: List[str], optional
                Optional per-query ISO date strings used to override LLM time extraction.
            observed_time_cutoffs: List[str], optional
                Optional per-query ISO date strings used to enforce an observation-time cutoff.
            observed_time_bounds: List[Tuple[str | None, str | None]], optional
                Optional per-query (start, end) ISO date strings to enforce observation-time bounds.

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        if query_time_overrides is not None and len(query_time_overrides) != len(queries):
            logger.warning(
                "query_time_overrides length mismatch (%d vs %d); ignoring overrides",
                len(query_time_overrides),
                len(queries),
            )
            query_time_overrides = None
        if observed_time_cutoffs is not None and len(observed_time_cutoffs) != len(queries):
            logger.warning(
                "observed_time_cutoffs length mismatch (%d vs %d); ignoring cutoffs",
                len(observed_time_cutoffs),
                len(queries),
            )
            observed_time_cutoffs = None
        if observed_time_bounds is not None and len(observed_time_bounds) != len(queries):
            logger.warning(
                "observed_time_bounds length mismatch (%d vs %d); ignoring bounds",
                len(observed_time_bounds),
                len(queries),
            )
            observed_time_bounds = None

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries), disable=len(queries) <= 1):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            query_time = None
            temporal_ordering = None
            time_request = False
            if query_time_overrides is not None:
                override = query_time_overrides[q_idx]
                if override:
                    try:
                        from .kge.time_utils import QueryTime, parse_time_text, time_to_scalar

                        dt = parse_time_text(
                            happen_time=str(override),
                            obs_time="",
                            mode="happen",
                        )
                        if dt is not None:
                            unix = time_to_scalar(dt)
                            if unix > 0:
                                query_time = QueryTime(unix_seconds=unix)
                    except Exception:
                        query_time = None
            if query_time is None:
                query_time, temporal_ordering, time_request = self._extract_query_time_and_ordering(query)
            query_has_time = (query_time is not None) or bool(time_request)
            if (
                self.tkge_tunnel_enabled
                and self.tkge_retriever is not None
                and getattr(self.tkge_retriever, "verbose", 0) >= 1
            ):
                qt_str = "None"
                if query_time is not None and getattr(query_time, "unix_seconds", None):
                    try:
                        qt_str = datetime.fromtimestamp(
                            float(query_time.unix_seconds), tz=timezone.utc
                        ).strftime("%Y-%m-%d")
                    except Exception:
                        qt_str = str(query_time.unix_seconds)
                print(f"[retrieve] q=\"{query[:80]}\" query_time={qt_str} "
                      f"ordering={temporal_ordering} time_request={time_request}")
            if self.tkge_tunnel_enabled and self.tkge_retriever is not None:
                query_fact_scores = self._apply_tkge_tunnel(
                    query,
                    query_fact_scores,
                    query_time=query_time,
                    temporal_ordering=temporal_ordering,
                    time_request=time_request,
                )
            observed_time_cutoff = None
            if observed_time_cutoffs is not None:
                observed_time_cutoff = observed_time_cutoffs[q_idx]
            observed_time_bound = None
            if observed_time_bounds is not None:
                observed_time_bound = observed_time_bounds[q_idx]

            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                query,
                query_fact_scores,
                query_time=query_time,
                temporal_ordering=temporal_ordering,
                time_request=time_request,
                observed_time_cutoff=observed_time_cutoff,
                observed_time_bounds=observed_time_bound,
            )
            if int(getattr(self.global_config, "tkge_verbose", 0) or 0) >= 3:
                query_time_iso = None
                if query_time is not None and getattr(query_time, "unix_seconds", None):
                    try:
                        query_time_iso = datetime.fromtimestamp(
                            float(query_time.unix_seconds),
                            tz=timezone.utc,
                        ).date().isoformat()
                    except Exception:
                        query_time_iso = None
                fact_debug: list[dict[str, Any]] = []
                for rank, (idx, fact) in enumerate(zip(top_k_fact_indices, top_k_facts), start=1):
                    happen = ""
                    observed = ""
                    chunk_ids = []
                    if idx < len(self.fact_happen_times):
                        happen = self.fact_happen_times[int(idx)]
                    if idx < len(self.fact_observed_times):
                        observed = self.fact_observed_times[int(idx)]
                    if idx < len(self.fact_chunk_ids):
                        chunk_ids = self.fact_chunk_ids[int(idx)] or []
                    fact_debug.append(
                        {
                            "rank": rank,
                            "fact_idx": int(idx),
                            "fact": fact,
                            "happen_time": happen,
                            "observed_time": observed,
                            "chunk_ids": chunk_ids,
                        }
                    )
                self.last_rerank_debug = {
                    "query": query,
                    "query_time": query_time_iso,
                    "time_request": bool(time_request),
                    "facts": fact_debug,
                }

            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            fact_strings = [" ".join(str(x) for x in f) for f in top_k_facts] if top_k_facts else []

            if len(top_k_facts) == 0:
                logger.debug('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                passage_node_weight = self.global_config.passage_node_weight
                if bool(getattr(self.global_config, "enable_tkge_tunnel", False)) and query_has_time:
                    passage_node_weight = float(getattr(self.global_config, "temporal_passage_node_weight", 0.0))
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=passage_node_weight,
                )

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve], facts=fact_strings))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        misc_time = self.all_retrieval_time - (self.rerank_time + self.ppr_time)
        logger.info(
            f"Retrieval total={self.all_retrieval_time:.2f}s "
            f"(rerank={self.rerank_time:.2f}s ppr={self.ppr_time:.2f}s misc={misc_time:.2f}s)"
        )

        return retrieval_results

    def _extract_query_time(self, query: str):
        time_value, _ordering, _time_request = self._extract_query_time_and_ordering(query)
        return time_value

    def _extract_query_time_and_ordering(self, query: str) -> tuple["QueryTime | None", str | None, bool]:
        """
        LLM-based query time and temporal ordering extraction.
        Returns (QueryTime|None, ordering|None, time_request).
        ordering is one of: "earliest", "latest", or None.
        time_request is True when the query asks for a time as the answer.
        """
        try:
            import re
            from .kge.time_utils import QueryTime, parse_time_text, time_to_scalar

            now = datetime.now(tz=timezone.utc)
            now_iso = now.date().isoformat()
            try:
                messages = self.prompt_template_manager.render(
                    "time_extraction",
                    query=query,
                    reference_date=now_iso,
                )
            except Exception:
                system_msg = (
                    "You extract the time constraint from a query. "
                    "Return a single line only."
                )
                user_msg = (
                    "Given the query, output the time constraint, temporal ordering intent, and whether the query asks for a time.\n"
                    "Return exactly one line in this format:\n"
                    "time=YYYY-MM-DD; ordering=earliest|latest|none; time_request=yes|no\n\n"
                    "Rules:\n"
                    "- If a time constraint exists, return it as YYYY-MM-DD.\n"
                    "- If the query only specifies a month or year, use the first day of that month or year.\n"
                    "- If the query omits the year, use the year from the reference date.\n"
                    "- Resolve relative expressions using the reference date.\n"
                    "- If no time constraint exists, return time=NONE.\n"
                    "- ordering=earliest for queries like \"earliest/first/oldest\".\n"
                    "- ordering=latest for queries like \"latest/most recent/last time\".\n"
                    "- ordering=none otherwise.\n\n"
                    "- time_request=yes if the query asks for a time as the answer (e.g., when/what year/which year/what date).\n"
                    "- time_request=no if the query only uses time as a constraint or does not ask for time.\n\n"
                    f"Reference date (UTC): {now_iso}\n"
                    f"Query: {query}"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]

            result = self.llm_model.infer(messages)
            if isinstance(result, (list, tuple)) and len(result) == 3:
                response, _metadata, _cache_hit = result
            else:
                response, _metadata = result

            if not response:
                return None, None, False
            text = str(response).strip()
            ordering = None
            time_text = ""
            time_request = False
            m_time = re.search(r"time\s*=\s*([^\s;]+)", text, re.IGNORECASE)
            if m_time:
                time_text = m_time.group(1).strip()
            m_order = re.search(r"ordering\s*=\s*(earliest|latest|none)", text, re.IGNORECASE)
            if m_order:
                val = m_order.group(1).lower()
                if val in ("earliest", "latest"):
                    ordering = val
            m_request = re.search(r"time_request\s*=\s*(yes|no)", text, re.IGNORECASE)
            if m_request:
                time_request = m_request.group(1).lower() == "yes"
            if ordering is None:
                if re.search(r"\b(earliest|oldest|first time|first ever|initially)\b", text, re.IGNORECASE):
                    ordering = "earliest"
                elif re.search(r"\b(latest|newest|most recent|last time|most recently)\b", text, re.IGNORECASE):
                    ordering = "latest"

            if not time_text:
                if re.search(r"\bNONE\b", text, re.IGNORECASE):
                    return None, ordering, time_request
                time_text = text

            if re.search(r"\bNONE\b", time_text, re.IGNORECASE):
                return None, ordering, time_request
            if re.search(r"\b(NOW|CURRENT|TODAY)\b", time_text, re.IGNORECASE):
                return QueryTime.now(), ordering, time_request

            dt = parse_time_text(
                happen_time=time_text,
                obs_time=now.isoformat(),
                mode="happen_else_obs",
                reference_time=now,
            )
            if dt is None:
                return None, ordering, time_request
            unix = time_to_scalar(dt)
            if unix <= 0:
                return None, ordering, time_request
            return QueryTime(unix_seconds=unix), ordering, time_request
        except Exception:
            return None, None, False

    def _apply_tkge_tunnel(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        query_time=None,
        temporal_ordering: str | None = None,
        time_request: bool = False,
    ) -> np.ndarray:
        """
        Structural tunnel (TKGE):
        - Link query -> entities using text embeddings (entity store)
        - Collect candidate facts mentioning linked entities
        - Score candidates structurally with TKGE
        - Merge into query_fact_scores (used by existing reranking + PPR logic)
        """
        self.last_tkge_debug = None
        if query_fact_scores.size == 0:
            return query_fact_scores
        if not self.fact_tuples or not self.entity_texts or self.entity_embeddings.size == 0:
            return query_fact_scores

        try:
            from .kge.entity_linker import EntityLinker
            from .kge.time_utils import QueryTime, parse_time_text, time_to_scalar

            # Match the entity store embedding mode (no instruction).
            q_emb = self.embedding_model.batch_encode([query])[0]
            linker = EntityLinker(top_k=self.global_config.tkge_entity_top_k)
            linked = linker.link(q_emb, self.entity_texts, self.entity_embeddings)
            if not linked:
                return query_fact_scores

            linked_entities = [text_processing(x[0]) for x in linked]
            cand_indices: Set[int] = set()
            for ent in linked_entities:
                cand_indices |= self.entity_to_fact_indices.get(ent, set())
            if not cand_indices:
                return query_fact_scores

            cand_indices = list(cand_indices)[: self.global_config.tkge_candidate_fact_top_k]
            cand_triples = [self.fact_tuples[i] for i in cand_indices]

            qtime = None
            time_mode = "query_time"
            use_fact_happen = False
            if query_time is None and bool(time_request):
                if temporal_ordering in {"earliest", "latest"}:
                    use_fact_happen = True
                else:
                    # Only enable fact-time scoring when a relation has multiple distinct timestamps.
                    time_buckets: dict[tuple[str, str], set[str]] = {}
                    for idx, triple in zip(cand_indices, cand_triples):
                        happen = ""
                        obs = ""
                        if idx < len(self.fact_happen_times):
                            happen = self.fact_happen_times[int(idx)]
                        if not happen:
                            continue
                        if idx < len(self.fact_observed_times):
                            obs = self.fact_observed_times[int(idx)]
                        ref = None
                        if obs:
                            try:
                                ref = datetime.fromisoformat(obs.replace("Z", "+00:00"))
                            except Exception:
                                ref = None
                        dt = parse_time_text(
                            happen_time=str(happen),
                            obs_time="",
                            mode="happen",
                            reference_time=ref,
                        )
                        if dt is None:
                            continue
                        bucket = time_buckets.setdefault((triple[0], triple[1]), set())
                        bucket.add(dt.date().isoformat())
                        if len(bucket) >= 2:
                            use_fact_happen = True
                            break
            if query_time is not None or not use_fact_happen:
                qtime = query_time if query_time is not None else QueryTime.now()
                time_mode = "query_time" if query_time is not None else "now"
                struct_scores = self.tkge_retriever.score(cand_triples, query_time=qtime)
            else:
                time_mode = "fact_happen"
                struct_scores_list = []
                for idx, triple in zip(cand_indices, cand_triples):
                    happen = ""
                    obs = ""
                    if idx < len(self.fact_happen_times):
                        happen = self.fact_happen_times[int(idx)]
                    if idx < len(self.fact_observed_times):
                        obs = self.fact_observed_times[int(idx)]
                    ref = None
                    if obs:
                        try:
                            ref = datetime.fromisoformat(obs.replace("Z", "+00:00"))
                        except Exception:
                            ref = None
                    dt = parse_time_text(
                        happen_time=str(happen),
                        obs_time="",
                        mode="happen",
                        reference_time=ref,
                    )
                    if dt is None:
                        struct_scores_list.append(float("-inf"))
                        continue
                    unix = time_to_scalar(dt)
                    if unix <= 0:
                        struct_scores_list.append(float("-inf"))
                        continue
                    qtime_fact = QueryTime(unix_seconds=unix)
                    struct_scores_list.append(float(self.tkge_retriever.score([triple], query_time=qtime_fact)[0]))
                struct_scores = np.asarray(struct_scores_list, dtype=np.float32)
            if struct_scores.size == 0 or np.all(np.isneginf(struct_scores)):
                if self.tkge_retriever and getattr(self.tkge_retriever, "verbose", 0) >= 1:
                    print(f"[TKGE-tunnel] all scores -inf, skipping temporal boost "
                          f"(candidates={len(cand_indices)} mode={time_mode})")
                return query_fact_scores

            finite = np.isfinite(struct_scores)
            if not finite.any():
                return query_fact_scores

            n_inf = int((~finite).sum())
            n_finite = int(finite.sum())
            if self.tkge_retriever and getattr(self.tkge_retriever, "verbose", 0) >= 1:
                finite_scores = struct_scores[finite]
                print(f"[TKGE-tunnel] mode={time_mode} candidates={len(cand_indices)} "
                      f"scored={n_finite} missing={n_inf} "
                      f"score_range=[{finite_scores.min():.1f}, {finite_scores.max():.1f}]")

            struct_raw = struct_scores.copy()
            s = struct_scores.copy()
            s[~finite] = np.min(s[finite])
            s = min_max_normalize(s)

            # Compute per-candidate α_r to weight the temporal boost.
            # Static relations (α_r→0) get minimal boost; dynamic relations (α_r→1) get full boost.
            alpha_weights = np.ones(len(cand_indices), dtype=np.float32)
            try:
                import torch
                rel_texts = [self.fact_tuples[i][1] for i in cand_indices]
                rel_embs = self.tkge_retriever.relation_embedder(rel_texts)
                rel_tensor = torch.tensor(rel_embs, dtype=torch.float32)
                with torch.no_grad():
                    alpha_vals = self.tkge_retriever.encoder.model.time_gate_alpha(rel_tensor)
                alpha_weights = alpha_vals.squeeze(-1).cpu().numpy()
                if self.tkge_retriever and getattr(self.tkge_retriever, "verbose", 0) >= 1:
                    print(f"[TKGE-tunnel] alpha_r weighting: "
                          f"mean={alpha_weights.mean():.3f} min={alpha_weights.min():.3f} "
                          f"max={alpha_weights.max():.3f}")
            except Exception:
                pass  # fallback: uniform weight (no α_r gating)

            boost = np.zeros_like(query_fact_scores, dtype=np.float32)
            for idx, score, alpha in zip(cand_indices, s.tolist(), alpha_weights.tolist()):
                boost[int(idx)] = float(score) * float(alpha)
            # Track top structural candidates (global indices) for candidate union in reranking.
            try:
                struct_order = np.argsort(s)[::-1]
                struct_top_k = int(getattr(self.global_config, "linking_top_k", 0) or 0)
                if struct_top_k <= 0:
                    struct_top_k = min(50, len(struct_order))
                struct_top_indices = [int(cand_indices[int(i)]) for i in struct_order[:struct_top_k]]
            except Exception:
                struct_top_indices = []

            # Fuse semantic fact scores with structural scores using multiplicative gating.
            # This prevents "right time, wrong topic" candidates from being boosted when semantic score is low.
            beta = float(self.global_config.tkge_weight)
            base = min_max_normalize(query_fact_scores.astype(np.float32, copy=False))
            merged = base * (1.0 + beta * boost)
            merged = min_max_normalize(merged)
            # Store debug info for scripts/analysis.
            top_dbg = sorted(
                [(self.fact_tuples[i], float(boost[int(i)])) for i in cand_indices],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            debug_top_k = int(getattr(self.global_config, "tkge_debug_top_k", 3) or 0)
            debug_shift_days = int(getattr(self.global_config, "tkge_debug_shift_days", 30) or 0)
            debug_candidates = []
            rotation_debug = []
            order_before = []
            order_after = []
            rank_deltas = []
            time_recall = {}
            extracted_query_time = None
            if query_time is not None and getattr(query_time, "unix_seconds", None):
                try:
                    extracted_query_time = datetime.fromtimestamp(
                        float(query_time.unix_seconds),
                        tz=timezone.utc,
                    ).date().isoformat()
                except Exception:
                    extracted_query_time = None
            if debug_top_k > 0 and getattr(self.tkge_retriever, "verbose", 0) >= 3:
                order = np.argsort(s)[::-1]
                for local_idx in order[:debug_top_k]:
                    fact_idx = int(cand_indices[int(local_idx)])
                    debug_candidates.append(
                        {
                            "fact": self.fact_tuples[fact_idx],
                            "base_score": float(base[fact_idx]),
                            "struct_score": float(s[int(local_idx)]),
                            "struct_raw": float(struct_raw[int(local_idx)]),
                            "fused_score": float(merged[fact_idx]),
                        }
                    )
                    if debug_shift_days > 0 and self.tkge_retriever is not None:
                        shift = float(debug_shift_days) * 86400.0
                        base_time = qtime
                        if base_time is None and time_mode == "fact_happen":
                            happen = ""
                            obs = ""
                            if fact_idx < len(self.fact_happen_times):
                                happen = self.fact_happen_times[fact_idx]
                            if fact_idx < len(self.fact_observed_times):
                                obs = self.fact_observed_times[fact_idx]
                            ref = None
                            if obs:
                                try:
                                    ref = datetime.fromisoformat(obs.replace("Z", "+00:00"))
                                except Exception:
                                    ref = None
                            dt = parse_time_text(
                                happen_time=str(happen),
                                obs_time="",
                                mode="happen",
                                reference_time=ref,
                            )
                            if dt is not None:
                                unix = time_to_scalar(dt)
                                if unix > 0:
                                    base_time = QueryTime(unix_seconds=unix)
                        if base_time is None:
                            continue
                        q_unix = float(getattr(base_time, "unix_seconds", 0.0))
                        q_plus = QueryTime(unix_seconds=q_unix + shift)
                        q_minus = QueryTime(unix_seconds=q_unix - shift)
                        score_now = float(self.tkge_retriever.score([self.fact_tuples[fact_idx]], query_time=base_time)[0])
                        score_plus = float(self.tkge_retriever.score([self.fact_tuples[fact_idx]], query_time=q_plus)[0])
                        score_minus = float(self.tkge_retriever.score([self.fact_tuples[fact_idx]], query_time=q_minus)[0])
                        rotation_debug.append(
                            {
                                "fact": self.fact_tuples[fact_idx],
                                "anchor_time": datetime.fromtimestamp(q_unix, tz=timezone.utc).date().isoformat(),
                                "shift_days": int(debug_shift_days),
                                "score_at_query_time": score_now,
                                "score_at_query_plus": score_plus,
                                "score_at_query_minus": score_minus,
                            }
                        )
                # Order change diagnostics (before vs after TKGE fusion).
                pre_order = np.argsort(base[cand_indices])[::-1]
                post_order = np.argsort(merged[cand_indices])[::-1]
                pre_rank = {int(cand_indices[int(i)]): int(r) for r, i in enumerate(pre_order)}
                post_rank = {int(cand_indices[int(i)]): int(r) for r, i in enumerate(post_order)}
                top_pre = list(pre_order[:debug_top_k])
                top_post = list(post_order[:debug_top_k])
                for local_idx in top_pre:
                    fact_idx = int(cand_indices[int(local_idx)])
                    chunk_ids = []
                    if fact_idx < len(self.fact_chunk_ids):
                        chunk_ids = self.fact_chunk_ids[fact_idx] or []
                    order_before.append(
                        {
                            "rank": int(pre_rank[fact_idx]) + 1,
                            "fact_idx": fact_idx,
                            "fact": self.fact_tuples[fact_idx],
                            "base_score": float(base[fact_idx]),
                            "happen_time": self.fact_happen_times[fact_idx] if fact_idx < len(self.fact_happen_times) else "",
                            "chunk_ids": chunk_ids,
                        }
                    )
                for local_idx in top_post:
                    fact_idx = int(cand_indices[int(local_idx)])
                    chunk_ids = []
                    if fact_idx < len(self.fact_chunk_ids):
                        chunk_ids = self.fact_chunk_ids[fact_idx] or []
                    order_after.append(
                        {
                            "rank": int(post_rank[fact_idx]) + 1,
                            "fact_idx": fact_idx,
                            "fact": self.fact_tuples[fact_idx],
                            "fused_score": float(merged[fact_idx]),
                            "happen_time": self.fact_happen_times[fact_idx] if fact_idx < len(self.fact_happen_times) else "",
                            "chunk_ids": chunk_ids,
                        }
                    )
                moved = []
                for fact_idx in set([int(cand_indices[int(i)]) for i in top_pre + top_post]):
                    moved.append(
                        {
                            "fact": self.fact_tuples[fact_idx],
                            "rank_before": int(pre_rank.get(fact_idx, -1)) + 1,
                            "rank_after": int(post_rank.get(fact_idx, -1)) + 1,
                            "delta": int(pre_rank.get(fact_idx, 0)) - int(post_rank.get(fact_idx, 0)),
                        }
                    )
                # Time-aware recall summary: how many time-stamped facts are in top-K before/after,
                # and (if explicit query_time) average absolute day distance to query time.
                def _fact_time_days(idx: int) -> float | None:
                    happen = self.fact_happen_times[idx] if idx < len(self.fact_happen_times) else ""
                    if not happen:
                        return None
                    obs = self.fact_observed_times[idx] if idx < len(self.fact_observed_times) else ""
                    ref = None
                    if obs:
                        try:
                            ref = datetime.fromisoformat(obs.replace("Z", "+00:00"))
                        except Exception:
                            ref = None
                    dt = parse_time_text(
                        happen_time=str(happen),
                        obs_time="",
                        mode="happen",
                        reference_time=ref,
                    )
                    if dt is None or qtime is None:
                        return None
                    try:
                        q_dt = datetime.fromtimestamp(float(qtime.unix_seconds), tz=timezone.utc)
                    except Exception:
                        return None
                    return abs((dt - q_dt).days)

                top_pre_fact_idx = [int(cand_indices[int(i)]) for i in top_pre]
                top_post_fact_idx = [int(cand_indices[int(i)]) for i in top_post]
                pre_time = [i for i in top_pre_fact_idx if i < len(self.fact_happen_times) and self.fact_happen_times[i]]
                post_time = [i for i in top_post_fact_idx if i < len(self.fact_happen_times) and self.fact_happen_times[i]]
                time_recall = {
                    "k": int(debug_top_k),
                    "time_facts_before": int(len(pre_time)),
                    "time_facts_after": int(len(post_time)),
                }
                if qtime is not None and time_mode == "query_time":
                    pre_days = [d for d in (_fact_time_days(i) for i in top_pre_fact_idx) if d is not None]
                    post_days = [d for d in (_fact_time_days(i) for i in top_post_fact_idx) if d is not None]
                    if pre_days:
                        time_recall["avg_abs_days_before"] = float(sum(pre_days) / len(pre_days))
                        time_recall["min_abs_days_before"] = float(min(pre_days))
                    if post_days:
                        time_recall["avg_abs_days_after"] = float(sum(post_days) / len(post_days))
                        time_recall["min_abs_days_after"] = float(min(post_days))
                rank_deltas = sorted(moved, key=lambda x: -x["delta"])
            self.last_tkge_debug = {
                "query": query,
                "extracted_query_time": extracted_query_time,
                "linked_entities": linked,
                "candidate_facts": len(cand_indices),
                "struct_top_indices": struct_top_indices,
                "top_boosted_facts": top_dbg,
                "candidate_scores": debug_candidates,
                "rotation_debug": rotation_debug,
                "order_before": order_before,
                "order_after": order_after,
                "rank_deltas": rank_deltas,
                "time_recall": time_recall,
                "tkge_weight": float(self.global_config.tkge_weight),
                "query_time_unix": float(getattr(qtime, "unix_seconds", 0.0)) if qtime is not None else 0.0,
                "time_request": bool(time_request),
                "time_mode": time_mode,
                "fusion": "multiplicative_gate",
            }
            # Keep verbose diagnostics inside the merged per-sample log to avoid duplicate lines.
            # (Detailed debug is still available via the returned payload.)
            return merged
        except Exception as exc:
            logger.error(f"TKGE tunnel failed for query: {exc}")
            raise

    def rag_qa(self,
               queries: List[str|QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using the RoMem framework.
        """
        # Retrieving (if necessary)
        if not isinstance(queries[0], QuerySolution):
            queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None) -> List[QuerySolution]:
        """
        Performs retrieval using a standard DPR (dense passage retrieval) framework.

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query.

        Returns:
            List[QuerySolution]: Retrieved documents and scores for each query.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries), disable=len(queries) <= 1):
            logger.debug('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.
        """
        # Retrieving (if necessary)
        if not isinstance(queries[0], QuerySolution):
            queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts", disable=len(queries) <= 1):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading", disable=len(all_qa_messages) <= 1)]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response", disable=len(queries) <= 1):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.debug("Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples), disable=len(chunk_ids) <= 2):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.debug("Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids), disable=len(chunk_ids) <= 2):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys), disable=len(query_node_key2knn_node_keys) <= 2):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                timed = []
                for tt in triple_results_dict[chunk_key].timed_triples:
                    timed.append(
                        {
                            "triple": [str(tt.triple[0]), str(tt.triple[1]), str(tt.triple[2])],
                            "happen_time": str(tt.happen_time),
                            "system_time": str(tt.system_time),
                        }
                    )
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples,
                                 'extracted_timed_triples': timed}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': [],
                                 'extracted_timed_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph_manager.save()
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())
        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        # TKGE tunnel helpers: align text-space entity strings and fact triples with store ordering.
        self.entity_texts = []
        try:
            entity_rows = self.entity_embedding_store.get_rows(self.entity_node_keys)
            self.entity_texts = [entity_rows[k]["content"] for k in self.entity_node_keys]
        except Exception:
            self.entity_texts = []

        self.fact_tuples = []
        self.entity_to_fact_indices = {}
        self.fact_happen_times = []
        self.fact_observed_times = []
        self.fact_chunk_ids: List[List[str]] = []
        try:
            fact_rows = self.fact_embedding_store.get_rows(self.fact_node_keys)
            for idx, fid in enumerate(self.fact_node_keys):
                try:
                    row = fact_rows.get(fid, {})
                    meta = row.get("meta") or {}
                    self.fact_happen_times.append(str(meta.get("happen_time") or ""))
                    self.fact_observed_times.append(str(meta.get("observed_time") or ""))
                    chunk_ids = meta.get("chunk_ids") or []
                    if isinstance(chunk_ids, str):
                        chunk_ids = [chunk_ids]
                    self.fact_chunk_ids.append([str(cid) for cid in chunk_ids])
                    t = eval(fact_rows[fid]["content"])
                    if isinstance(t, tuple) and len(t) == 3:
                        self.fact_tuples.append((str(t[0]), str(t[1]), str(t[2])))
                    else:
                        self.fact_tuples.append(("", "", ""))
                except Exception:
                    self.fact_happen_times.append("")
                    self.fact_observed_times.append("")
                    self.fact_chunk_ids.append([])
                    self.fact_tuples.append(("", "", ""))
            for idx, (h, r, t) in enumerate(self.fact_tuples):
                if not h or not t:
                    continue
                self.entity_to_fact_indices.setdefault(h, set()).add(idx)
                self.entity_to_fact_indices.setdefault(t, set()).add(idx)
        except Exception:
            self.fact_tuples = []
            self.entity_to_fact_indices = {}
            self.fact_happen_times = []
            self.fact_observed_times = []
            self.fact_chunk_ids = []

        if self.tkge_tunnel_enabled and self.tkge_retriever is not None and self.fact_tuples:
            try:
                # Ensure the KGE model is aware of all currently indexed facts.
                if self.global_config.tkge_temporal_mode == "romem":
                    timed_facts = []
                    for fid in self.fact_node_keys:
                        row = fact_rows.get(fid, {})
                        meta = row.get("meta") or {}
                        try:
                            t = eval(row.get("content", ""))
                        except Exception:
                            t = ()
                        if not isinstance(t, tuple) or len(t) != 3:
                            continue
                        timed_facts.append(
                            TimedTriple(
                                triple=(str(t[0]), str(t[1]), str(t[2])),
                                happen_time=str(meta.get("happen_time", "")),
                                system_time=str(meta.get("observed_time", "")),
                            )
                        )
                    if timed_facts:
                        self.tkge_retriever.update(timed_facts)
                else:
                    self.tkge_retriever.update(self.fact_tuples)
            except Exception as exc:
                logger.error(f"TKGE initialization from existing facts failed: {exc}")
                raise

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        for doc in all_openie_info:
            cid = doc.get("idx")
            if cid is None or cid not in triple_results_dict:
                continue
            triples = flatten_facts([triple_results_dict[cid].triples])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([cid]))

        if self.ent_node_to_chunk_ids is None:
            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            timed_triples=[]
                        )

            # prepare data_store
            chunk_timed_triples = [triple_results_dict[chunk_id].timed_triples for chunk_id in self.passage_node_keys]
            chunk_triples = [
                [text_processing(list(tt.triple)) for tt in timed_list] for timed_list in chunk_timed_triples
            ]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.debug(f"Encoding {len(all_query_strings)} queries (fact + passage).")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # Choose top-ranked nodes in linking_score_map.
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # Only keep the top-k phrases in all_phrase_weights.
        # Some phrases may not exist as nodes in the current graph; filter them out to avoid inconsistent states.
        filtered_linking_score_map: Dict[str, float] = {}
        top_k_phrases_keys: set[str] = set()
        for phrase, score in linking_score_map.items():
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
            if phrase_id is None:
                continue
            filtered_linking_score_map[phrase] = score
            top_k_phrases_keys.add(phrase_key)
        linking_score_map = filtered_linking_score_map

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        # In practice, multiple top phrases can map to the same node weight (or be suppressed during pruning),
        # so we avoid a hard assertion here and instead keep the filtered map consistent with existing nodes.
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        # Avoid NaNs when a phrase_id never occurred (number_of_occurs==0).
        phrase_weights = np.divide(
            phrase_weights,
            number_of_occurs,
            out=np.zeros_like(phrase_weights),
            where=number_of_occurs > 0,
        )

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Running PPR algorithm based on the passage and phrase weights previously assigned.
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights, damping=self.global_config.damping, weights="weight"
        )
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        query_time=None,
        temporal_ordering: str | None = None,
        time_request: bool = False,
        observed_time_cutoff: str | None = None,
        observed_time_bounds: tuple[str | None, str | None] | None = None,
    ) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        candidate_pool_k = link_top_k
        if self.tkge_tunnel_enabled and self.tkge_retriever is not None:
            kge_pool = int(getattr(self.global_config, "fact_candidate_k_for_kge", 0) or 0)
            if kge_pool > candidate_pool_k:
                candidate_pool_k = kge_pool
        if query_time is None and bool(getattr(self.global_config, "enable_tkge_tunnel", False)) and not time_request:
            query_time = self._extract_query_time(query)

        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}

        try:
            observed_lower_unix = None
            observed_upper_unix = None
            observed_rows = None
            if observed_time_bounds or observed_time_cutoff:
                try:
                    from .kge.time_utils import parse_time_text, time_to_scalar

                    if observed_time_bounds:
                        lower_text, upper_text = observed_time_bounds
                        if lower_text:
                            lower_dt = parse_time_text(happen_time="", obs_time=str(lower_text), mode="obs")
                            if lower_dt is None:
                                lower_dt = parse_time_text(happen_time=str(lower_text), obs_time="", mode="happen")
                            if lower_dt is not None:
                                if lower_dt.tzinfo is None:
                                    lower_dt = lower_dt.replace(tzinfo=timezone.utc)
                                observed_lower_unix = time_to_scalar(lower_dt)
                        if upper_text:
                            upper_dt = parse_time_text(happen_time="", obs_time=str(upper_text), mode="obs")
                            if upper_dt is None:
                                upper_dt = parse_time_text(happen_time=str(upper_text), obs_time="", mode="happen")
                            if upper_dt is not None:
                                if upper_dt.tzinfo is None:
                                    upper_dt = upper_dt.replace(tzinfo=timezone.utc)
                                upper_text_str = str(upper_text)
                                if len(upper_text_str) == 10 and upper_text_str[4] == "-" and upper_text_str[7] == "-":
                                    upper_dt = upper_dt + timedelta(days=1) - timedelta(seconds=1)
                                observed_upper_unix = time_to_scalar(upper_dt)
                    elif observed_time_cutoff:
                        cutoff_text = str(observed_time_cutoff).strip()
                        cutoff_dt = parse_time_text(happen_time="", obs_time=cutoff_text, mode="obs")
                        if cutoff_dt is None:
                            cutoff_dt = parse_time_text(happen_time=cutoff_text, obs_time="", mode="happen")
                        if cutoff_dt is not None:
                            if cutoff_dt.tzinfo is None:
                                cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)
                            if len(cutoff_text) == 10 and cutoff_text[4] == "-" and cutoff_text[7] == "-":
                                cutoff_dt = cutoff_dt + timedelta(days=1) - timedelta(seconds=1)
                            observed_upper_unix = time_to_scalar(cutoff_dt)

                    if observed_lower_unix is not None or observed_upper_unix is not None:
                        observed_rows = self.fact_embedding_store.get_rows(self.fact_node_keys)
                except Exception:
                    observed_lower_unix = None
                    observed_upper_unix = None
                    observed_rows = None

            scores_order = np.argsort(query_fact_scores)[::-1].tolist()
            if observed_lower_unix is None and observed_upper_unix is None:
                if len(scores_order) <= candidate_pool_k:
                    candidate_fact_indices = scores_order
                else:
                    candidate_fact_indices = scores_order[:candidate_pool_k]
            else:
                candidate_fact_indices = []
                for idx in scores_order:
                    if len(candidate_fact_indices) >= candidate_pool_k:
                        break
                    fid = self.fact_node_keys[idx]
                    meta = (observed_rows.get(fid) or {}).get("meta") or {}
                    obs_text = str(meta.get("observed_time") or "")
                    if not obs_text:
                        candidate_fact_indices.append(idx)
                        continue
                    dt = parse_time_text(happen_time="", obs_time=obs_text, mode="obs")
                    if dt is None:
                        candidate_fact_indices.append(idx)
                        continue
                    obs_unix = time_to_scalar(dt)
                    if observed_lower_unix is not None and obs_unix < observed_lower_unix:
                        continue
                    if observed_upper_unix is not None and obs_unix > observed_upper_unix:
                        continue
                    candidate_fact_indices.append(idx)

            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]

            # Rerank the facts (LLM filter) to preserve HippoRAG's semantic anchoring.
            if getattr(self.global_config, "use_llm_fact_filter", True):
                top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(
                    query,
                    candidate_facts,
                    candidate_fact_indices,
                    len_after_rerank=link_top_k,
                )
            else:
                top_k_fact_indices = candidate_fact_indices[:link_top_k]
                top_k_facts = candidate_facts[:link_top_k]
                reranker_dict = {"confidence": None}

            # NOTE: No hard year-matching filter here. Temporal discrimination is handled
            # by the KGE fusion in _apply_tkge_tunnel (multiplicative gating). Facts
            # without parseable timestamps keep their pure semantic score (boost=0).

            # Ordering queries: keep all facts, but apply temporal ordering by happen_time.
            if temporal_ordering is not None and top_k_fact_indices:
                try:
                    from .kge.time_utils import parse_time_text

                    fact_ids = [self.fact_node_keys[idx] for idx in top_k_fact_indices]
                    rows = self.fact_embedding_store.get_rows(fact_ids)
                    sortable = []
                    unsorted = []
                    for idx, fact, fid in zip(top_k_fact_indices, top_k_facts, fact_ids):
                        meta = (rows.get(fid) or {}).get("meta") or {}
                        happen_time = str(meta.get("happen_time") or "")
                        obs_time = str(meta.get("observed_time") or "")
                        dt = parse_time_text(happen_time=happen_time, obs_time=obs_time, mode="happen")
                        if dt is None:
                            unsorted.append((idx, fact))
                        else:
                            sortable.append((dt.timestamp(), idx, fact))
                    if sortable:
                        reverse = temporal_ordering == "latest"
                        sortable.sort(key=lambda x: x[0], reverse=reverse)
                        ordered = [(idx, fact) for _, idx, fact in sortable]
                        ordered.extend(unsorted)
                        top_k_fact_indices = [idx for idx, _ in ordered]
                        top_k_facts = [fact for _, fact in ordered]
                except Exception:
                    pass

            if len(top_k_fact_indices) > link_top_k:
                top_k_fact_indices = top_k_fact_indices[:link_top_k]
                top_k_facts = top_k_facts[:link_top_k]

            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(
        self,
        reset_prob: np.ndarray,
        damping: float = 0.5,
        weights: str | list[float] | None = "weight",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights=weights,
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
