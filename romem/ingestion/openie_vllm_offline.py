import json
from datetime import datetime, timezone
from typing import Dict, Tuple

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from romem.utils.misc_utils import NerRawOutput, TripleRawOutput, TimedTriple
from romem.kge.time_utils import normalize_time_text
from romem.utils.logging_utils import get_logger
from romem.prompts import PromptTemplateManager
from romem.llm.vllm_offline import VLLMOffline
from romem.utils.llm_utils import filter_invalid_triples

logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_ids = list(chunks.keys())
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        chunk_obs_times = [
            (chunks[chunk_id].get("meta") or {}).get("observed_time")
            for chunk_id in chunk_ids
        ]

        ner_input_messages = [self.prompt_template_manager.render(name='ner', passage=p) for p in chunk_passages.values()]
        ner_output, ner_output_metadata = self.llm_model.batch_infer(ner_input_messages, json_template='ner', max_tokens=512)

        default_observed_time = datetime.now(timezone.utc).isoformat()
        triple_extract_input_messages = [
            self.prompt_template_manager.render(
                name='triple_extraction',
                passage=passage,
                named_entity_json=named_entities,
                observed_time=obs_time or default_observed_time,
            )
            for passage, named_entities, obs_time in zip(
                chunk_passages.values(),
                ner_output,
                chunk_obs_times,
            )
        ]
        triple_output, triple_output_metadata = self.llm_model.batch_infer(triple_extract_input_messages, json_template='triples', max_tokens=2048)

        ner_raw_outputs = []
        for idx, ner_output_instance in enumerate(ner_output):
            chunk_id = chunk_ids[idx]
            response = ner_output_instance
            try:
                unique_entities = json.loads(response)["named_entities"]
            except Exception as e:
                unique_entities = []
                logger.warning(f"Could not parse response from OpenIE: {e}")
            if len(unique_entities) == 0:
                logger.warning("No entities extracted for chunk_id: {}".format(chunk_id))
            ner_raw_output = NerRawOutput(chunk_id, response, unique_entities, {})
            ner_raw_outputs.append(ner_raw_output)
        ner_results_dict = {chunk_key: ner_raw_output for chunk_key, ner_raw_output in zip(chunk_ids, ner_raw_outputs)}

        triple_raw_outputs = []
        for idx, triple_output_instance in enumerate(triple_output):
            chunk_id = chunk_ids[idx]
            response = triple_output_instance
            text_time_list = []
            triples = []
            timed_triples = []
            observed_time = chunk_obs_times[idx] or default_observed_time
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict) and "triples" in parsed:
                    raw_triples = parsed.get("triples", [])
                elif isinstance(parsed, list):
                    raw_triples = parsed
                else:
                    raw_triples = []
                for item in raw_triples:
                    head = relation = tail = None
                    text_time = ""
                    obs_time = observed_time
                    if isinstance(item, dict):
                        head = item.get("head") or item.get("subject")
                        relation = item.get("relation") or item.get("predicate")
                        tail = item.get("tail") or item.get("object")
                        text_time = item.get("text_time") or ""
                        obs_time = item.get("observed_time") or observed_time
                    elif isinstance(item, list) and len(item) >= 3:
                        head, relation, tail = item[0], item[1], item[2]
                    if not (head and relation and tail):
                        continue
                    if text_time and (head == text_time or tail == text_time):
                        continue
                    triple_tuple = (str(head), str(relation), str(tail))
                    norm_time = normalize_time_text(text_time, obs_time)
                    triples.append(triple_tuple)
                    timed_triples.append(TimedTriple(triple=triple_tuple, happen_time=str(norm_time), system_time=str(obs_time)))
            except Exception as e:
                logger.warning(f"Could not parse response from OpenIE: {e}")

            # Deduplicate and convert to lists to match legacy expectations
            seen = set()
            dedup_triples = []
            dedup_timed = []
            for t, tt in zip(triples, timed_triples):
                if t in seen:
                    continue
                seen.add(t)
                dedup_triples.append(list(t))
                dedup_timed.append(tt)

            dedup_triples = filter_invalid_triples(triples=dedup_triples)
            if len(triples) == 0:
                logger.warning("No triples extracted for chunk_id: {}".format(chunk_id))
            metadata = {
                "observed_time": observed_time,
                "text_time_list": [{"text_time": tt.happen_time, "observed_time": tt.system_time} for tt in dedup_timed],
            }
            triple_raw_output = TripleRawOutput(
                chunk_id=chunk_id,
                response=response,
                metadata=metadata,
                timed_triples=dedup_timed,
            )
            triple_raw_outputs.append(triple_raw_output)
        triple_results_dict = {chunk_key: triple_raw_output for chunk_key, triple_raw_output in zip(chunk_ids, triple_raw_outputs)}

        return ner_results_dict, triple_results_dict
