import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timezone

from romem.prompts import PromptTemplateManager
from romem.utils.logging_utils import get_logger
from romem.utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from romem.utils.misc_utils import TripleRawOutput, NerRawOutput, TimedTriple
from romem.kge.time_utils import normalize_time_text
from romem.llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict, total=False):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]
    meta: Dict[str, Any]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return eval(match.group())["named_entities"]


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, max_workers: int | None = None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))
            logger.debug("[RoMem NER] chunk=%s entities=%d raw_response=%s", chunk_key, len(unique_entities), real_response[:500])

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(
        self,
        chunk_key: str,
        passage: str,
        named_entities: List[str],
        observed_time_override: str | None = None,
    ) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            """
            Support both HippoRAG's JSON object with a `triples` field and a raw JSON list.
            """
            # Try strict JSON parse first
            try:
                parsed = json.loads(real_response)
                if isinstance(parsed, dict) and "triples" in parsed:
                    return parsed.get("triples", [])
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

            # Fallback to regex extraction from free-form text
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            try:
                return eval(match.group())["triples"]
            except Exception:
                return []

        # PREPROCESSING
        observed_time = observed_time_override or datetime.now(timezone.utc).isoformat()
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities}),
            observed_time=observed_time,
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted = _extract_triples_from_response(real_response)
            logger.debug("[RoMem Triple] chunk=%s raw_response=%s", chunk_key, real_response[:1000])
            logger.debug("[RoMem Triple] chunk=%s parsed=%d items: %s", chunk_key, len(extracted), json.dumps(extracted[:5], ensure_ascii=False, default=str))

            raw_triplets = []
            raw_timed = []
            for item in extracted:
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
                # Drop triples that use the extracted time expression as an entity.
                if text_time and (head == text_time or tail == text_time):
                    continue
                norm_time = normalize_time_text(text_time, obs_time)
                triple_tuple = (str(head), str(relation), str(tail))
                raw_triplets.append(triple_tuple)
                raw_timed.append(TimedTriple(triple=triple_tuple, happen_time=str(norm_time), system_time=str(obs_time)))

            # Deduplicate while preserving order, aligned across triples and timed metadata.
            seen = set()
            timed_triples = []
            text_time_list = []
            for t, tt in zip(raw_triplets, raw_timed):
                if t in seen:
                    continue
                seen.add(t)
                timed_triples.append(tt)
                text_time_list.append({"text_time": tt.happen_time, "observed_time": tt.system_time})

            # Validate and filter triples, then drop corresponding timed triples.
            validated_order = [
                tuple(t)
                for t in filter_invalid_triples(triples=[list(tt.triple) for tt in timed_triples])
            ]
            timed_by_triple = {tt.triple: tt for tt in timed_triples}
            timed_triples = [timed_by_triple[t] for t in validated_order if t in timed_by_triple]
            text_time_list = [{"text_time": tt.happen_time, "observed_time": tt.system_time} for tt in timed_triples]
            metadata["text_time_list"] = text_time_list
            metadata["observed_time"] = observed_time
            logger.debug("[RoMem Triple] chunk=%s extracted=%d -> validated=%d timed_triples=%s", chunk_key, len(extracted), len(timed_triples), [(tt.triple, tt.happen_time) for tt in timed_triples[:5]])

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                timed_triples=[],
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            timed_triples=timed_triples,
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        chunk_obs = {
            chunk_key: (chunk.get("meta") or {}).get("observed_time")
            for chunk_key, chunk in chunks.items()
        }

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER", disable=len(ner_futures) <= 1)
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                    chunk_obs.get(ner_result.chunk_id),
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples", disable=len(re_futures) <= 1)
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
