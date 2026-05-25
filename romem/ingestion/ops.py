import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from romem.llm.base import BaseLLM
from romem.utils.llm_utils import TextChatMessage, fix_broken_generated_json
from romem.utils.misc_utils import compute_mdhash_id, text_processing

logger = logging.getLogger(__name__)

@dataclass
class FactRecord:
    fact_id: str
    triple: Tuple[str, str, str]
    happen_time: str
    observed_time: str
    status: str


@dataclass
class MemoryDecision:
    decision: str
    target_ids: List[str]
    confidence: float
    rationale: str


class MemoryOpsManager:
    def __init__(self, working_dir: str, llm_model: BaseLLM, max_candidates: int = 12) -> None:
        self.llm_model = llm_model
        self.max_candidates = max_candidates
        self.registry_path = os.path.join(working_dir, "memory_registry.jsonl")
        self._legacy_registry_path = os.path.join(working_dir, "memory_registry.json")
        self.registry: Dict[str, Any] = {"version": 1, "facts": {}}
        self._load_registry()

    def _load_registry(self) -> None:
        if not os.path.exists(self.registry_path):
            return
        try:
            with open(self.registry_path, "r") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if "facts" in payload:
                        if isinstance(payload.get("facts"), dict):
                            self.registry = payload
                        continue
                    fact_id = payload.get("fact_id")
                    if not fact_id:
                        continue
                    entry = {k: v for k, v in payload.items() if k != "fact_id"}
                    if entry:
                        self.registry["facts"][fact_id] = entry
        except Exception:
            return

    def _entry_for_fact(self, fact_id: str) -> Dict[str, Any]:
        entry = dict(self.registry["facts"].get(fact_id, {}))
        entry["fact_id"] = fact_id
        return entry

    def _append_registry_entries(self, entries: List[Dict[str, Any]]) -> None:
        if not entries:
            return
        try:
            with open(self.registry_path, "a") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry))
                    handle.write("\n")
        except Exception:
            return

    def _fact_id(self, triple: Tuple[str, str, str]) -> str:
        return compute_mdhash_id(str(tuple(triple)), prefix="fact-")

    def _normalize_triple(self, triple: Iterable[Any]) -> Optional[Tuple[str, str, str]]:
        try:
            h, r, t = triple
        except Exception:
            return None
        norm = tuple(text_processing([str(h), str(r), str(t)]))
        if any(not x for x in norm):
            return None
        return norm

    def _parse_fact_row(self, row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        raw = row.get("content")
        if not raw:
            return None
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            return None
        return self._normalize_triple(parsed)

    def reconcile_with_store(self, fact_rows: Dict[str, Dict[str, Any]]) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        new_ids: List[str] = []
        for fid, row in fact_rows.items():
            if fid in self.registry["facts"]:
                continue
            triple = self._parse_fact_row(row)
            if not triple:
                continue
            meta = row.get("meta") or {}
            self.registry["facts"][fid] = {
                "triple": list(triple),
                "status": "active",
                "happen_time": str(meta.get("happen_time", "")),
                "observed_time": str(meta.get("observed_time", "")),
                "last_action": "import",
                "rationale": "",
                "updated_at": now,
            }
            new_ids.append(fid)
        if new_ids:
            self._append_registry_entries([self._entry_for_fact(fid) for fid in new_ids])

    def active_fact_ids(self, fact_ids: Iterable[str]) -> List[str]:
        active = []
        for fid in fact_ids:
            status = self.registry["facts"].get(fid, {}).get("status", "active")
            if status != "inactive":
                active.append(fid)
        return active

    def filter_inactive_timed_triples(
        self,
        chunk_ids: List[str],
        chunk_timed_triples: List[List[Any]],
    ) -> List[List[Any]]:
        filtered: List[List[Any]] = []
        for cid, timed_list in zip(chunk_ids, chunk_timed_triples):
            kept = []
            for tt in timed_list:
                triple = self._normalize_triple(tt.triple)
                if not triple:
                    continue
                fid = self._fact_id(triple)
                status = self.registry["facts"].get(fid, {}).get("status", "active")
                if status != "inactive":
                    kept.append(tt)
            filtered.append(kept)
        return filtered

    def resolve_new_facts(
        self,
        new_chunk_ids: List[str],
        triple_results_dict: Dict[str, Any],
        fact_rows: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, str, str]]:
        self.reconcile_with_store(fact_rows)

        existing_records = self._build_existing_records(fact_rows)
        entity_index = self._build_entity_index(existing_records)
        deactivated_triples: List[Tuple[str, str, str]] = []

        for cid in new_chunk_ids:
            triple_output = triple_results_dict.get(cid)
            if not triple_output:
                continue
            kept = []
            append_entries: List[Dict[str, Any]] = []
            for tt in triple_output.timed_triples:
                triple_norm = self._normalize_triple(tt.triple)
                if not triple_norm:
                    continue
                candidates = self._candidate_facts(triple_norm, entity_index)
                if not candidates:
                    kept.append(tt)
                    fid = self._register_fact(triple_norm, tt.happen_time, tt.system_time, "ADD", "")
                    append_entries.append(self._entry_for_fact(fid))
                    continue
                decision = self._decide_action(triple_norm, tt.happen_time, tt.system_time, candidates)
                if self._apply_decision(decision, triple_norm, tt.happen_time, tt.system_time, append_entries):
                    kept.append(tt)
                deactivated_triples.extend(self._deactivate_targets(decision, candidates, append_entries))
            triple_output.timed_triples = kept
            # Flush after each chunk update to make decisions visible during long ingests.
            self._append_registry_entries(append_entries)

        return deactivated_triples

    def _build_existing_records(self, fact_rows: Dict[str, Dict[str, Any]]) -> List[FactRecord]:
        records: List[FactRecord] = []
        for fid, row in fact_rows.items():
            triple = self._parse_fact_row(row)
            if not triple:
                continue
            meta = row.get("meta") or {}
            status = self.registry["facts"].get(fid, {}).get("status", "active")
            records.append(
                FactRecord(
                    fact_id=fid,
                    triple=triple,
                    happen_time=str(meta.get("happen_time", "")),
                    observed_time=str(meta.get("observed_time", "")),
                    status=status,
                )
            )
        return records

    def _build_entity_index(self, records: List[FactRecord]) -> Dict[str, List[FactRecord]]:
        index: Dict[str, List[FactRecord]] = {}
        for rec in records:
            h, _, t = rec.triple
            index.setdefault(h, []).append(rec)
            index.setdefault(t, []).append(rec)
        return index

    def _candidate_facts(
        self,
        triple: Tuple[str, str, str],
        entity_index: Dict[str, List[FactRecord]],
    ) -> List[FactRecord]:
        h, _, t = triple
        candidates = list(entity_index.get(h, [])) + list(entity_index.get(t, []))
        unique: Dict[str, FactRecord] = {}
        for rec in candidates:
            unique[rec.fact_id] = rec
        ordered = list(unique.values())
        return ordered[: self.max_candidates]

    def _build_prompt(
        self,
        triple: Tuple[str, str, str],
        happen_time: str,
        observed_time: str,
        candidates: List[FactRecord],
    ) -> List[TextChatMessage]:
        cand_lines = []
        for idx, rec in enumerate(candidates):
            cand_lines.append(
                f"F{idx}: triple={rec.triple}, happen_time={rec.happen_time}, status={rec.status}"
            )
        cand_block = "\n".join(cand_lines) if cand_lines else "None"
        user = (
            "New fact:\n"
            f"triple={triple}\n"
            f"happen_time={happen_time}\n"
            "observed_time is ingestion metadata only and must not be used to judge factual validity.\n\n"
            "Existing facts:\n"
            f"{cand_block}\n\n"
            "Decide the memory operation.\n"
            "Do not assume a current date.\n"
            "Use only explicit happen_time values and logical content when judging conflicts.\n"
            "If happen_time is missing or vague for one or both facts, treat time as unknown and do not use it to justify forgetting.\n"
            "Choose one action: ADD, UPDATE, FORGET, MERGE, KEEP_BOTH.\n"
            "Rules:\n"
            "- ADD keeps the new fact and does not change existing facts.\n"
            "- UPDATE keeps the new fact and supersedes one or more existing facts.\n"
            "- FORGET drops the new fact. Use target NEW.\n"
            "- MERGE keeps the new fact and links it with related facts.\n"
            "- KEEP_BOTH keeps the new fact and all existing facts.\n\n"
            "Output format:\n"
            "Return JSON with lowercase keys:\n"
            '{"decision": "<ACTION>", "targets": ["F0"], "confidence": 0.7, "rationale": "<short reason>"}\n'
            "If you cannot output JSON, use:\n"
            "Decision: <ACTION>\n"
            "Targets: <comma-separated ids like F0, F2 or NEW or NONE>\n"
            "Confidence: <0-1>\n"
            "Rationale: <short reason>"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You decide memory operations for a temporal knowledge base. "
                    "Do not assume a current date. "
                    "Observed time is ingestion metadata, not factual time."
                ),
            },
            {"role": "user", "content": user},
        ]

    def _decide_action(
        self,
        triple: Tuple[str, str, str],
        happen_time: str,
        observed_time: str,
        candidates: List[FactRecord],
    ) -> MemoryDecision:
        messages = self._build_prompt(triple, happen_time, observed_time, candidates)
        result = self.llm_model.infer(messages)
        if isinstance(result, (list, tuple)) and len(result) == 3:
            response, _metadata, _cache_hit = result
        else:
            response, _metadata = result
        decision = self._parse_decision(response, candidates)
        return decision

    def _parse_decision(self, text: str, candidates: List[FactRecord]) -> MemoryDecision:
        decision = None
        targets: List[str] = []
        confidence = 0.5
        rationale = ""
        parsed = False

        json_obj = self._try_extract_json(text)
        if isinstance(json_obj, dict):
            lowered = {str(key).lower(): value for key, value in json_obj.items()}
            decision = lowered.get("decision") or lowered.get("action")
            if decision is not None:
                parsed = True
            raw_targets = lowered.get("targets") or lowered.get("target") or []
            if isinstance(raw_targets, str):
                targets = [raw_targets]
            elif isinstance(raw_targets, list):
                targets = [str(t) for t in raw_targets]
            conf = lowered.get("confidence")
            if isinstance(conf, (int, float)):
                confidence = float(conf)
            rationale = str(lowered.get("rationale") or lowered.get("reason") or "")
        else:
            decision_match = re.search(r"(decision|action)\s*[:=]\s*([A-Za-z_ ]+)", text, re.IGNORECASE)
            if decision_match:
                decision = decision_match.group(2).strip()
                parsed = True
            targets_match = re.search(r"targets?\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
            if targets_match:
                targets = [t.strip() for t in re.split(r"[,\s]+", targets_match.group(1)) if t.strip()]
            conf_match = re.search(r"confidence\s*[:=]\s*([0-9.]+)", text, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except Exception:
                    confidence = 0.5
            rationale_match = re.search(r"(rationale|reason)\s*[:=]\s*(.+)", text, re.IGNORECASE)
            if rationale_match:
                rationale = rationale_match.group(2).strip()

        decision = self._normalize_decision(decision)
        targets = self._normalize_targets(targets, candidates)

        if decision in ("UPDATE", "MERGE") and not targets:
            targets = [candidates[0].fact_id]
        if decision == "FORGET" and not targets:
            targets = ["NEW"]

        if not parsed and candidates:
            snippet = re.sub(r"\s+", " ", (text or "")).strip()[:200]
            logger.warning(
                "Memory ops parse fallback; defaulting to ADD. response=%s",
                snippet,
            )
        elif parsed:
            logger.info(
                "Memory ops decision parsed: action=%s targets=%s confidence=%.2f",
                decision,
                ",".join(targets) if targets else "none",
                confidence,
            )

        return MemoryDecision(
            decision=decision,
            target_ids=targets,
            confidence=confidence,
            rationale=rationale,
        )

    def _try_extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            try:
                fixed = fix_broken_generated_json(snippet)
                return json.loads(fixed)
            except Exception:
                return None

    def _normalize_decision(self, decision: Optional[str]) -> str:
        if not decision:
            return "ADD"
        key = decision.strip().upper().replace("-", "_").replace(" ", "_")
        mapping = {
            "KEEP": "KEEP_BOTH",
            "KEEP_BOTH": "KEEP_BOTH",
            "ADD": "ADD",
            "UPDATE": "UPDATE",
            "FORGET": "FORGET",
            "MERGE": "MERGE",
        }
        return mapping.get(key, "ADD")

    def _normalize_targets(self, targets: List[str], candidates: List[FactRecord]) -> List[str]:
        if not targets:
            return []
        id_map = {f"F{idx}": rec.fact_id for idx, rec in enumerate(candidates)}
        triple_map = {str(rec.triple): rec.fact_id for rec in candidates}
        normalized: List[str] = []
        for token in targets:
            if not token:
                continue
            key = token.strip().upper()
            if key in ("NEW", "NONE"):
                normalized.append(key)
                continue
            if token in id_map:
                normalized.append(id_map[token])
                continue
            if token in triple_map:
                normalized.append(triple_map[token])
                continue
            if key in id_map:
                normalized.append(id_map[key])
        return normalized

    def _register_fact(self, triple: Tuple[str, str, str], happen_time: str, observed_time: str, action: str, rationale: str) -> str:
        fid = self._fact_id(triple)
        now = datetime.now(tz=timezone.utc).isoformat()
        self.registry["facts"][fid] = {
            "triple": list(triple),
            "status": "active",
            "happen_time": str(happen_time or ""),
            "observed_time": str(observed_time or ""),
            "last_action": action,
            "rationale": rationale,
            "updated_at": now,
        }
        return fid

    def _apply_decision(
        self,
        decision: MemoryDecision,
        triple: Tuple[str, str, str],
        happen_time: str,
        observed_time: str,
        append_entries: List[Dict[str, Any]],
    ) -> bool:
        if decision.decision == "FORGET" and any(t in ("NEW", "NONE") for t in decision.target_ids):
            fid = self._fact_id(triple)
            self.registry["facts"][fid] = {
                "triple": list(triple),
                "status": "inactive",
                "happen_time": str(happen_time or ""),
                "observed_time": str(observed_time or ""),
                "last_action": "FORGET",
                "rationale": decision.rationale,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            append_entries.append(self._entry_for_fact(fid))
            return False
        fid = self._register_fact(triple, happen_time, observed_time, decision.decision, decision.rationale)
        append_entries.append(self._entry_for_fact(fid))
        return True

    def _deactivate_targets(
        self,
        decision: MemoryDecision,
        candidates: List[FactRecord],
        append_entries: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, str]]:
        if decision.decision not in ("UPDATE", "FORGET"):
            return []
        deactivated: List[Tuple[str, str, str]] = []
        now = datetime.now(tz=timezone.utc).isoformat()
        candidate_map = {rec.fact_id: rec for rec in candidates}
        for fid in decision.target_ids:
            if fid in ("NEW", "NONE"):
                continue
            rec = candidate_map.get(fid)
            if not rec:
                continue
            entry = self.registry["facts"].get(fid, {})
            entry.update(
                {
                    "status": "inactive",
                    "last_action": decision.decision,
                    "rationale": decision.rationale,
                    "updated_at": now,
                }
            )
            self.registry["facts"][fid] = entry
            append_entries.append(self._entry_for_fact(fid))
            deactivated.append(rec.triple)
        return deactivated
