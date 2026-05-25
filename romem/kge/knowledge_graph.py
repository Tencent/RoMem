"""
Knowledge graph snapshot handling adapted from TKGE.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from .utils import *


def load_fact(path):
    """Load facts from a whitespace-delimited text file."""
    facts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            h, r, t = line[0], line[1], line[2]
            facts.append((h, r, t))
    return facts


def build_edge_index(h, t):
    """Build edge_index using head and tail lists."""
    index = [h + t, t + h]
    return torch.LongTensor(index)


class KnowledgeGraph:
    """
    Minimal port of TKGE's KnowledgeGraph helper.

    This version can be built from in-memory triples for integration with GraphBuilder.
    """

    def __init__(self, config) -> None:
        self.args = config
        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = {}, {}, {}, {}
        self.relationid2invid = {}
        self.snapshots = {0: Snapshot(self.args)}
        self.facts = []
        self.fact_times = []
        self.fact_has_happen = []
        self._fact_set = set()

    @classmethod
    def from_triples(cls, triples, config):
        kg = cls(config)
        kg.add_triples(triples)
        return kg

    def add_triples(self, triples):
        hr2t_all = {}
        self.new_entities = set()
        new_triples = []
        for (h, r, t) in triples:
            key = (h, r, t)
            if key in self._fact_set:
                continue
            self._fact_set.add(key)
            new_triples.append(key)
            self.expend_entity_relation([(h, r, t)])
        if not new_triples:
            return False
        fact_ids = self.fact2id(new_triples)
        edge_h, edge_r, edge_t = [], [], []
        edge_h, edge_r, edge_t = self.expand_kg(fact_ids, 'train', edge_h, edge_r, edge_t, hr2t_all)
        self.store_snapshot(0, fact_ids, fact_ids, [], [], [], [], edge_h, edge_r, edge_t, hr2t_all)
        self.facts.extend(new_triples)
        # Default time is 0.0 (unknown) for non-temporal updates.
        self.fact_times.extend([0.0] * len(new_triples))
        self.fact_has_happen.extend([False] * len(new_triples))
        self.new_entities.clear()
        return True

    def add_timed_triples(self, timed_triples):
        """
        Add triples with an associated scalar time (unix seconds).

        timed_triples: List[(h, r, t, time_scalar, has_happen)]
        """
        hr2t_all = {}
        self.new_entities = set()
        new_triples = []
        new_times = []
        new_has_happen = []
        for (h, r, t, ts, has_happen) in timed_triples:
            key = (h, r, t)
            if key in self._fact_set:
                continue
            self._fact_set.add(key)
            new_triples.append(key)
            new_times.append(float(ts))
            new_has_happen.append(bool(has_happen))
            self.expend_entity_relation([(h, r, t)])
        if not new_triples:
            return False
        fact_ids = self.fact2id(new_triples)
        edge_h, edge_r, edge_t = [], [], []
        edge_h, edge_r, edge_t = self.expand_kg(fact_ids, 'train', edge_h, edge_r, edge_t, hr2t_all)
        self.store_snapshot(0, fact_ids, fact_ids, [], [], [], [], edge_h, edge_r, edge_t, hr2t_all)
        self.facts.extend(new_triples)
        self.fact_times.extend(new_times)
        self.fact_has_happen.extend(new_has_happen)
        self.new_entities.clear()
        return True

    def remove_triples(self, triples):
        # Simplistic removal: rebuild from remaining facts
        to_remove = set(triples)
        remaining = [f for f in self.facts if f not in to_remove]
        remaining_times = [ts for f, ts in zip(self.facts, self.fact_times) if f not in to_remove]
        remaining_has_happen = [hh for f, hh in zip(self.facts, self.fact_has_happen) if f not in to_remove]
        self.__init__(self.args)
        for f in remaining:
            self._fact_set.add(f)
        # Re-add with stored times (if any); default to 0.0 when absent.
        if remaining:
            self.add_timed_triples(
                [
                    (h, r, t, ts, hh)
                    for (h, r, t), ts, hh in zip(remaining, remaining_times, remaining_has_happen)
                ]
            )

    def get_id_triples(self):
        return self.fact2id(self.facts)

    def get_id_triples_with_times(self):
        """
        Return (h_id, r_id, t_id, time_scalar) aligned with self.facts.
        """
        ids = self.fact2id(self.facts)
        times = list(self.fact_times)
        has_happen = list(self.fact_has_happen)
        if len(times) != len(ids):
            times = [0.0] * len(ids)
        if len(has_happen) != len(ids):
            has_happen = [False] * len(ids)
        return [(h, r, t, float(ts), bool(hh)) for (h, r, t), ts, hh in zip(ids, times, has_happen)]

    @property
    def num_entities(self):
        return self.num_ent

    @property
    def num_relations(self):
        return self.num_rel

    def store_snapshot(self, ss_id, train, train_all, valid, valid_all, test, test_all, edge_h, edge_r, edge_t, hr2t_all):
        """Store snapshot-specific metadata."""
        self.snapshots[ss_id].num_ent = deepcopy(self.num_ent)
        self.snapshots[ss_id].num_rel = deepcopy(self.num_rel)

        self.snapshots[ss_id].train = deepcopy(train)
        self.snapshots[ss_id].train_all = deepcopy(train_all)
        self.snapshots[ss_id].valid = deepcopy(valid)
        self.snapshots[ss_id].valid_all = deepcopy(valid_all)
        self.snapshots[ss_id].test = deepcopy(test)
        self.snapshots[ss_id].test_all = deepcopy(test_all)

        self.snapshots[ss_id].edge_h = deepcopy(edge_h)
        self.snapshots[ss_id].edge_r = deepcopy(edge_r)
        self.snapshots[ss_id].edge_t = deepcopy(edge_t)

        self.snapshots[ss_id].hr2t_all = deepcopy(hr2t_all)
        self.snapshots[ss_id].edge_index = build_edge_index(edge_h, edge_t).to(self.args.device)
        self.snapshots[ss_id].edge_type = torch.cat([torch.LongTensor(edge_r), torch.LongTensor(edge_r) + 1]).to(
            self.args.device
        )
        self.snapshots[ss_id].new_entities = deepcopy(list(self.new_entities))

    def expand_kg(self, facts, split, edge_h, edge_r, edge_t, hr2t_all):
        """Update helper structures with new facts."""

        def add_key2val(mapping, key, val):
            if key not in mapping:
                mapping[key] = set()
            mapping[key].add(val)

        for (h, r, t) in facts:
            self.new_entities.add(h)
            self.new_entities.add(t)
            if split == 'train':
                edge_h.append(h)
                edge_r.append(r)
                edge_t.append(t)
            add_key2val(hr2t_all, (h, r), t)
            add_key2val(hr2t_all, (t, self.relationid2invid[r]), h)
        return edge_h, edge_r, edge_t

    def fact2id(self, facts, order=False):
        """Convert triple strings into identifier triples."""
        fact_id = []
        if order:
            i = 0
            while len(fact_id) < len(facts):
                for (h, r, t) in facts:
                    if self.relation2id[r] == i:
                        fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
                i += 2
        else:
            for (h, r, t) in facts:
                fact_id.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
        return fact_id

    def expend_entity_relation(self, facts):
        """Register new entities and relations."""
        for (h, r, t) in facts:
            if h not in self.entity2id:
                self.entity2id[h] = self.num_ent
                self.id2entity[self.num_ent] = h
                self.num_ent += 1
            if t not in self.entity2id:
                self.entity2id[t] = self.num_ent
                self.id2entity[self.num_ent] = t
                self.num_ent += 1

            if r not in self.relation2id:
                self.relation2id[r] = self.num_rel
                self.id2relation[self.num_rel] = r
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.id2relation[self.num_rel + 1] = r + '_inv'
                self.relationid2invid[self.num_rel] = self.num_rel + 1
                self.relationid2invid[self.num_rel + 1] = self.num_rel
                self.num_rel += 2


class Snapshot:
    """Container for per-snapshot artifacts."""

    def __init__(self, args) -> None:
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.train_all, self.valid, self.valid_all, self.test, self.test_all = [], [], [], [], [], []
        self.edge_h, self.edge_r, self.edge_t = [], [], []
        self.hr2t_all = {}
        self.edge_index, self.edge_type = None, None
        self.new_entities = []
