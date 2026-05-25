"""Utilities for LoCoMo runners."""

from __future__ import annotations

from benchmarks.types import EpisodePayload, LocomoSample


def _evidence_session_ids(sample: LocomoSample) -> set[str]:
    session_ids: set[str] = set()
    for qa in sample.qa:
        for ev in qa.evidence:
            if not ev:
                continue
            prefix = ev.split(":", 1)[0].strip()
            if not prefix:
                continue
            token = prefix
            if token.lower().startswith("session_"):
                token = token.split("_", 1)[1]
            if token and token[0].isalpha():
                token = token[1:]
            token = token.strip()
            if token.isdigit():
                session_ids.add(f"S{int(token)}")
            elif prefix.startswith("S"):
                session_ids.add(prefix)
    return session_ids


def filter_sample_episodes(sample: LocomoSample) -> LocomoSample:
    session_ids = _evidence_session_ids(sample)
    if not session_ids:
        return sample
    episodes: list[EpisodePayload] = [
        episode
        for episode in sample.episodes
        if str(episode.metadata.get("session_id")) in session_ids
    ]
    if not episodes:
        return sample
    return LocomoSample(
        sample_id=sample.sample_id,
        conversation=sample.conversation,
        qa=sample.qa,
        episodes=episodes,
    )
