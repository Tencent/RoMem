"""Dispatcher for MultiTQ runners."""

from __future__ import annotations


def _filter(kwargs: dict, prefix: str) -> dict:
    """Keep only kwargs whose key starts with the given prefix."""
    return {k: v for k, v in kwargs.items() if k.startswith(prefix + '_')}


def _shared(kwargs: dict) -> dict:
    """Extract shared (non-prefixed) kwargs."""
    prefixes = ('hippo_', 'romem_', 'licomem_', 'amem_', 'zep_')
    return {k: v for k, v in kwargs.items() if not any(k.startswith(p) for p in prefixes)}


def get_multitq_runner(
    memory_framework: str = "romem",
    *,
    data_path: str | None = None,
    eval_split: str = "test",
    max_time_ids: int | None = None,
    max_examples: int | None = None,
    exp_name: str | None = None,
    search_limit: int | None = None,
    answer_context_sizes: list[int] | None = None,
    answer_llm_model: str | None = None,
    answer_llm_api_key: str | None = None,
    answer_llm_base_url: str | None = None,
    # Neo4j (Graphiti / Mem0)
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    search_reranker: str | None = None,
    group_id: str | None = None,
    # Framework-specific
    **framework_kwargs,
):
    framework = (memory_framework or "graphiti").strip().lower()
    _common = dict(
        data_path=data_path, eval_split=eval_split,
        max_time_ids=max_time_ids, max_examples=max_examples, exp_name=exp_name,
        search_limit=search_limit, answer_context_sizes=answer_context_sizes,
        answer_llm_model=answer_llm_model, answer_llm_api_key=answer_llm_api_key,
        answer_llm_base_url=answer_llm_base_url,
    )

    if framework == "romem":
        from benchmarks.runners.multitq.romem_runner import MultiTQRoMemRunner
        return MultiTQRoMemRunner(**_common, **_filter(framework_kwargs, 'romem'))
    if framework == "hipporag":
        from benchmarks.runners.multitq.hipporag_runner import MultiTQHippoRAGRunner
        return MultiTQHippoRAGRunner(**_common, **_filter(framework_kwargs, 'hippo'))
    if framework == "mem0":
        from benchmarks.runners.multitq.mem0_runner import MultiTQMem0Runner
        return MultiTQMem0Runner(
            **_common,
            neo4j_uri=uri, neo4j_user=user, neo4j_password=password, neo4j_database=database,
        )
    if framework == "graphiti":
        from benchmarks.runners.multitq.graphiti_runner import MultiTQGraphitiRunner
        if uri is None or user is None or password is None:
            raise ValueError("Graphiti runner requires Neo4j uri/user/password")
        return MultiTQGraphitiRunner(
            **_common,
            uri=uri, user=user, password=password, database=database,
            search_reranker=search_reranker, group_id=group_id,
        )
    if framework == "licomem":
        from benchmarks.runners.multitq.licomem_runner import MultiTQLiCoMemRunner
        return MultiTQLiCoMemRunner(**_common, **_filter(framework_kwargs, 'licomem'))
    if framework == "amem":
        from benchmarks.runners.multitq.amem_runner import MultiTQAMemRunner
        return MultiTQAMemRunner(**_common, **_filter(framework_kwargs, 'amem'))
    raise ValueError(f"Unknown memory framework: {memory_framework}")


async def run_multitq(memory_framework: str = "romem", **kwargs) -> None:
    runner = get_multitq_runner(memory_framework, **kwargs)
    await runner.run()
