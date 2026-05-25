"""Dispatcher runner for the LongMemEval benchmark."""

from __future__ import annotations

from typing import Optional


def _filter(kwargs: dict, prefix: str) -> dict:
    """Keep only kwargs whose key starts with the given prefix."""
    return {k: v for k, v in kwargs.items() if k.startswith(prefix + '_')}


def get_longmemeval_runner(
    memory_framework: str = 'graphiti',
    *,
    data_path: str | None = None,
    max_examples: Optional[int] = None,
    exp_name: str | None = None,
    reset_mode: str = 'delete',
    search_limit: int | None = None,
    answer_context_sizes: list[int] | None = None,
    answer_llm_model: str | None = None,
    answer_llm_api_key: str | None = None,
    answer_llm_base_url: str | None = None,
    # Judge LLM (LongMemEval-specific)
    judge_llm_model: str | None = None,
    judge_llm_api_key: str | None = None,
    judge_llm_base_url: str | None = None,
    disable_judge: bool = False,
    enable_official_comparison: bool = False,
    official_baseline_path: str | None = None,
    # Neo4j (Graphiti / Mem0)
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    search_reranker: str | None = None,
    # Framework-specific
    **framework_kwargs,
):
    framework = (memory_framework or 'graphiti').strip().lower()
    _common = dict(
        data_path=data_path, max_examples=max_examples, exp_name=exp_name,
        reset_mode=reset_mode, search_limit=search_limit,
        answer_context_sizes=answer_context_sizes,
        answer_llm_model=answer_llm_model, answer_llm_api_key=answer_llm_api_key,
        answer_llm_base_url=answer_llm_base_url,
        judge_llm_model=judge_llm_model, judge_llm_api_key=judge_llm_api_key,
        judge_llm_base_url=judge_llm_base_url,
        disable_judge=disable_judge,
    )

    if framework == 'graphiti':
        from benchmarks.runners.longmemeval.graphiti_runner import LongmemevalGraphitiRunner
        return LongmemevalGraphitiRunner(
            **_common,
            uri=uri, user=user, password=password, database=database,
            search_reranker=search_reranker,
            enable_official_comparison=enable_official_comparison,
            official_baseline_path=official_baseline_path,
        )
    if framework == 'hipporag':
        from benchmarks.runners.longmemeval.hipporag_runner import LongmemevalHippoRAGRunner
        return LongmemevalHippoRAGRunner(**_common, **_filter(framework_kwargs, 'hippo'))
    if framework == 'mem0':
        from benchmarks.runners.longmemeval.mem0_runner import LongmemevalMem0Runner
        return LongmemevalMem0Runner(
            **_common,
            neo4j_uri=uri, neo4j_user=user, neo4j_password=password, neo4j_database=database,
        )
    if framework == 'romem':
        from benchmarks.runners.longmemeval.romem_runner import LongmemevalRoMemRunner
        return LongmemevalRoMemRunner(**_common, **_filter(framework_kwargs, 'romem'))
    raise ValueError(f'Unknown memory framework: {framework}')


async def run_longmemeval(memory_framework: str = 'graphiti', **kwargs) -> None:
    runner = get_longmemeval_runner(memory_framework, **kwargs)
    await runner.run()


async def run_graphiti_longmemeval(**kwargs) -> None:
    """Legacy entry point for Graphiti-only LongMemEval runs."""
    runner = get_longmemeval_runner('graphiti', **kwargs)
    await runner.run()
