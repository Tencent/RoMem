"""Dispatcher for LoCoMo runners."""

from __future__ import annotations

from typing import Optional


def _filter(kwargs: dict, prefix: str) -> dict:
    """Keep only kwargs whose key starts with the given prefix."""
    return {k: v for k, v in kwargs.items() if k.startswith(prefix + '_')}


def get_locomo_runner(
    data_path: str | None = None,
    max_examples: Optional[int] = None,
    exp_name: str | None = None,
    memory_framework: str = 'graphiti',
    question_type: str | None = None,
    # Shared LLM / search params (used by most frameworks)
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    hippo_search_limit: int | None = None,
    hippo_llm_context_sizes: list[int] | None = None,
    # Neo4j (Graphiti / Mem0)
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    reset_mode: str = 'delete',
    search_reranker: str | None = None,
    # Framework-specific (prefixed: romem_*, hippo_*, licomem_*, amem_*)
    **framework_kwargs,
):
    framework = (memory_framework or 'graphiti').strip().lower()
    _common = dict(
        data_path=data_path, max_examples=max_examples, exp_name=exp_name,
        question_type=question_type,
    )

    if framework == 'hipporag':
        from benchmarks.runners.locomo.hipporag_runner import LocomoHippoRAGRunner
        return LocomoHippoRAGRunner(
            **_common,
            search_limit=hippo_search_limit,
            llm_context_sizes=hippo_llm_context_sizes,
            llm_model=llm_model, llm_api_key=llm_api_key, llm_base_url=llm_base_url,
            **_filter(framework_kwargs, 'hippo'),
        )
    if framework == 'graphiti':
        from benchmarks.runners.locomo.graphiti_runner import LocomoGraphitiRunner
        if uri is None or user is None or password is None:
            raise ValueError('Graphiti runner requires Neo4j uri/user/password')
        return LocomoGraphitiRunner(
            **_common,
            uri=uri, user=user, password=password, database=database,
            reset_mode=reset_mode, search_reranker=search_reranker,
            llm_model=llm_model, llm_api_key=llm_api_key, llm_base_url=llm_base_url,
        )
    if framework == 'mem0':
        from benchmarks.runners.locomo.mem0_runner import LocomoMem0Runner
        return LocomoMem0Runner(
            **_common,
            neo4j_uri=uri, neo4j_user=user, neo4j_password=password, neo4j_database=database,
            search_limit=hippo_search_limit,
            llm_context_sizes=hippo_llm_context_sizes,
            answer_llm_model=llm_model, answer_llm_api_key=llm_api_key, answer_llm_base_url=llm_base_url,
        )
    if framework == 'romem':
        from benchmarks.runners.locomo.romem_runner import LocomoRoMemRunner
        return LocomoRoMemRunner(
            **_common,
            search_limit=hippo_search_limit,
            llm_context_sizes=hippo_llm_context_sizes,
            llm_model=llm_model, llm_api_key=llm_api_key, llm_base_url=llm_base_url,
            **_filter(framework_kwargs, 'romem'),
        )
    if framework == 'licomem':
        from benchmarks.runners.locomo.licomem_runner import LocomoLiCoMemRunner
        return LocomoLiCoMemRunner(
            **_common,
            search_limit=hippo_search_limit,
            llm_context_sizes=hippo_llm_context_sizes,
            answer_llm_model=llm_model, answer_llm_api_key=llm_api_key, answer_llm_base_url=llm_base_url,
            **_filter(framework_kwargs, 'licomem'),
        )
    if framework == 'amem':
        from benchmarks.runners.locomo.amem_runner import LocomoAMemRunner
        return LocomoAMemRunner(
            **_common,
            search_limit=hippo_search_limit,
            llm_context_sizes=hippo_llm_context_sizes,
            answer_llm_model=llm_model, answer_llm_api_key=llm_api_key, answer_llm_base_url=llm_base_url,
            **_filter(framework_kwargs, 'amem'),
        )
    raise ValueError(f'Unsupported memory framework: {memory_framework}')


async def run_locomo(memory_framework: str = 'graphiti', **kwargs) -> None:
    runner = get_locomo_runner(memory_framework=memory_framework, **kwargs)
    await runner.run()
