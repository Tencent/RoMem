"""Dispatcher for FinTMMBench runners."""

from __future__ import annotations

from typing import Optional


def _filter(kwargs: dict, prefix: str) -> dict:
    """Keep only kwargs whose key starts with the given prefix."""
    return {k: v for k, v in kwargs.items() if k.startswith(prefix + '_')}


def _shared(kwargs: dict) -> dict:
    """Extract shared (non-prefixed) kwargs."""
    prefixes = ('hippo_', 'romem_', 'licomem_', 'amem_', 'zep_')
    return {k: v for k, v in kwargs.items() if not any(k.startswith(p) for p in prefixes)}


def get_fintmmbench_runner(
    data_path: str | None = None,
    max_examples: Optional[int] = None,
    exp_name: str | None = None,
    memory_framework: str = 'graphiti',
    **framework_kwargs,
):
    framework = (memory_framework or 'graphiti').strip().lower()
    _common = dict(data_path=data_path, max_examples=max_examples, exp_name=exp_name)

    if framework == 'hipporag':
        from benchmarks.runners.fintmmbench.hipporag_runner import FinTMMBenchHippoRAGRunner
        return FinTMMBenchHippoRAGRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'hippo'))
    if framework == 'graphiti':
        from benchmarks.runners.fintmmbench.graphiti_runner import FinTMMBenchGraphitiRunner
        shared = _shared(framework_kwargs)
        if shared.get('uri') is None or shared.get('user') is None or shared.get('password') is None:
            raise ValueError('Graphiti runner requires Neo4j uri/user/password')
        return FinTMMBenchGraphitiRunner(**_common, **shared)
    if framework == 'mem0':
        from benchmarks.runners.fintmmbench.mem0_runner import FinTMMBenchMem0Runner
        return FinTMMBenchMem0Runner(**_common, **_shared(framework_kwargs))
    if framework == 'romem':
        from benchmarks.runners.fintmmbench.romem_runner import FinTMMBenchRoMemRunner
        return FinTMMBenchRoMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'romem'))
    if framework == 'licomem':
        from benchmarks.runners.fintmmbench.licomem_runner import FinTMMBenchLiCoMemRunner
        return FinTMMBenchLiCoMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'licomem'))
    if framework == 'amem':
        from benchmarks.runners.fintmmbench.amem_runner import FinTMMBenchAMemRunner
        return FinTMMBenchAMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'amem'))
    raise ValueError(f'Unsupported memory framework: {memory_framework}')


async def run_fintmmbench(memory_framework: str = 'graphiti', **kwargs) -> None:
    runner = get_fintmmbench_runner(memory_framework=memory_framework, **kwargs)
    await runner.run()
