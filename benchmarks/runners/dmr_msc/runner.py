"""Dispatcher for DMR-MSC runners."""

from __future__ import annotations

from typing import Optional


def _filter(kwargs: dict, prefix: str) -> dict:
    """Keep only kwargs whose key starts with the given prefix."""
    return {k: v for k, v in kwargs.items() if k.startswith(prefix + '_')}


def _shared(kwargs: dict) -> dict:
    """Extract shared (non-prefixed) kwargs like search_limit, answer_*."""
    prefixes = ('hippo_', 'romem_', 'licomem_', 'amem_', 'zep_')
    return {k: v for k, v in kwargs.items() if not any(k.startswith(p) for p in prefixes)}


def get_dmr_msc_runner(
    data_path: str | None = None,
    max_examples: Optional[int] = None,
    exp_name: str | None = None,
    memory_framework: str = 'graphiti',
    **framework_kwargs,
):
    framework = (memory_framework or 'graphiti').strip().lower()
    _common = dict(data_path=data_path, max_examples=max_examples, exp_name=exp_name)

    if framework == 'hipporag':
        from benchmarks.runners.dmr_msc.hipporag_runner import DmrMscHippoRAGRunner
        return DmrMscHippoRAGRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'hippo'))
    if framework == 'graphiti':
        from benchmarks.runners.dmr_msc.graphiti_runner import DmrMscGraphitiRunner
        shared = _shared(framework_kwargs)
        if shared.get('uri') is None or shared.get('user') is None or shared.get('password') is None:
            raise ValueError('Graphiti runner requires Neo4j uri/user/password')
        return DmrMscGraphitiRunner(**_common, **shared)
    if framework == 'mem0':
        from benchmarks.runners.dmr_msc.mem0_runner import DmrMscMem0Runner
        return DmrMscMem0Runner(**_common, **_shared(framework_kwargs))
    if framework == 'romem':
        from benchmarks.runners.dmr_msc.romem_runner import DmrMscRoMemRunner
        return DmrMscRoMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'romem'))
    if framework == 'zep':
        from benchmarks.runners.dmr_msc.zep_runner import DmrMscZepRunner
        return DmrMscZepRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'zep'))
    if framework == 'licomem':
        from benchmarks.runners.dmr_msc.licomem_runner import DmrMscLiCoMemRunner
        return DmrMscLiCoMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'licomem'))
    if framework == 'amem':
        from benchmarks.runners.dmr_msc.amem_runner import DmrMscAMemRunner
        return DmrMscAMemRunner(**_common, **_shared(framework_kwargs), **_filter(framework_kwargs, 'amem'))
    raise ValueError(f'Unsupported memory framework: {memory_framework}')


async def run_dmr_msc(memory_framework: str = 'graphiti', **kwargs) -> None:
    runner = get_dmr_msc_runner(memory_framework=memory_framework, **kwargs)
    await runner.run()
