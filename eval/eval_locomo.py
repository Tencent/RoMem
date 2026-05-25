"""
CLI entry point for running the LoCoMo benchmark with memory frameworks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

from benchmarks.runners.locomo.runner import run_locomo
from neo4j import GraphDatabase, exceptions as neo4j_exceptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate memory frameworks on the LoCoMo dataset.')
    parser.add_argument('--config', help='Path to JSON config file.')
    parser.add_argument('--data-path', default=os.getenv('LOCOMO_DATA_PATH'))
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--neo4j-uri', default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-user', default=os.getenv('NEO4J_USER', 'neo4j'))
    parser.add_argument('--neo4j-password', default=os.getenv('NEO4J_PASSWORD', 'password'))
    parser.add_argument('--neo4j-database', default=os.getenv('NEO4J_DATABASE'))
    parser.add_argument('--recreate-database', action='store_true')
    parser.add_argument('--exp-name', help='Experiment name for logging results.')
    parser.add_argument('--reset-mode', choices=['delete', 'recreate'], default='delete', help='How to reset the graph between questions (delete nodes vs recreate driver).')
    parser.add_argument('--answer-llm-model', help='Optional override for the LLM used to answer questions (different from ingestion).')
    parser.add_argument('--answer-llm-api-key', help='API key for the answer LLM if different from default.')
    parser.add_argument('--answer-llm-base-url', help='Base URL for the answer LLM if different from default.')
    parser.add_argument(
        '--question-type',
        default=os.getenv('LOCOMO_QUESTION_TYPE'),
        help='Optional filter to evaluate only a specific question_type (e.g., temporal).',
    )
    parser.add_argument('--llm-debug', action='store_true')
    parser.add_argument('--ingest-debug', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument(
        '--search-reranker',
        default=os.getenv('BENCHMARK_SEARCH_RERANKER'),
        help='Search reranker recipe (basic, cross_encoder, rrf, mmr, node_distance, episode_mentions).',
    )
    parser.add_argument(
        '--memory-framework',
        default=os.getenv('MEMORY_FRAMEWORK', 'graphiti'),
        help='Memory framework to use (graphiti, hipporag, mem0, romem, amem, licomem).',
    )
    parser.add_argument('--hippo-save-dir', default=os.getenv('HIPPO_SAVE_DIR'))
    parser.add_argument('--hippo-llm-model', default=os.getenv('HIPPO_LLM_MODEL'))
    parser.add_argument('--hippo-embedding-model', default=os.getenv('HIPPO_EMBED_MODEL'))
    parser.add_argument('--hippo-openie-mode', default=os.getenv('HIPPO_OPENIE_MODE'))
    parser.add_argument('--romem-save-dir', default=os.getenv('ROMEM_SAVE_DIR'))
    parser.add_argument('--romem-llm-model', default=os.getenv('ROMEM_LLM_MODEL'))
    parser.add_argument('--romem-embedding-model', default=os.getenv('ROMEM_EMBED_MODEL'))
    parser.add_argument('--romem-openie-mode', default=os.getenv('ROMEM_OPENIE_MODE'))
    parser.add_argument('--romem-temporal-awareness', default=os.getenv('ROMEM_TEMPORAL_AWARENESS'))
    parser.add_argument('--enable-tkge-tunnel', default=os.getenv('ROMEM_ENABLE_TKGE_TUNNEL'))
    parser.add_argument('--romem-tkge-verbose', type=int, default=os.getenv('ROMEM_TKGE_VERBOSE'))
    args = parser.parse_args()
    defaults = {action.dest: action.default for action in parser._actions}
    setattr(args, "_defaults", defaults)
    return args


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return payload


def _apply_config(args: argparse.Namespace, config: dict, defaults: dict) -> None:
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if key in defaults and current == defaults[key]:
            setattr(args, key, value)


def ensure_database(uri: str, user: str, password: str, database: str | None, recreate: bool) -> None:
    if not database:
        return
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database='system') as session:
            exists = session.run(
                'SHOW DATABASES YIELD name WHERE name = $name RETURN name',
                name=database,
            ).single()
            if exists:
                if recreate:
                    logging.info('Dropping existing Neo4j database %s', database)
                    session.run(f'DROP DATABASE `{database}` IF EXISTS WAIT')
                    session.run(f'CREATE DATABASE `{database}`')
                else:
                    logging.info('Neo4j database %s already exists. Reuse with --recreate-database to rebuild.', database)
            else:
                logging.info('Creating Neo4j database %s', database)
                session.run(f'CREATE DATABASE `{database}`')
    except neo4j_exceptions.ClientError as exc:
        if exc.code == 'Neo.ClientError.Statement.UnsupportedAdministrationCommand':
            logging.warning(
                'Neo4j server does not support database create/drop commands. Clearing graph instead.'
            )
            _clear_default_database(driver)
            return
        raise
    finally:
        driver.close()


def _clear_default_database(driver) -> None:
    with driver.session(database='neo4j') as session:
        total_deleted = 0
        while True:
            result = session.run(
                """
                CALL {
                    MATCH (n)
                    WITH n LIMIT $batch_size
                    DETACH DELETE n
                    RETURN COUNT(*) AS deleted
                }
                RETURN deleted
                """,
                batch_size=1000,
            ).single()
            deleted = result['deleted'] if result else 0
            total_deleted += deleted
            if deleted == 0:
                break
        logging.info('Cleared %s existing nodes from default Neo4j database.', total_deleted)


def main():
    args = parse_args()
    config = _load_config(args.config)
    _apply_config(args, config, getattr(args, "_defaults", {}))
    romem_config = config if args.memory_framework.lower() == 'romem' else None

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if not args.llm_debug:
        logging.getLogger('httpx').setLevel(logging.WARNING)
    if not args.ingest_debug:
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        logging.getLogger('neo4j.notifications').setLevel(logging.WARNING)

    if args.memory_framework.lower() == 'graphiti':
        ensure_database(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database, args.recreate_database)

    asyncio.run(
        run_locomo(
            data_path=args.data_path,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            max_examples=args.max_examples,
            exp_name=args.exp_name,
            reset_mode=args.reset_mode,
            llm_model=args.answer_llm_model,
            llm_api_key=args.answer_llm_api_key,
            llm_base_url=args.answer_llm_base_url,
            search_reranker=args.search_reranker,
            question_type=args.question_type,
            memory_framework=args.memory_framework,
            romem_config=romem_config,
            hippo_save_dir=args.hippo_save_dir,
            hippo_llm_model=args.hippo_llm_model,
            hippo_embedding_model=args.hippo_embedding_model,
            hippo_openie_mode=args.hippo_openie_mode,
            romem_save_dir=args.romem_save_dir,
            romem_llm_model=args.romem_llm_model,
            romem_embedding_model=args.romem_embedding_model,
            romem_openie_mode=args.romem_openie_mode,
            romem_temporal_awareness=args.romem_temporal_awareness,
            romem_enable_tkge_tunnel=args.enable_tkge_tunnel,
            romem_tkge_verbose=args.romem_tkge_verbose,
        )
    )


if __name__ == '__main__':
    main()
