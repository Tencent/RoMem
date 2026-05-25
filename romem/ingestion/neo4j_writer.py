from __future__ import annotations

from typing import Iterable

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dep
    GraphDatabase = None  # type: ignore

from romem.utils.misc_utils import TimedTriple


class Neo4jWriter:
    """
    Optional Neo4j writer for inspection.

    Usage:
        writer = Neo4jWriter(uri, user, password, database="neo4j")
        writer.push(entities, timed_triples)
        writer.close()
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        if GraphDatabase is None:
            raise ImportError("neo4j driver is not installed. Please `pip install neo4j` to use Neo4jWriter.")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def push(self, entities: Iterable[str], timed_triples: Iterable[TimedTriple]) -> None:
        entities = list(entities)
        timed_triples = list(timed_triples)

        def _tx_fn(tx):
            if entities:
                tx.run(
                    """
                    UNWIND $entities AS name
                    MERGE (e:Entity {name: name})
                    """,
                    entities=entities,
                )
            if timed_triples:
                tx.run(
                    """
                    UNWIND $triples AS row
                    MERGE (h:Entity {name: row.head})
                    MERGE (t:Entity {name: row.tail})
                    MERGE (h)-[r:RELATES_TO {relation: row.relation}]->(t)
                    SET r.relation = row.relation,
                        r.happen_time = row.happen_time,
                        r.observed_time = row.observed_time
                    """,
                    triples=[
                        {
                            "head": tt.triple[0],
                            "relation": tt.triple[1],
                            "tail": tt.triple[2],
                            "happen_time": getattr(tt, "happen_time", "") or "",
                            "observed_time": getattr(tt, "system_time", "") or "",
                        }
                        for tt in timed_triples
                    ],
                )

        with self._driver.session(database=self._database) as session:
            session.execute_write(_tx_fn)
