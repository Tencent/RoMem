from __future__ import annotations

import os
import igraph as ig


class GraphManager:
    """Handles loading/initializing and saving the igraph backbone."""

    def __init__(self, pickle_path: str, directed: bool = False):
        self.pickle_path = pickle_path
        self.directed = directed
        self.graph: ig.Graph | None = None

    def load_or_init(self, force_from_scratch: bool) -> ig.Graph:
        if (not force_from_scratch) and os.path.exists(self.pickle_path):
            self.graph = ig.Graph.Read_Pickle(self.pickle_path)
        else:
            self.graph = ig.Graph(directed=self.directed)
        return self.graph

    def save(self) -> None:
        if self.graph is None:
            return
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        self.graph.write_pickle(self.pickle_path)

    def export_graphml(self, path: str) -> str:
        """
        Export the current igraph graph to GraphML for visualization (e.g., in Gephi).
        """
        if self.graph is None:
            raise ValueError("Graph is not initialized.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.graph.write_graphml(path)
        return path
