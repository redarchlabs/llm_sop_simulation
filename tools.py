# tools.py

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import Driver

class ResearchTools:
    def __init__(self, qdrant_driver: QdrantClient, graph_driver: Driver):
        self.qdrant = qdrant_driver
        self.graph = graph_driver

    def search_vector_db(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        from llm import get_embedding

        query_vector = get_embedding(query)[0]
        search_result = self.qdrant.search(
            collection_name="JBAF_LAW_doc_chunks",
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [point.payload for point in search_result]

    def search_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        with self.graph.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
