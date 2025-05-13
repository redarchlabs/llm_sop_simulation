# drivers.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

# Qdrant setup
qdrant_driver = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333"))
)

# Neo4j setup
graph_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "neo4j")
    )
)
