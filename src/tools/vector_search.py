from crewai.tools import BaseTool
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer
from typing import List
import os

model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")

client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "test_collection"

class VectorSearchTool(BaseTool):
    name: str = "VectorSearchTool"
    description: str = "Retrieves semantically relevant text chunks from Qdrant"

    def _run(self, query: str) -> List[str]:
        query_vector = model.encode(f"passage: {query}").tolist()
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5,
            search_params=SearchParams(hnsw_ef=512)
        )
        return [hit.payload.get("text", "") for hit in results]

vector_search_tool = VectorSearchTool()
