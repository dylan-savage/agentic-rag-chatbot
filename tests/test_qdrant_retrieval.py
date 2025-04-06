from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

# Setup
collection_name = "test_collection"
query = "What is the NFL's new kickoff rule?"
model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")
client = QdrantClient(host="localhost", port=6333)

# Embed query
query_vector = model.encode(query).tolist()

# Search
results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
    search_params=SearchParams(hnsw_ef=512)
)

# Output
print(f"Top {len(results)} results:\n")
for i, hit in enumerate(results, 1):
    print(f"Result {i}")
    print(f"Score: {hit.score}")
    print(f"Payload: {hit.payload['text'][:300]}...\n")
