from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, CollectionStatus
from pathlib import Path
from chunk_utils import chunk_and_embed
import uuid

# Setup
collection_name = "test_collection"
vector_size = 256

# Make sure this is the correct host and port for your local instance of Qdrant
client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

file_name = input("Enter the file name: ")
# Chunk + embed
file_path = Path("data/"+ file_name)
chunks, embeddings = chunk_and_embed(file_path)

# Upload to Qdrant
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),  # Convert numpy array to list
        payload={"text": chunk, "source": file_path.name}
    )
    for chunk, embedding in zip(chunks, embeddings)
]


client.upsert(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} vectors to Qdrant.")

