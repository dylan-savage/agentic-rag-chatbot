from chonkie.chunker.semantic import SemanticChunker
from sentence_transformers import SentenceTransformer
import fitz
from pathlib import Path
from bs4 import BeautifulSoup

def load_text_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in [".txt", ".md"]:
        return Path(file_path).read_text(encoding="utf-8")
    elif ext == ".pdf":
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    elif ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_and_embed(file_path: str):
    print(f"Processing {file_path}")
    text = load_text_from_file(file_path)

    # 1. Chunk
    chunker = SemanticChunker(return_type="texts")
    chunks = chunker.chunk(text)

    # 2. Embed

    # On Apple Silicon, MPS doesn't support embedding_bag. Run on CPU to avoid runtime error.
    # Can run on GPU if you have a supported GPU
    model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")
    inputs = [f"passage: {chunk}" for chunk in chunks]
    embeddings = model.encode(inputs, convert_to_numpy=True)

    print(f"Chunked into {len(chunks)} chunks.\n")
    return chunks, embeddings
