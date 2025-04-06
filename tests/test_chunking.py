from chonkie.chunker.semantic import SemanticChunker
import fitz  
from pathlib import Path
from bs4 import BeautifulSoup

def load_text_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    
    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif ext == ".pdf":
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])

    elif ext == ".html":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n")
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_file(file_path: str):
    print(f"Processing: {file_path}")
    try:
        text = load_text_from_file(file_path)
        chunker = SemanticChunker()
        chunks = chunker.chunk(text)
        print(f"Chunked into {len(chunks)} chunks.\n")
        for i, chunk in enumerate(chunks[:5]):  
            print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...\n")
        return chunks
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return []


if __name__ == "__main__":
    data_folder = Path("data/")
    for file in data_folder.glob("*"):
        chunk_file(str(file))