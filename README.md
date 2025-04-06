
Agentic **Retrieval-Augmented Generation (RAG) chatbot** powered by Crew AI, Qdrant, Chonkie (semantic chunking), and DeepEval.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

---

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Environment Variables

## Together.ai API Setup (For Model Generation)

This project uses Together.ai’s hosted `meta-llama/Llama-3.3-70B-Instruct-Turbo"` model for generation.

1. Sign up at [Together.ai Console](https://console.together.ai/)
2. Go to your account settings and create a new API key
3. Add the key to your `.env` file under `TOGETHER_API_KEY`
4. No billing is required — they provide $1 in credit with no per minute rate limiting allowing for plenty of testing in this case

---

### 4. Set up Qdrant via Docker Desktop

If you’re running locally, Qdrant is easiest to setup up with Docker:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Run Qdrant in a container:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
This exposes the Qdrant API at `http://localhost:6333`.

---

## Uploading Files to Qdrant

To upload a document and index its chunks in Qdrant:

1. Place your `.txt` file in the `data/` folder.
2. In `src/qdrant_upload/qdrant_connection.py`, and 'src/tools/vector_search.py' make sure the Qdrant client points to your instance:
```python
client = QdrantClient(url="http://localhost:6333")  # Or your custom host
```
3. Run the script to upload:
```bash
PYTHONPATH=src python src/qdrant_upload/qdrant_connection.py
```
You will be prompted to enter the filename to upload.

---

## Running the Main Chatbot Pipeline

Start the dynamic agentic chatbot pipeline:

```bash
PYTHONPATH=src python src/workflows/main_pipeline.py
```

You’ll be prompted to enter a query. The system will dynamically decide which agents to run (including clarification, retrieval, and generation).

---

## Running DeepEval Evaluation (Optional)

To evaluate the chatbot’s retrieval and generation quality using DeepEval metrics:

```bash
PYTHONPATH=src python tests/deepeval_test.py
```

---

## Project Structure Highlights

```
agentic-rag-chatbot/
│
├── src/
│   ├── workflows/              # Main Crew AI task pipeline
│   ├── qdrant_upload/          # Scripts to chunk/upload docs to Qdrant
│   │   ├── qdrant_connection.py
│   │   └── chunk_utils.py
│   └── ...
│
├── data/                       # Place your input documents here
├── tests/
│   └── deepeval_test.py        # Evaluation using DeepEval
├── .env                        # Place your API keys here
└── requirements.txt
```

---

## Notes

- Make sure Qdrant is running before uploading files or starting the pipeline.
- The `deepeval_test.py` script requires an OpenAI API key -- which is what I used.
- If you haven’t used OpenAI’s API before, you may need to enable billing (a small credit card authorization may be required). Alternatively, DeepEval also supports custom or self-hosted models if preferred.

---

