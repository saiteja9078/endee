# SkillLens -- AI-Powered Resume Screening with Vector Search

## Project Overview

SkillLens is an AI-powered resume screening tool that lets recruiters upload resumes and search across them using natural language queries. It combines **semantic vector search** with **LLM-powered analysis** to find the best-matching candidates for any role or skill set.

### Problem Statement

Manual resume screening is time-consuming and error-prone. Recruiters often sift through hundreds of resumes to find candidates matching specific skill requirements. Keyword-based search fails to capture semantic meaning -- a query for "machine learning experience" won't match resumes that mention "trained neural networks" or "built predictive models."

SkillLens solves this with a **Retrieval-Augmented Generation (RAG) pipeline** that:

1. Understands the _meaning_ behind queries, not just keywords
2. Retrieves the most relevant resume sections using hybrid vector search
3. Uses an LLM to verify relevance and generate detailed candidate evaluations
4. Presents results through a clean, real-time streaming interface

### Use Case

**RAG (Retrieval-Augmented Generation)** for resume retrieval and evaluation, where vector search is the core retrieval mechanism.

---

## System Design

### Architecture

```
User Query
    |
    v
[Prompt Enhancement] --- Gemini 2.5 Flash rewrites the query
    |                     for optimal retrieval
    v
[Hybrid Search] --------- Endee performs dense + sparse vector search
    |                     across indexed resume chunks
    v
[LLM Verification] ------ Gemini filters retrieved chunks for relevance
    |                     (removes false positives)
    v
[Streaming Analysis] ---- Gemini generates a detailed evaluation
    |                     streamed token-by-token via SSE
    v
[Frontend Display] ------ Chat interface shows verified sources,
                          analysis, and clickable resume links
```

### Technical Stack

| Component         | Technology                                     |
| ----------------- | ---------------------------------------------- |
| Vector Database   | **Endee** (local instance, port 8080)          |
| LLM               | Google Gemini 2.5 Flash (via LangChain)        |
| Dense Embeddings  | BAAI/bge-base-en-v1.5 (768-dim, via fastembed) |
| Sparse Embeddings | SPLADE++ (prithvida/Splade_PP_en_v1)           |
| Backend           | Python, Quart (async), Server-Sent Events      |
| Frontend          | HTML, CSS, JavaScript (no frameworks)          |
| PDF Parsing       | PyMuPDF (fitz)                                 |

### Project Structure

```
SkillLens/
|-- agent.py                    # RAG pipeline: prompt enhancement, verification, streaming
|-- app.py                      # Quart web server with REST + SSE endpoints
|-- requirements.txt            # Python dependencies
|-- .env                        # Environment variables (GOOGLE_API_KEY)
|-- templates/
|   |-- index.html              # Frontend: chat interface, upload, collection management
|-- search_engine/
|   |-- search_engine.py        # Endee client: index creation, upsert, hybrid search
|   |-- build_collection.py     # Ingestion pipeline: PDF -> chunks -> Endee
|-- chunking/
|   |-- extract.py              # PDF/DOCX text extraction and section detection
|   |-- chunker.py              # Hierarchical chunking with metadata
|-- uploads/                    # Uploaded resume files (gitignored)
```

---

## How Endee Is Used

[Endee](https://github.com/endee-io/endee) serves as the **vector database** at the core of the retrieval pipeline. Here is how it is integrated:

### 1. Index Creation

When resumes are uploaded, SkillLens creates an Endee index configured for **hybrid search** (dense + sparse vectors):

```python
client = Endee()
client.set_base_url("http://127.0.0.1:8080/api/v1")

client.create_index(
    name="skilllens_resumes",
    dimension=768,          # bge-base-en-v1.5 embedding dimension
    sparse_dim=30000,       # SPLADE vocabulary size
    space_type="cosine",
)
```

### 2. Document Ingestion

Each resume is processed through a pipeline: **PDF parsing -> section detection -> hierarchical chunking -> embedding -> Endee upsert**.

Chunks are embedded using both dense (BGE) and sparse (SPLADE) models, then stored in Endee with metadata:

```python
index = client.get_index(name=collection_name)

index.upsert(
    documents=[{
        "id": chunk_id,
        "vector": dense_embedding,           # 768-dim float vector
        "sparse_indices": sparse_indices,     # SPLADE token positions
        "sparse_values": sparse_values,       # SPLADE token weights
        "meta": {
            "person_name": "john doe",
            "section": "technical skills",
            "content": "Python, PyTorch, LangChain...",
            "resume_id": "john_doe_resume",
            ...
        }
    }]
)
```

### 3. Hybrid Search

At query time, the user's query is embedded with the same models and searched against Endee using both dense and sparse vectors simultaneously:

```python
results = index.query(
    vector=query_dense_embedding,
    sparse_indices=query_sparse_indices,
    sparse_values=query_sparse_values,
    top_k=8,
)
```

This hybrid approach combines:

- **Dense search** (BGE embeddings): captures semantic meaning ("machine learning" matches "neural networks")
- **Sparse search** (SPLADE): captures exact keyword matches and rare terms

### 4. Collection Management

The application uses the Endee SDK to list, create, and query multiple indexes (collections), allowing users to organize resumes by job posting, department, or batch:

```python
result = client.list_indexes()   # List all collections
index = client.get_index(name="collection_name")  # Access specific collection
```

---

## RAG Pipeline Detail

The RAG pipeline in `agent.py` consists of five stages:

### Stage 1: Prompt Enhancement

The user's natural language query is rewritten by Gemini into a retrieval-optimized form. For example:

- Input: "find me a good AI engineer"
- Enhanced: "Software engineer with hands-on experience in AI/ML, deep learning frameworks (PyTorch, TensorFlow), NLP, and production ML pipelines"

### Stage 2: Hybrid Search via Endee

The enhanced prompt is embedded and searched against the Endee index using hybrid (dense + sparse) search, returning the top-k most similar resume chunks.

### Stage 3: LLM Relevance Verification

Retrieved chunks are sent to Gemini for relevance filtering. The LLM evaluates each chunk against the original query and returns only the resume IDs that genuinely match. This step eliminates false positives from vector search.

### Stage 4: Streaming Analysis

Only verified chunks are passed to Gemini for the final evaluation. The response is streamed token-by-token via Server-Sent Events (SSE) for real-time display.

### Stage 5: Resume File Links

After the analysis completes, the pipeline yields links to the actual PDF files of verified resumes, allowing recruiters to open and review them directly.

---

## Setup and Execution

### Prerequisites

- Python 3.11+
- Git
- A Google API key for Gemini (get one at https://aistudio.google.com/apikey)

### Step 1: Clone the Repository

```bash
git clone https://github.com/saiteja9078/endee.git
cd endee
```

### Step 2: Start the Endee Server

Endee runs as a local server. From the project root:

```bash
./run.sh
```

Verify it is running by visiting http://localhost:8080 in your browser. You should see the Endee dashboard.

### Step 3: Set Up the SkillLens Application

```bash
cd SkillLens
```

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the `SkillLens/` directory:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Replace `your_google_api_key_here` with your actual Gemini API key.

### Step 5: Run the Application

```bash
python app.py
```

The server starts at http://localhost:5001.

### Step 6: Use the Application

1. Open http://localhost:5001 in your browser
2. **Upload resumes**: Use the sidebar to drag-and-drop or browse for PDF/DOCX files. Enter a collection name and click "Upload"
3. **Select a collection**: Click a collection in the sidebar or use the dropdown in the header
4. **Search**: Type a natural language query in the chat input (e.g., "Find candidates with Python and machine learning experience") and press Enter
5. **Review results**: The interface shows retrieved chunks, a streamed LLM analysis, and clickable links to the verified resume files

---

## API Endpoints

| Method | Endpoint            | Description                    |
| ------ | ------------------- | ------------------------------ |
| GET    | `/`                 | Serves the frontend            |
| POST   | `/upload`           | Upload and ingest resume files |
| GET    | `/query`            | SSE streaming RAG query        |
| GET    | `/collections`      | List all Endee indexes         |
| GET    | `/files/<filename>` | Serve an uploaded resume file  |

---

## Key Design Decisions

1. **Hybrid search over pure dense search**: Combining dense (semantic) and sparse (keyword) embeddings provides better recall than either approach alone. Dense search captures meaning while sparse search catches exact terms and rare keywords.

2. **Hierarchical chunking**: Resumes are chunked at two levels -- full sections (e.g., all of "Projects") and individual entries (e.g., one specific project). This gives the retrieval system both broad context and fine-grained detail.

3. **LLM verification step**: Vector search can return false positives. Adding a verification step where the LLM filters chunks before the final analysis ensures only genuinely relevant resumes are presented to the user.

4. **Server-Sent Events for streaming**: SSE provides real-time token-by-token display of the LLM response without the complexity of WebSockets, giving users immediate feedback while the analysis is generated.

---

## License

This project is part of the Endee internship evaluation program.
