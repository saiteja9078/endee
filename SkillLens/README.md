# SkillLens -- AI-Powered Resume Screening with Vector Search

## Project Overview

SkillLens is an AI-powered resume screening tool that lets recruiters upload resumes and search across them using natural language queries. It combines **semantic vector search** with **LLM-powered analysis** to find the best-matching candidates for any role or skill set.

### Problem Statement

Manual resume screening is time-consuming and error-prone. Recruiters often sift through hundreds of resumes to find candidates matching specific skill requirements. Keyword-based search fails to capture semantic meaning -- a query for "machine learning experience" won't match resumes that mention "trained neural networks" or "built predictive models."

SkillLens solves this with a **Retrieval-Augmented Generation (RAG) pipeline** that:

1. Understands the _meaning_ behind queries, not just keywords
2. Retrieves all resume chunks and scores every resume using hybrid vector search
3. Ranks resumes by top-k mean similarity and lets the LLM control how many to return
4. Generates detailed candidate evaluations streamed in real time

### Use Case

**RAG (Retrieval-Augmented Generation)** for resume retrieval and evaluation, where vector search is the core retrieval mechanism.

---

## System Design

### Architecture

```
User Query
    |
    v
[Prompt Enhancement] --- Gemini rewrites the query for retrieval
    |                     AND extracts desired resume count (k)
    v
[Full Collection Search]  Endee hybrid search across ALL chunks
    |                     (top_k = total elements in index)
    v
[Resume-Level Ranking] -- Group chunks by resume_id, compute
    |                     top-3 chunk mean score per resume
    v
[Top-K Selection] ------- Take top k resumes (k from user query)
    |                     Only these resumes proceed
    v
[Streaming Analysis] ---- Gemini generates a detailed evaluation
    |                     streamed token-by-token via SSE
    v
[Frontend Display] ------ Chat interface shows ranking table,
                          sources, analysis, and resume PDF links
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
|-- agent.py                    # RAG pipeline: prompt enhancement, ranking, streaming
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
# Retrieve ALL chunks to score every resume
total = client.list_indexes()  # get total_elements count
results = index.query(
    vector=query_dense_embedding,
    sparse_indices=query_sparse_indices,
    sparse_values=query_sparse_values,
    top_k=total_elements,  # fetch everything
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

The RAG pipeline in `agent.py` consists of six stages:

### Stage 1: Prompt Enhancement + Count Extraction

The user's natural language query is sent to Gemini, which returns a JSON object containing both the retrieval-optimized query and the desired number of results:

- Input: "find me a good AI engineer"
- Output: `{"query": "AI/ML engineer with deep learning, PyTorch, TensorFlow, NLP...", "count": 1}`

The LLM infers `count` from the user's phrasing: "a candidate" = 1, "top 5" = 5, plural with no number = 3.

### Stage 2: Full Collection Search via Endee

The enhanced prompt is embedded and searched against the entire Endee index using hybrid (dense + sparse) search with `top_k` set to the total number of elements in the collection. This ensures every chunk in every resume receives a similarity score.

### Stage 3: Resume-Level Ranking (Top-K Mean)

Chunks are grouped by `resume_id`. For each resume, the top 3 highest-scoring chunks are selected and their scores are averaged. This produces a single per-resume score that reflects the resume's best-matching sections without being diluted by irrelevant ones.

```python
# Example scoring:
# Resume A chunks: [0.92, 0.85, 0.78, 0.30, 0.10] -> top 3 mean = 0.85
# Resume B chunks: [0.65, 0.60, 0.55, 0.50, 0.48] -> top 3 mean = 0.60
# Result: Resume A ranks higher
```

### Stage 4: Top-K Selection

Resumes are sorted by their mean score and the top `count` resumes (from Stage 1) are selected. Only these resumes' chunks proceed to the analysis step.

### Stage 5: Streaming Analysis

Chunks from the top-ranked resumes are passed to Gemini for evaluation. The response is streamed token-by-token via Server-Sent Events (SSE) for real-time display.

### Stage 6: Resume File Links

After the analysis completes, the pipeline yields clickable links to the original PDF files of the top-ranked resumes, allowing recruiters to open and review them directly in a new browser tab.

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

3. **Resume-level ranking with top-k mean**: Instead of returning individual chunks, SkillLens scores entire resumes by averaging their top 3 chunk similarity scores. This prevents resumes with more sections from being penalized (mean dilution) and ensures ranking reflects the strongest matching areas.

4. **LLM-controlled result count**: The number of resumes returned is extracted from the user's natural language query by the LLM ("a candidate" = 1, "top 5" = 5), eliminating the need for manual configuration.

5. **Server-Sent Events for streaming**: SSE provides real-time token-by-token display of the LLM response without the complexity of WebSockets, giving users immediate feedback while the analysis is generated.

---

## License

This project is part of the Endee internship evaluation program.
