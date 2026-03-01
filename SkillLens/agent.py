"""
RAG Agent — Google Generative AI + Endee
=========================================
Prompt enhancement → Endee hybrid search → streaming answer generation.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from search_engine.search_engine import SearchEngine

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

guidelines = """You are a prompt enhancement agent operating inside a Retrieval-Augmented Generation (RAG) pipeline for resume retrieval.
Your task is to rewrite and enhance the user's input into a clear, structured, and retrieval-optimized query that maximizes the relevance of resumes fetched from the vector database.

Constraints:
- You are given only one opportunity to enhance the prompt.
- You must NOT ask follow-up questions.
- You must NOT add assumptions that are not reasonably inferable from the user input.
- Do NOT generate answers, summaries, or explanations.

Enhancement Guidelines:
- Extract and explicitly state key skills, technologies, roles, experience level, domain, and constraints if present.
- Normalize vague terms into concrete, searchable phrasing (e.g., "good experience" → "2+ years hands-on experience" only if implied).
- Preserve the original intent and meaning of the user query.
- Output a single, concise, well-structured retrieval query optimized for semantic search over resumes.

Output Format:
- Return ONLY the enhanced prompt.
- No metadata, no commentary, no markdown."""


def execute(prompt: str, collection_name: str, limit: int = 8):
    """
    RAG pipeline generator.

    Pipeline:
      1. Enhance user prompt for retrieval
      2. Hybrid search via Endee
      3. LLM verifies which chunks are relevant
      4. Stream final answer using only verified chunks
      5. Yield verified resume file links

    Yields:
      - dict with '__sources__'       : verified source chunks
      - str tokens                    : streamed LLM answer
      - dict with '__resume_files__'  : links to verified resumes
    """
    import json as _json
    from pathlib import Path as _Path

    # Step 1: Enhance the prompt
    enhance_temp = ChatPromptTemplate.from_messages(
        [
            ("system", guidelines),
            ("user", "prompt: {prompt}"),
        ]
    )
    enhanced_prompt = model.invoke(
        enhance_temp.format_messages(prompt=prompt)
    ).content

    # Step 2: Hybrid search via Endee
    engine = SearchEngine(collection_name=collection_name)
    results = engine.hybrid_search(enhanced_prompt, limit)

    # Build source list
    sources = []
    for item in results:
        meta = item.get("meta", {})
        sources.append({
            "person_name": meta.get("person_name", "Unknown"),
            "section": meta.get("section", ""),
            "sub_section": meta.get("sub_section", ""),
            "level": meta.get("level", ""),
            "content": meta.get("content", ""),
            "content_with_context": meta.get(
                "content_with_context", meta.get("content", str(meta))
            ),
            "resume_id": meta.get("resume_id", ""),
            "similarity": item.get("similarity", "N/A"),
        })

    # Step 3: LLM verifies which chunks are relevant
    verify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a resume chunk relevance filter. Given a user query and "
                "a list of retrieved resume chunks, determine which chunks are "
                "actually relevant to the query.\n\n"
                "Return ONLY a valid JSON object with this exact format:\n"
                '{{"relevant_ids": ["resume_id_1", "resume_id_2"]}}\n\n'
                "Rules:\n"
                "- Only include resume_ids whose chunks genuinely match the query\n"
                "- If no chunks are relevant, return {{\"relevant_ids\": []}}\n"
                "- Do NOT include any text before or after the JSON\n"
                "- Do NOT wrap in markdown code blocks",
            ),
            (
                "user",
                "Query: {prompt}\n\n"
                "Retrieved chunks:\n{chunks}",
            ),
        ]
    )

    # Format chunks for verification
    verify_chunks = ""
    for i, s in enumerate(sources):
        verify_chunks += (
            f"\n--- Chunk {i+1} ---\n"
            f"resume_id: {s['resume_id']}\n"
            f"person_name: {s['person_name']}\n"
            f"section: {s['section']}\n"
            f"content: {s['content'][:300]}\n"
        )

    verify_resp = model.invoke(
        verify_prompt.format_messages(prompt=prompt, chunks=verify_chunks)
    ).content.strip()

    # Parse relevant IDs
    relevant_ids = set()
    try:
        # Try to extract JSON from the response
        json_start = verify_resp.find("{")
        json_end = verify_resp.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = _json.loads(verify_resp[json_start:json_end])
            relevant_ids = set(parsed.get("relevant_ids", []))
    except Exception:
        # If parsing fails, fall back to all sources
        relevant_ids = {s["resume_id"] for s in sources if s["resume_id"]}

    # If LLM returned nothing, fall back to top results
    if not relevant_ids:
        relevant_ids = {s["resume_id"] for s in sources if s["resume_id"]}

    # Filter sources to only verified ones
    verified_sources = [s for s in sources if s["resume_id"] in relevant_ids]

    # Yield only verified sources
    yield {"__sources__": verified_sources}

    # Step 4: Stream the final answer using only verified chunks
    verified_chunks_text = ""
    for s in verified_sources:
        content = s.get("content_with_context", s.get("content", ""))
        similarity = s.get("similarity", "N/A")
        verified_chunks_text += f"\n---\n[Score: {similarity}]\n{content}\n"

    answer_temp = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a resume evaluator. You have to evaluate the resume chunks "
                "based on the user's prompt and give them the best matching resumes with "
                "detailed explanation.\n\nResume chunks:\n{chunks}",
            ),
            ("user", "prompt: {prompt}"),
        ]
    )
    formatted = answer_temp.format_messages(
        chunks=verified_chunks_text, prompt=prompt
    )
    for ch in model.stream(formatted):
        yield ch.content

    # Step 5: Yield verified resume files
    uploads_dir = _Path(__file__).parent / "uploads"
    seen = set()
    resume_files = []
    for s in verified_sources:
        rid = s.get("resume_id", "")
        if rid and rid not in seen:
            seen.add(rid)
            actual_filename = rid + ".pdf"
            for f in uploads_dir.iterdir():
                if f.stem == rid:
                    actual_filename = f.name
                    break
            resume_files.append({
                "resume_id": rid,
                "person_name": s.get("person_name", "Unknown"),
                "filename": actual_filename,
            })
    yield {"__resume_files__": resume_files}