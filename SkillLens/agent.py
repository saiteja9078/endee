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
Your task is to:
1. Rewrite and enhance the user's input into a retrieval-optimized query
2. Determine how many resumes the user wants returned

Constraints:
- You are given only one opportunity to enhance the prompt.
- You must NOT ask follow-up questions.
- You must NOT add assumptions that are not reasonably inferable from the user input.
- Do NOT generate answers, summaries, or explanations.

Enhancement Guidelines:
- Extract and explicitly state key skills, technologies, roles, experience level, domain, and constraints if present if not present then mention the skills / experience or anything that has to be need for user query.
- Normalize vague terms into concrete, searchable phrasing.
- Preserve the original intent and meaning of the user query.

Number Detection:
- If the user says "a candidate" or "one resume" -> count = 1
- If the user says "top 3" or "3 candidates" -> count = 3
- If the user says "find candidates" (plural, no number) -> count = 3
- If the user says "all matching" or "all resumes" -> count = 10
- Default to 3 if unclear.

Output Format:
- Return ONLY a valid JSON object, nothing else:
  {{"query": "the enhanced retrieval query", "count": 3}}
- No markdown, no code blocks, no commentary."""


def rank_resumes(sources, top_k_chunks=3, top_n_resumes=5):
    """
    Rank resumes by top-k mean similarity.

    For each resume:
      1. Collect all chunk scores
      2. Sort descending, take top `top_k_chunks`
      3. Compute mean of those top scores

    Returns:
      - Sorted list of (resume_id, mean_score, chunk_count)
      - Dict mapping resume_id -> list of source chunks
    """
    from collections import defaultdict

    resume_chunks = defaultdict(list)
    for s in sources:
        rid = s.get("resume_id", "")
        if rid:
            resume_chunks[rid].append(s)

    # Compute top-k mean per resume
    resume_scores = []
    for rid, chunks in resume_chunks.items():
        scores = sorted(
            [c["similarity"] for c in chunks if isinstance(c["similarity"], (int, float))],
            reverse=True,
        )
        top_scores = scores[:top_k_chunks]
        if top_scores:
            mean_score = sum(top_scores) / len(top_scores)
        else:
            mean_score = 0.0
        resume_scores.append((rid, mean_score, len(chunks)))

    # Sort by mean score descending
    resume_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top N resumes
    top_resumes = resume_scores[:top_n_resumes]
    top_ids = {r[0] for r in top_resumes}

    return top_resumes, {rid: resume_chunks[rid] for rid in top_ids}


def execute(prompt: str, collection_name: str, top_n_resumes: int = 3):
    """
    RAG pipeline with resume-level ranking.

    Pipeline:
      1. Enhance user prompt + extract desired count
      2. Retrieve ALL chunks from Endee (single query)
      3. Group by resume_id, compute top-3 mean score per resume
      4. Rank resumes, take top N (N from user query)
      5. Stream LLM analysis using only top-ranked resumes
      6. Yield resume file links

    Yields:
      - dict '__ranking__'      : resume scores and ranking
      - dict '__sources__'      : chunks from top-ranked resumes
      - str tokens              : streamed LLM answer
      - dict '__resume_files__' : links to top-ranked resumes
    """
    import json as _json
    from pathlib import Path as _Path

    # Step 1: Enhance the prompt + extract desired count
    enhance_temp = ChatPromptTemplate.from_messages(
        [
            ("system", guidelines),
            ("user", "prompt: {prompt}"),
        ]
    )
    enhance_resp = model.invoke(
        enhance_temp.format_messages(prompt=prompt)
    ).content.strip()

    # Parse JSON response
    enhanced_prompt = prompt  # fallback
    try:
        json_start = enhance_resp.find("{")
        json_end = enhance_resp.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = _json.loads(enhance_resp[json_start:json_end])
            enhanced_prompt = parsed.get("query", prompt)
            top_n_resumes = parsed.get("count", top_n_resumes)
    except Exception:
        enhanced_prompt = enhance_resp

    # Step 2: Retrieve ALL chunks from the collection
    engine = SearchEngine(collection_name=collection_name)
    total = engine.get_total_elements()
    if total == 0:
        total = 50  # fallback

    results = engine.hybrid_search(enhanced_prompt, limit=max(total, 50))

    # Build source list with scores
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
            "similarity": item.get("similarity", 0.0),
        })

    # Step 3: Rank resumes by top-3 mean similarity
    resume_scores, resume_chunks = rank_resumes(
        sources, top_k_chunks=3, top_n_resumes=top_n_resumes
    )

    # Yield ranking info for the frontend
    yield {"__ranking__": [
        {"resume_id": rid, "score": round(score, 4), "chunks_matched": count}
        for rid, score, count in resume_scores
    ]}

    # Get the top-ranked resumes' chunks
    top_ids = {r[0] for r in resume_scores[:top_n_resumes]}
    verified_sources = [s for s in sources if s["resume_id"] in top_ids]

    # Yield verified sources
    yield {"__sources__": verified_sources}

    # Step 4: Stream the final answer using only top-ranked chunks
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
                "detailed explanation. The resumes have been ranked by similarity score "
                "(top-3 chunk mean). Focus on explaining why the top-ranked candidates "
                "are the best match.\n\nResume chunks:\n{chunks}",
            ),
            ("user", "prompt: {prompt}"),
        ]
    )
    formatted = answer_temp.format_messages(
        chunks=verified_chunks_text, prompt=prompt
    )
    for ch in model.stream(formatted):
        yield ch.content

    # Step 5: Yield top-ranked resume files
    uploads_dir = _Path(__file__).parent / "uploads"
    seen = set()
    resume_files = []
    for rid, score, count in resume_scores[:top_n_resumes]:
        if rid and rid not in seen:
            seen.add(rid)
            actual_filename = rid + ".pdf"
            if uploads_dir.exists():
                for f in uploads_dir.iterdir():
                    if f.stem == rid:
                        actual_filename = f.name
                        break
            # Get person name from chunks
            pname = "Unknown"
            for s in verified_sources:
                if s["resume_id"] == rid:
                    pname = s["person_name"]
                    break
            resume_files.append({
                "resume_id": rid,
                "person_name": pname,
                "filename": actual_filename,
                "score": round(score, 4),
            })
    yield {"__resume_files__": resume_files}