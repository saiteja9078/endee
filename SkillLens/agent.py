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
    RAG pipeline generator — yields streamed answer tokens.

    1. Enhance user prompt for better retrieval
    2. Hybrid search in Endee
    3. Stream the final answer
    """
    # Step 1: Enhance the prompt
    enhance_temp = ChatPromptTemplate.from_messages(
        [
            ("system", guidelines),
            ("user", "prompt: {prompt}"),
        ]
    )
    resp = model.invoke(enhance_temp.format_messages(prompt=prompt)).content
    enhanced_prompt = resp

    # Step 2: Hybrid search via Endee
    engine = SearchEngine(collection_name=collection_name)
    results = engine.hybrid_search(enhanced_prompt, limit)

    # Format results for the LLM context
    chunks_text = ""
    for item in results:
        meta = item.get("meta", {})
        content = meta.get("content_with_context", meta.get("content", str(meta)))
        similarity = item.get("similarity", "N/A")
        chunks_text += f"\n---\n[Score: {similarity}]\n{content}\n"

    # Step 3: Stream the answer
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
    formatted = answer_temp.format_messages(chunks=chunks_text, prompt=prompt)
    for ch in model.stream(formatted):
        yield ch.content