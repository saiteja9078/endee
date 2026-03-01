"""
SkillLens — Quart Web Application
====================================
Resume upload + RAG-powered streaming chat using Endee vector DB
and Google Generative AI.

Run:  python app.py
"""

import os
import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from quart import Quart, request, jsonify, Response, send_from_directory
from quart_cors import cors

from search_engine.build_collection import collection_init
from agent import execute

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()

app = Quart(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)
app = cors(app, allow_origin="*")

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"

DEFAULT_COLLECTION = "skilllens_resumes"


@app.route("/")
async def index():
    """Serve the frontend."""
    return await send_from_directory(str(TEMPLATES_DIR), "index.html")


@app.route("/files/<path:filename>")
async def serve_file(filename):
    """Serve an uploaded resume file."""
    return await send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/upload", methods=["POST"])
async def upload():
    """
    Upload one or more resume files (PDF/DOCX).
    Form fields:
      - files: one or more files
      - collection: (optional) collection/index name
    """
    try:
        files = await request.files
        form = await request.form

        collection = form.get("collection", DEFAULT_COLLECTION).strip()
        if not collection:
            collection = DEFAULT_COLLECTION
        # Sanitize: replace spaces and special chars
        collection = collection.replace(" ", "_").lower()

        uploaded = files.getlist("files")
        if not uploaded:
            return jsonify({"error": "No files uploaded"}), 400

        saved_paths = []
        for f in uploaded:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in (".pdf", ".docx", ".doc"):
                continue
            dest = UPLOAD_DIR / f.filename
            await f.save(str(dest))
            saved_paths.append(str(dest))

        if not saved_paths:
            return jsonify({"error": "No valid PDF/DOCX files found"}), 400

        # Run the ingestion pipeline in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, collection_init, saved_paths, collection)

        return jsonify({
            "status": "success",
            "message": f"Ingested {len(saved_paths)} resume(s) into '{collection}'",
            "files": [Path(p).name for p in saved_paths],
            "collection": collection,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/collections")
async def list_collections():
    """List all existing Endee indexes."""
    try:
        from endee import Endee
        client = Endee()
        client.set_base_url("http://127.0.0.1:8080/api/v1")
        result = client.list_indexes()
        index_list = result.get("indexes", []) if isinstance(result, dict) else result
        names = []
        for idx in index_list:
            if isinstance(idx, dict):
                name = idx.get("name", "")
            else:
                name = str(idx)
            if name:
                names.append(name)
        return jsonify({"collections": names})
    except Exception as e:
        return jsonify({"collections": [], "error": str(e)})


@app.route("/query")
async def query():
    """
    SSE streaming endpoint for RAG queries.
    Query params:
      - q: the user query
      - collection: (optional) collection name
      - limit: (optional) number of results, default 8
    """
    import json as json_mod

    user_query = request.args.get("q", "").strip()
    if not user_query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    collection = request.args.get("collection", DEFAULT_COLLECTION).strip()
    limit = int(request.args.get("limit", "3"))

    async def generate():
        loop = asyncio.get_event_loop()

        import queue
        import threading

        token_queue = queue.Queue()
        done_sentinel = object()

        def run_agent():
            try:
                for token in execute(prompt=user_query, collection_name=collection, top_n_resumes=limit):
                    if isinstance(token, dict):
                        # Sources payload — serialize as JSON
                        token_queue.put(token)
                    elif token is not None and token != "":
                        token_queue.put(str(token))
            except Exception as e:
                token_queue.put(f"[Error: {e}]")
            finally:
                token_queue.put(done_sentinel)

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        while True:
            try:
                item = await loop.run_in_executor(None, token_queue.get, True, 60)
            except Exception:
                break

            if item is done_sentinel:
                break

            # JSON-encode to safely handle newlines and special chars
            safe_data = json_mod.dumps(item)
            yield f"data: {safe_data}\n\n"

        yield "data: \"[DONE]\"\n\n"

    return Response(
        generate(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
