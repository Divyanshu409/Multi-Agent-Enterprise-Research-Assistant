from __future__ import annotations

import json
import logging
import sys
import time

import faiss
import numpy as np

from src.config import settings
from src.ingest.chunk import chunk_document, parse_source_header
from src.providers.embeddings import LocalTfidfEmbedding, get_embedding_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")


def load_raw_documents() -> list[tuple[str, str, str]]:
    docs = []
    if not settings.data_raw_dir.exists():
        log.error("data/raw/ does not exist at %s", settings.data_raw_dir)
        return docs
    for path in sorted(settings.data_raw_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        meta, _ = parse_source_header(raw_text)
        company = meta.get("COMPANY", path.stem)
        docs.append((path.stem, company, raw_text))
    return docs


def build_index() -> None:
    start = time.time()
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    docs = load_raw_documents()
    if not docs:
        log.error("No documents found in %s -- nothing to index.", settings.data_raw_dir)
        sys.exit(1)
    log.info("Loaded %d raw document(s) from %s", len(docs), settings.data_raw_dir)

    all_chunks = []
    for doc_id, company, raw_text in docs:
        chunks = chunk_document(
            doc_id=doc_id,
            company=company,
            raw_text=raw_text,
            chunk_size_chars=settings.chunk_size_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        log.info("  %-20s -> %d chunks", doc_id, len(chunks))
        all_chunks.extend(chunks)

    log.info("Total chunks: %d", len(all_chunks))
    texts = [c.text for c in all_chunks]

    # --- embed ---------------------------------------------------------------
    log.info("Embedding provider: %s", settings.embedding_provider)
    if settings.embedding_provider == "local-tfidf":
        embedder = LocalTfidfEmbedding()
        embedder.fit(texts)
        embedder.save(str(settings.data_processed_dir / "tfidf_vectorizer.pkl"))
    else:
        embedder = get_embedding_provider(
            settings.embedding_provider,
            model_name=settings.sentence_transformer_model
            if settings.embedding_provider == "sentence-transformers"
            else settings.openai_embedding_model,
        )

    vectors = embedder.embed(texts)
    matrix = np.array(vectors, dtype=np.float32)
    log.info("Embedded %d chunks -> matrix shape %s (provider=%s, dim=%d)",
              len(texts), matrix.shape, embedder.name, embedder.dimension)

    # --- build FAISS index -----------------------------------------------------
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(settings.faiss_index_path))
    log.info("Wrote FAISS index (%d vectors, dim=%d) -> %s",
              index.ntotal, matrix.shape[1], settings.faiss_index_path)

    # --- persist chunk metadata, row order == FAISS internal id order ---------
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "company": chunk.company,
                "section": chunk.section,
                "text": chunk.text,
            }) + "\n")
    log.info("Wrote chunk metadata -> %s", settings.chunks_path)

    elapsed = time.time() - start
    log.info("Index build complete in %.1fs", elapsed)


if __name__ == "__main__":
    build_index()
