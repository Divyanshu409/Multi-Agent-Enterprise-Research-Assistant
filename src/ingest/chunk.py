
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company: str
    section: str
    text: str


_SECTION_HEADER_RE = re.compile(r"^(ITEM\s+\d+[A-Z]?\.\s+.+)$", re.MULTILINE)


def parse_source_header(raw_text: str) -> tuple[dict, str]:

    lines = raw_text.split("\n")
    meta = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("=" * 10):
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().upper()] = value.strip()
    body = "\n".join(lines[body_start:]).strip()
    return meta, body


def _current_section(offset: int, headers: list[tuple[int, str]]) -> str:
    section = "GENERAL"
    for pos, header in headers:
        if pos <= offset:
            section = header
        else:
            break
    return section


def chunk_document(
    doc_id: str,
    company: str,
    raw_text: str,
    chunk_size_chars: int = 1400,
    overlap_chars: int = 200,
) -> list[Chunk]:
    _, body = parse_source_header(raw_text)

    headers = [(m.start(), m.group(1).strip()) for m in _SECTION_HEADER_RE.finditer(body)]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""
    buffer_start_offset = 0
    running_offset = 0
    chunk_idx = 0

    def flush(end_offset: int):
        nonlocal buffer, chunk_idx
        if not buffer.strip():
            return
        section = _current_section(buffer_start_offset, headers)
        chunk_id = f"{doc_id}_{chunk_idx:03d}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, company=company, section=section, text=buffer.strip()))
        chunk_idx += 1

    for para in paragraphs:
        if len(buffer) + len(para) + 1 > chunk_size_chars and buffer:
            flush(running_offset)
            overlap_text = buffer[-overlap_chars:] if overlap_chars else ""
            buffer = (overlap_text + "\n" + para).strip()
            buffer_start_offset = running_offset
        else:
            if not buffer:
                buffer_start_offset = running_offset
            buffer = (buffer + "\n" + para).strip() if buffer else para
        running_offset += len(para) + 2

    flush(running_offset)
    return chunks
