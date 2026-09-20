#!/usr/bin/env python3
"""Indexa el corpus de docs en embeddings locales (Ollama) para RAG.

Uso:
  python3 build_index.py                    # indexa el proyecto actual (cwd)
  python3 build_index.py /ruta/al/proyecto   # indexa ese proyecto puntual
  python3 build_index.py --config config.yaml  # modo legacy, config explícito
"""
import glob
import json
import os
import sys
import time

import requests
import yaml

import project_config

OLLAMA_URL = "http://localhost:11434/api/embeddings"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_cli_config():
    args = sys.argv[1:]
    if args and args[0] == "--config":
        return load_config(args[1])
    proyecto = args[0] if args else os.getcwd()
    return project_config.resolve(proyecto)


def collect_files(cfg):
    root = cfg["corpus_root"]
    excluded = set()
    for pat in cfg.get("exclude", []):
        excluded.update(glob.glob(os.path.join(root, pat), recursive=True))

    files = []
    for pat in cfg["include"]:
        for path in glob.glob(os.path.join(root, pat), recursive=True):
            if path in excluded or not os.path.isfile(path):
                continue
            files.append(path)
    return sorted(set(files))


import re

_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _split_by_fixed_size(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0 or end >= len(text):
            break
    return chunks


def chunk_text(text, size, overlap):
    """Parte por secciones Markdown (## headers) para no cortar tablas/listas a la mitad.
    Una sección que igual supere `size` se subdivide con la ventana fija de antes."""
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        return _split_by_fixed_size(text, size, overlap)

    sections = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append(text[start:end])
    if headers[0].start() > 0:
        sections.insert(0, text[: headers[0].start()])

    chunks = []
    for section in sections:
        if len(section) <= size * 1.3:
            chunks.append(section)
        else:
            chunks.extend(_split_by_fixed_size(section, size, overlap))
    return chunks


def embed(text, model):
    resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=60)
    if resp.status_code == 500 and "context length" in resp.text:
        # chunk denso (código/tablas) excede los 512 tokens del modelo de embeddings: truncar y reintentar
        text = text[: len(text) // 2]
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def build(cfg, log=print):
    """Indexa un proyecto ya resuelto (cfg de project_config.resolve o config.yaml legacy).
    Reusable tanto desde la CLI como desde el demonio (indexado automático on-demand)."""
    files = collect_files(cfg)
    log(f"Archivos a indexar: {len(files)}")

    entries = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        rel = os.path.relpath(path, cfg["corpus_root"])
        j = -1
        for j, chunk in enumerate(chunk_text(text, cfg["chunk_size_chars"], cfg["chunk_overlap_chars"])):
            if not chunk.strip():
                continue
            vec = embed(chunk, cfg["embed_model"])
            entries.append({"file": rel, "chunk_id": j, "text": chunk, "embedding": vec})
        log(f"  [{i}/{len(files)}] {rel} -> {j + 1} chunks")

    with open(cfg["index_path"], "w") as f:
        json.dump(entries, f)

    log(f"Listo: {len(entries)} chunks indexados en {cfg['index_path']} ({time.time() - t0:.1f}s)")
    return len(entries)


def main():
    cfg = resolve_cli_config()
    print(f"Proyecto: {cfg['corpus_root']}")
    print(f"Índice: {cfg['index_path']}")
    build(cfg)


if __name__ == "__main__":
    main()
