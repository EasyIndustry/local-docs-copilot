#!/usr/bin/env python3
"""Retrieval por similitud coseno sobre el índice generado por build_index.py."""
import json
import math
import os
import re

import requests

# Retrieval puramente semántico es frágil a la redacción exacta de la pregunta: la misma
# pregunta reformulada puede hacer caer el chunk correcto muchos puestos en el ranking, porque
# nombres propios/siglas técnicas pesan poco en el embedding si el resto de la frase cambia.
# Boost simple de keyword exacto como mitigación barata: términos "tipo nombre propio"
# de la pregunta (con mayúscula que no sea de arranque de oración, o siglas) que aparecen
# literalmente en un chunk lo empujan hacia arriba antes de rankear por coseno.
_KEYWORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚñÑ]{3,}")
_KEYWORD_BOOST = 0.15
_STOPWORDS_INICIALES = {"por", "qué", "que", "cómo", "como", "cuál", "cual", "según", "cuáles"}


def _extract_keywords(query):
    words = _KEYWORD_RE.findall(query)[1:]  # descarta la primera palabra (mayúscula de arranque)
    return {
        w for w in words
        if (w.isupper() or (w[0].isupper() and w[1:].islower()))
        and w.lower() not in _STOPWORDS_INICIALES
    }

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"

# Filosofía: preferir arranque en frío y liberar VRAM cuando no se usa, en vez de tener el
# modelo cargado todo el tiempo por "las dudas" — la GPU la necesita el usuario para otras
# cosas (dev, juegos, etc.). keep_alive corto en vez de "Forever" (default de este servidor).
KEEP_ALIVE_DEFAULT = "2m"

_INDEX_CACHE = None


def precalentar(model, keep_alive=KEEP_ALIVE_DEFAULT):
    """Carga el modelo a VRAM sin generar nada (prompt vacío = solo load, según la API de
    Ollama). Para que un agente dispare esto ANTES de saber que va a necesitar el modelo, y
    siga con otra cosa mientras carga, en vez de pagar el arranque en frío recién al pedir."""
    requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": "", "keep_alive": keep_alive},
        timeout=120,
    )


def modelo_cargado(model):
    """True si el modelo ya está residente en VRAM (para no bloquear esperando la carga)."""
    resp = requests.get(OLLAMA_PS_URL, timeout=10)
    resp.raise_for_status()
    return any(m["name"] == model for m in resp.json().get("models", []))


def _load_index(index_path):
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        with open(index_path) as f:
            _INDEX_CACHE = json.load(f)
    return _INDEX_CACHE


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_query(text, model="mxbai-embed-large", keep_alive=KEEP_ALIVE_DEFAULT):
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": model, "prompt": text, "keep_alive": keep_alive},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def search(query, index_path, top_k=10, embed_model="mxbai-embed-large"):
    index = _load_index(index_path)
    qvec = embed_query(query, embed_model)
    keywords = _extract_keywords(query)

    scored = []
    for e in index:
        score = _cosine(qvec, e["embedding"])
        n_matches = sum(1 for k in keywords if k in e["text"])
        score += n_matches * _KEYWORD_BOOST
        scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "file": e["file"], "text": e["text"]} for s, e in scored[:top_k]]


def ask(query, index_path, model, top_k=10, embed_model="mxbai-embed-large"):
    """RAG completo: recupera contexto y le pide al modelo local que responda citando fuente."""
    hits = search(query, index_path, top_k=top_k, embed_model=embed_model)
    context = "\n\n".join(f"[{h['file']}]\n{h['text']}" for h in hits)
    prompt = (
        "Respondé la pregunta usando SOLO el contexto de documentación de abajo. "
        "Si el contexto no alcanza o la doc misma dice que algo está incompleto/desactualizado, decilo explícitamente. "
        "Citá el archivo fuente entre corchetes.\n\n"
        f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta:"
    )
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        # temperature baja: qwen3:4b-instruct a veces dice "el contexto no alcanza" con
        # sampling default aunque la info correcta esté en el contexto (falso negativo
        # intermitente, no una falla de retrieval) — bajar la temperatura lo hace más
        # consistente en usar lo que realmente tiene.
        json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2}, "keep_alive": KEEP_ALIVE_DEFAULT,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data["response"],
        "sources": sorted({h["file"] for h in hits}),
        "eval_duration_ns": data.get("eval_duration"),
        "total_duration_ns": data.get("total_duration"),
    }
