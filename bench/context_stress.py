#!/usr/bin/env python3
"""Prueba de 'contexto soportado': sin RAG, le mete al modelo contexto crudo cada vez más
grande (needle-in-haystack) y mide en qué tamaño empieza a degradarse o a fallar.

No usa el índice ni retrieval — mete documentos completos tal cual. La 'aguja' es un archivo
puntual de tu corpus que contenga la respuesta a PREGUNTA; la 'pila de heno' son los demás
docs, para que sea contexto realista y no relleno random.

Uso:
  NEEDLE_FILE=docs/algo.md PREGUNTA="tu pregunta" CHECK_TERMS="termino1,termino2" \
    .venv/bin/python bench/context_stress.py

Lee corpus_root e include/exclude de corpus_index/config.yaml para armar el haystack.
"""
import glob
import json
import os
import time

import requests
import yaml

HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(HERE, "..", "corpus_index", "config.yaml")

NEEDLE_FILE = os.environ["NEEDLE_FILE"]  # path relativo a corpus_root
PREGUNTA = os.environ["PREGUNTA"]
CHECK_TERMS = [t.strip().lower() for t in os.environ.get("CHECK_TERMS", "").split(",") if t.strip()]

MODELOS = os.environ.get("MODELOS", "qwen3:4b-instruct").split(",")
TARGET_CHARS = [4_000, 16_000, 40_000, 90_000, 160_000]

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
RESULTS_PATH = os.path.join(HERE, "resultados", "context_stress.json")


def load_haystack_docs(cfg, needle_abs):
    root = cfg["corpus_root"]
    excluded = set()
    for pat in cfg.get("exclude", []):
        excluded.update(glob.glob(os.path.join(root, pat), recursive=True))

    paths = []
    for pat in cfg["include"]:
        for p in glob.glob(os.path.join(root, pat), recursive=True):
            if p in excluded or not os.path.isfile(p) or os.path.abspath(p) == needle_abs:
                continue
            paths.append(p)

    texts = []
    for p in sorted(set(paths)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            texts.append(f"\n\n# --- {os.path.relpath(p, root)} ---\n\n" + f.read())
    return texts


def build_context(haystack_docs, needle_text, target_chars):
    context = ""
    for doc in haystack_docs:
        if len(context) >= target_chars:
            break
        context += doc
    context = context[:target_chars]
    mid = len(context) // 2
    return context[:mid] + "\n\n" + needle_text + "\n\n" + context[mid:]


def ask_raw(context, pregunta, model, num_ctx):
    prompt = f"Contexto:\n{context}\n\nPregunta: {pregunta}\n\nRespuesta:"
    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"num_ctx": num_ctx}},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def chars_to_ctx_tokens(n_chars):
    tokens_needed = int(n_chars / 3.2) + 512
    for size in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
        if size >= tokens_needed:
            return size
    return 131072


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    needle_abs = os.path.join(cfg["corpus_root"], NEEDLE_FILE)
    with open(needle_abs, encoding="utf-8") as f:
        needle_text = f.read()

    haystack_docs = load_haystack_docs(cfg, needle_abs)
    print(f"Haystack: {len(haystack_docs)} docs, {sum(len(d) for d in haystack_docs)} chars totales")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    resultados = []

    for modelo in MODELOS:
        print(f"\n=== {modelo} ===")
        for target in TARGET_CHARS:
            context = build_context(haystack_docs, needle_text, target)
            num_ctx = chars_to_ctx_tokens(len(context))
            t0 = time.time()
            try:
                respuesta = ask_raw(context, PREGUNTA, modelo, num_ctx)
                elapsed = time.time() - t0
                low = respuesta.lower()
                acierto = all(term in low for term in CHECK_TERMS) if CHECK_TERMS else None
                estado = "OK" if acierto else ("DEGRADADO" if acierto is False else "SIN-CHECK")
                print(f"  {len(context):>7} chars (num_ctx={num_ctx:>6}) -> {elapsed:5.1f}s "
                      f"{estado}: {respuesta[:100]!r}")
                resultados.append({
                    "modelo": modelo, "context_chars": len(context), "num_ctx": num_ctx,
                    "latencia_s": round(elapsed, 1), "acierto": acierto, "respuesta": respuesta,
                })
            except Exception as e:
                print(f"  {len(context):>7} chars (num_ctx={num_ctx:>6}) -> ERROR tras {time.time()-t0:.1f}s: {e}")
                resultados.append({"modelo": modelo, "context_chars": len(context), "num_ctx": num_ctx, "error": str(e)})

    with open(RESULTS_PATH, "w") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
