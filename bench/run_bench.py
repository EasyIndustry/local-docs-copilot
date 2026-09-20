#!/usr/bin/env python3
"""Corre el set de preguntas contra cada modelo candidato, mide latencia y guarda
resultados para revisar fidelidad a mano contra bench/preguntas.yaml.

Uso: .venv/bin/python bench/run_bench.py
"""
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus_index"))
from retrieve import ask  # noqa: E402

HERE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(HERE, "..", "corpus_index", "index.json")
RESULTS_DIR = os.path.join(HERE, "resultados")

# Modelos candidatos a comparar (separados por coma), ej:
#   MODELOS="qwen3:4b-instruct,gemma3:12b" .venv/bin/python bench/run_bench.py
MODELOS = os.environ.get("MODELOS", "qwen3:4b-instruct").split(",")


def main():
    with open(os.path.join(HERE, "preguntas.yaml")) as f:
        preguntas = yaml.safe_load(f)

    if not os.path.exists(INDEX_PATH):
        sys.exit(f"No existe {INDEX_PATH}. Corré primero: corpus_index/build_index.py")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for modelo in MODELOS:
        print(f"\n=== {modelo} ===")
        resultados = []
        for p in preguntas:
            t0 = time.time()
            try:
                r = ask(p["pregunta"], INDEX_PATH, model=modelo)
                elapsed = time.time() - t0
                print(f"  [{p['id']}] {elapsed:.1f}s -> {r['answer'][:80]}...")
                resultados.append({
                    "id": p["id"],
                    "pregunta": p["pregunta"],
                    "fuente_esperada": p["fuente"],
                    "respuesta": r["answer"],
                    "fuentes_citadas": r["sources"],
                    "latencia_s": round(elapsed, 2),
                })
            except Exception as e:
                print(f"  [{p['id']}] ERROR: {e}")
                resultados.append({"id": p["id"], "error": str(e)})

        out_path = os.path.join(RESULTS_DIR, f"{modelo.replace(':', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"  guardado en {out_path}")

    print(f"\nListo. Revisá {RESULTS_DIR}/ a mano contra bench/preguntas.yaml para el scoring de fidelidad.")


if __name__ == "__main__":
    main()
