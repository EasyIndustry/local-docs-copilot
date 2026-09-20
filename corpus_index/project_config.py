#!/usr/bin/env python3
"""Resuelve qué corpus/índice corresponde a CADA proyecto, para que el MCP sirva de uso
global en la máquina (no un único corpus fijo hardcodeado a un proyecto en particular).

Cada proyecto puede tener un `.doc-copiloto.yaml` en su raíz con include/exclude/chunk_size;
si no existe, se usan defaults razonables (indexar todo el markdown del proyecto). El índice
resultante se guarda aparte, en un caché compartido por fuera del proyecto, con un nombre
derivado de la ruta absoluta del proyecto — así dos proyectos distintos nunca se pisan.
"""
import hashlib
import os

import yaml

CACHE_DIR = os.path.expanduser("~/.cache/doc-copiloto/indices")

DEFAULTS = {
    "include": ["**/*.md", "**/*.mdx"],
    "exclude": [
        "**/node_modules/**", "**/.git/**", "**/.venv/**", "**/venv/**",
        "**/dist/**", "**/build/**", "**/.next/**", "**/.claude/**",
    ],
    "embed_model": "mxbai-embed-large",
    "chunk_size_chars": 1000,
    "chunk_overlap_chars": 100,
}


def _slug(proyecto_abs):
    h = hashlib.sha1(proyecto_abs.encode()).hexdigest()[:12]
    nombre = os.path.basename(proyecto_abs.rstrip("/")) or "raiz"
    return f"{nombre}-{h}"


def resolve(proyecto):
    """proyecto: ruta a la raíz de un proyecto (típicamente el cwd de la sesión que consulta).
    Devuelve un dict de config completo, listo para build_index.py/retrieve.py."""
    proyecto_abs = os.path.abspath(proyecto)
    cfg = dict(DEFAULTS)
    cfg["corpus_root"] = proyecto_abs

    override_path = os.path.join(proyecto_abs, ".doc-copiloto.yaml")
    if os.path.exists(override_path):
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        cfg.update(override)
        cfg["corpus_root"] = proyecto_abs  # el override no puede cambiar la raíz

    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg["index_path"] = os.path.join(CACHE_DIR, f"{_slug(proyecto_abs)}.json")
    return cfg


def index_existe(proyecto):
    return os.path.exists(resolve(proyecto)["index_path"])
