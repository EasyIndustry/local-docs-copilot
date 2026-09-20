#!/usr/bin/env bash
# Reconstruye el índice RAG cuando cambió la documentación fuente.
# El índice (index.json) es una foto fija — no se actualiza solo.
# Correr a mano después de editar docs, o agregar como cron si el corpus cambia seguido:
#   crontab -e
#   0 9 * * * /ruta/a/este/repo/corpus_index/reindex.sh
set -euo pipefail
cd "$(dirname "$0")"
../.venv/bin/python build_index.py
