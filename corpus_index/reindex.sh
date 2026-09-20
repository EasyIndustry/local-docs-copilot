#!/usr/bin/env bash
# Reconstruye el índice del benchmark (bench/run_bench.py), fijo al corpus que hayas puesto en
# config.yaml — este script es solo para el benchmark, NO hace falta para el uso en vivo del
# MCP (ese es multi-proyecto y se indexa solo por cwd, ver corpus_index/project_config.py, o
# usá reindexar_proyecto() desde el MCP).
# El índice es una foto fija — no se actualiza solo. Correr a mano después de editar docs, o
# agregar como cron si el corpus cambia seguido:
#   crontab -e
#   0 9 * * * /ruta/a/este/repo/corpus_index/reindex.sh
set -euo pipefail
cd "$(dirname "$0")"
../.venv/bin/python build_index.py --config config.yaml
