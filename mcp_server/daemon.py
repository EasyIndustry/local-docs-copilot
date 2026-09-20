#!/usr/bin/env python3
"""Demonio local de cola para el copiloto de docs.

Por qué existe: cada Claude CLI que usa el MCP levanta SU PROPIO proceso de servidor MCP
(stdio). Si dos agentes (p. ej. front y back charlando entre sí) consultan al mismo tiempo,
dos procesos de servidor MCP separados no comparten memoria — un lock adentro de server.py
no alcanza para serializar el acceso a Ollama (una sola GPU, un modelo a la vez).

Por eso la cola vive acá, en un demonio HTTP aparte en localhost que ambos servidores MCP
consultan. Un solo worker la procesa FIFO contra Ollama.

Dos formas de pedir:
- POST /job {tool, args, wait_s?} -> si termina dentro de wait_s (default 60s), devuelve el
  resultado ya resuelto (para el caso de un solo agente: pedís y esperás, sin poll manual).
- Si no termina a tiempo (la GPU está ocupada con otro pedido), devuelve
  {status: "queued", job_id, position} para que el que llamó decida: seguir esperando
  (consultar_estado) o hacer otra cosa mientras tanto.
- GET /job/<id> -> estado actual / resultado si ya está.
- GET /estado -> qué hay en cola y qué se está procesando (para decidir sin ni siquiera
  encolar un pedido).

Correr: .venv/bin/python mcp_server/daemon.py
"""
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus_index"))
import build_index  # noqa: E402
import project_config  # noqa: E402
from retrieve import ask, precalentar, search  # noqa: E402

OLLAMA_PS_URL = "http://localhost:11434/api/ps"
HOST, PORT = "127.0.0.1", 8799


def _index_path_para(proyecto):
    """Resuelve el índice del proyecto pedido, indexándolo la primera vez si no existe
    todavía (así el MCP funciona de entrada en cualquier proyecto de la máquina, no solo
    en el que se haya indexado manualmente una vez)."""
    cfg = project_config.resolve(proyecto)
    if not os.path.exists(cfg["index_path"]):
        build_index.build(cfg, log=lambda *_: None)
    return cfg["index_path"]

_queue = queue.Queue()
_jobs = {}  # job_id -> {"status": ..., "result": ..., "error": ..., "tool": ..., "submitted_at": ...}
_lock = threading.Lock()
_current_job_id = None


def _worker():
    global _current_job_id
    while True:
        job_id = _queue.get()
        with _lock:
            _jobs[job_id]["status"] = "running"
            _current_job_id = job_id
        job = _jobs[job_id]
        try:
            proyecto = job["args"].get("proyecto") or os.getcwd()
            if job["tool"] == "buscar_docs":
                index_path = _index_path_para(proyecto)
                hits = search(job["args"]["pregunta"], index_path, top_k=job["args"].get("top_k", 10))
                result = "\n\n".join(f"[{h['file']}] (score={h['score']:.3f})\n{h['text']}" for h in hits)
            elif job["tool"] == "preguntar_docs":
                index_path = _index_path_para(proyecto)
                r = ask(
                    job["args"]["pregunta"], index_path,
                    model=job["args"].get("modelo", "qwen3:4b-instruct"),
                    top_k=job["args"].get("top_k", 10),
                )
                result = f"{r['answer']}\n\nFuentes: {', '.join(r['sources'])}"
            elif job["tool"] == "reindexar":
                cfg = project_config.resolve(proyecto)
                n = build_index.build(cfg, log=lambda *_: None)
                result = f"Reindexado {proyecto}: {n} chunks en {cfg['index_path']}."
            elif job["tool"] == "precalentar":
                modelo = job["args"].get("modelo", "qwen3:4b-instruct")
                precalentar(modelo)
                result = f"{modelo} cargado en VRAM."
            else:
                raise ValueError(f"tool desconocida: {job['tool']}")
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
        except Exception as e:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)
        finally:
            with _lock:
                _current_job_id = None


def _submit(tool, args):
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        position = _queue.qsize() + (1 if _current_job_id else 0)
        _jobs[job_id] = {
            "status": "queued", "tool": tool, "args": args,
            "submitted_at": time.time(), "position_al_entrar": position,
        }
    _queue.put(job_id)
    return job_id, position


def _wait_for(job_id, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with _lock:
            job = dict(_jobs[job_id])
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.3)
    with _lock:
        return dict(_jobs[job_id])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silencioso; el server MCP ya loguea lo que le importa a Claude

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/estado":
            try:
                ps = requests.get(OLLAMA_PS_URL, timeout=5).json().get("models", [])
                modelos_cargados = [m["name"] for m in ps]
            except Exception:
                modelos_cargados = None  # Ollama no responde; no bloquear /estado por eso
            with _lock:
                self._send(200, {
                    "procesando": _jobs.get(_current_job_id, {}).get("tool") if _current_job_id else None,
                    "job_id_actual": _current_job_id,
                    "en_cola": _queue.qsize(),
                    "modelos_cargados": modelos_cargados,
                })
            return
        if self.path.startswith("/job/"):
            job_id = self.path.split("/job/", 1)[1]
            with _lock:
                job = _jobs.get(job_id)
            if job is None:
                self._send(404, {"error": "job_id no existe"})
            else:
                self._send(200, {"job_id": job_id, **job, "args": None})
            return
        self._send(404, {"error": "ruta no encontrada"})

    def do_POST(self):
        if self.path != "/job":
            self._send(404, {"error": "ruta no encontrada"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        tool = body.get("tool")
        args = body.get("args", {})
        wait_s = body.get("wait_s", 60)

        job_id, position = _submit(tool, args)
        job = _wait_for(job_id, wait_s)

        if job["status"] == "done":
            self._send(200, {"status": "done", "job_id": job_id, "result": job["result"]})
        elif job["status"] == "error":
            self._send(200, {"status": "error", "job_id": job_id, "error": job["error"]})
        else:
            self._send(202, {
                "status": "queued", "job_id": job_id, "position": job.get("position_al_entrar", position),
                "hint": f"Ocupado. Consultá GET /job/{job_id} para el resultado, o /estado para ver la cola.",
            })


def main():
    threading.Thread(target=_worker, daemon=True).start()
    print(f"Demonio doc-copiloto escuchando en http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
