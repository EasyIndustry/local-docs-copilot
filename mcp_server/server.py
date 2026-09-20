#!/usr/bin/env python3
"""Servidor MCP: expone búsqueda/resumen de documentación vía el modelo local, sin gastar
tokens de Claude. Es un cliente delgado del demonio (daemon.py), que es quien de verdad
serializa el acceso a Ollama cuando hay más de un Claude CLI consultando a la vez.

Correr el demonio primero: .venv/bin/python mcp_server/daemon.py
Después registrar este server.py como servidor MCP stdio en cada Claude CLI.
"""
import os

import requests
from mcp.server.fastmcp import FastMCP

DAEMON_URL = os.environ.get("DOC_COPILOTO_DAEMON_URL", "http://127.0.0.1:8799")
DEFAULT_MODEL = os.environ.get("DOC_COPILOTO_MODEL", "qwen3:4b-instruct")

mcp = FastMCP("doc-copiloto")


def _submit(tool, args, wait_s):
    resp = requests.post(f"{DAEMON_URL}/job", json={"tool": tool, "args": args, "wait_s": wait_s}, timeout=wait_s + 10)
    return resp.json()


@mcp.tool()
def buscar_docs(pregunta: str, top_k: int = 10) -> str:
    """Busca los fragmentos de documentación más relevantes para una pregunta, sin resumir
    (para citar exacto). Encolado: si el demonio está ocupado con otro pedido (p.ej. de otro
    Claude CLI consultando en paralelo), devuelve un job_id para seguir con consultar_estado.
    """
    data = _submit("buscar_docs", {"pregunta": pregunta, "top_k": top_k}, wait_s=30)
    return _format(data)


@mcp.tool()
def preguntar_docs(pregunta: str, modelo: str = DEFAULT_MODEL, top_k: int = 10) -> str:
    """Responde una pregunta sobre la documentación indexada con RAG usando el LLM local
    (100% local, no consume tokens de esta sesión). Si el demonio ya está atendiendo a otro
    Claude CLI, espera hasta 90s y si no alcanza devuelve un job_id: seguí con
    consultar_estado(job_id) en vez de reintentar la pregunta desde cero.
    """
    data = _submit("preguntar_docs", {"pregunta": pregunta, "modelo": modelo, "top_k": top_k}, wait_s=90)
    return _format(data)


@mcp.tool()
def consultar_estado(job_id: str) -> str:
    """Consulta si un pedido encolado (job_id devuelto por preguntar_docs/buscar_docs cuando
    el demonio estaba ocupado) ya terminó. Si sigue en cola o corriendo, decilo y sugerí
    reintentar en unos segundos en vez de asumir que falló."""
    resp = requests.get(f"{DAEMON_URL}/job/{job_id}", timeout=10)
    job = resp.json()
    if job.get("status") == "done":
        return job["result"]
    if job.get("status") == "error":
        return f"El pedido falló: {job['error']}"
    return f"Todavía {job.get('status', 'desconocido')} (job_id={job_id}). Reintentá en unos segundos."


@mcp.tool()
def estado_cola() -> str:
    """Mirá qué está procesando el copiloto local y cuántos pedidos hay encolados, ANTES de
    mandar una pregunta pesada — útil para decidir si conviene esperar o hacer otra cosa
    mientras tanto (p.ej. si hay otro Claude CLI usándolo en paralelo)."""
    resp = requests.get(f"{DAEMON_URL}/estado", timeout=10)
    data = resp.json()
    if data["procesando"]:
        return f"Ocupado: procesando '{data['procesando']}' (job_id={data['job_id_actual']}), {data['en_cola']} en cola."
    return "Libre, no hay nada corriendo ni en cola."


def _format(data):
    if data.get("status") == "done":
        return data["result"]
    if data.get("status") == "error":
        return f"Error: {data['error']}"
    return (
        f"Ocupado ahora mismo (probablemente otro Claude CLI consultando el copiloto local). "
        f"Tu pedido quedó encolado en la posición {data.get('position', '?')} "
        f"(job_id={data['job_id']}). Usá consultar_estado('{data['job_id']}') en unos segundos, "
        f"o hacé otra cosa mientras tanto y volvé después."
    )


if __name__ == "__main__":
    mcp.run()
