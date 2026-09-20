# local-docs-copilot

Un copiloto de documentación 100% local (Ollama, sin nube) expuesto como servidor **MCP**,
para que Claude Code (u otro cliente MCP) pueda buscar/resumir la documentación de un proyecto
sin gastar tokens en leer archivos completos — la idea es delegar esa tarea a un modelo chico
corriendo en tu propia GPU, y guardar el presupuesto de contexto/tokens del asistente principal
para el trabajo real.

No mide ni reemplaza la capacidad de generar código: el modelo local es solo un "researcher"
de documentación (RAG), el código lo seguís escribiendo con tu asistente de siempre.

## Cómo se armó

1. **Indexar** tu documentación con embeddings locales (`corpus_index/build_index.py`,
   usa `mxbai-embed-large` vía Ollama).
2. **Elegir un modelo** para responder con RAG: corré un torneo entre los modelos que ya
   tengas en Ollama (`bench/run_bench.py`), midiendo en este orden de importancia:
   1. **Fidelidad** — ¿el resumen inventa algo, o dice "no sé" cuando corresponde?
   2. **Latencia** — ¿es viable en el loop de trabajo diario en tu GPU?
   3. **Contexto** — ¿hasta qué tamaño de texto crudo aguanta sin degradarse?
      (`bench/context_stress.py`, prueba needle-in-haystack sin RAG).
3. **Exponerlo como MCP** (`mcp_server/server.py`) para que cualquier cliente MCP lo use
   como tool.

## Por qué hay un demonio aparte del servidor MCP

Cada Claude CLI (u otro cliente MCP) que uses levanta **su propio proceso** de servidor MCP
(stdio). Si dos agentes consultan al mismo tiempo (por ejemplo un agente "front" y uno "back"
charlando entre sí), dos procesos separados no pueden coordinarse solos para no pisarse en la
GPU — Ollama corre un modelo a la vez. Por eso `mcp_server/daemon.py` es un demonio HTTP
aparte en `localhost` que serializa los pedidos (cola FIFO), y `server.py` es apenas un
cliente delgado de esa cola.

Dos formas de pedir:
- **Bloqueante**: pedís y esperás la respuesta (hasta un timeout razonable) — simple si sos
  el único consumidor.
- **Encolado**: si el demonio ya está ocupado, te devuelve un `job_id` + tu posición en la
  cola en vez de perder el pedido o mezclar respuestas; seguís con `consultar_estado(job_id)`.

⚠️ Si editás `server.py`, cada sesión de Claude Code ya abierta sigue corriendo su proceso
viejo (un stdio subprocess no recarga código solo) — hace falta reiniciar la sesión. Un cambio
en `daemon.py` sí se puede aplicar con `systemctl --user restart doc-copiloto-daemon.service`
sin tocar las sesiones abiertas.

## Lección de retrieval: el embedding puro es frágil a la redacción exacta

Se detectó en uso real que la **misma pregunta reformulada con otras palabras** (mismo tema)
podía hacer caer el chunk correcto muy abajo en el ranking de similitud — el embedding le da
poco peso a nombres propios/siglas técnicas cuando cambia el resto de la frase. Mitigación
barata implementada en `corpus_index/retrieve.py::search()`: un **boost de keywords exactas**
— términos "tipo nombre propio" de la pregunta que aparecen literalmente en un chunk lo
empujan hacia arriba antes de rankear por coseno, proporcional a cuántos términos matchea. No
reemplaza un BM25/keyword search de verdad, pero es gratis y ayuda bastante.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp corpus_index/config.example.yaml corpus_index/config.yaml
# editá corpus_root e include/exclude para apuntar a tu propio proyecto

.venv/bin/python corpus_index/build_index.py   # indexa tu corpus

# demonio de cola (recomendado si vas a usar más de un cliente MCP a la vez)
cp mcp_server/doc-copiloto-daemon.service.example ~/.config/systemd/user/doc-copiloto-daemon.service
# editá /ruta/a/este/repo en ese archivo
systemctl --user daemon-reload
systemctl --user enable --now doc-copiloto-daemon.service

# registrar el MCP en Claude Code (a nivel usuario, disponible en cualquier sesión)
claude mcp add doc-copiloto -s user -- $(pwd)/.venv/bin/python $(pwd)/mcp_server/server.py
```

## Tools que expone el MCP

- `preguntar_docs(pregunta, modelo?, top_k?)` — RAG completo, responde con el LLM local.
- `buscar_docs(pregunta, top_k?)` — solo retrieval, sin resumir (para citar textual).
- `consultar_estado(job_id)` — si tu pedido quedó encolado por estar el demonio ocupado.
- `estado_cola()` — mirar si hay algo corriendo/encolado antes de mandar un pedido pesado.

## Benchmark propio

```bash
cp bench/preguntas.example.yaml bench/preguntas.yaml
# escribí preguntas reales sobre TU corpus, con la fuente exacta de la respuesta correcta

MODELOS="modelo-a,modelo-b" .venv/bin/python bench/run_bench.py
```

Los resultados (`bench/resultados/`) quedan en JSON para revisar fidelidad a mano contra
`preguntas.yaml`. No se versiona ningún resultado ni corpus real en este repo — cada quien
corre el torneo contra su propia documentación.
