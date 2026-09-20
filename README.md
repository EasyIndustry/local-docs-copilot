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

## Arranque en frío por diseño (no ocupar VRAM sin usarla)

Decisión explícita: el copiloto **no** mantiene el modelo cargado en VRAM todo el tiempo — la
GPU la necesitás para otras cosas (dev, juegos, lo que sea), y pagar un arranque en frío
(10-60s según el modelo y tu hardware) es aceptable si el agente que lo pide lo sabe de
antemano. `corpus_index/retrieve.py` manda `keep_alive: "2m"` en cada llamado a Ollama — a los
2 minutos de inactividad, libera la VRAM solo (el default de muchos servidores Ollama es
mantenerlo cargado indefinidamente, lo cual desperdicia recursos si no lo estás usando).

Para no pagar el arranque en frío justo cuando hace falta la respuesta:
1. Llamá `precalentar_modelo()` apenas empieza la tarea (no bloquea, dispara la carga).
2. Seguí con otra cosa mientras tanto.
3. Llamá `estado_cola()` para confirmar que el modelo ya figura en `modelos_cargados`, o
   directamente `preguntar_docs(...)` (que espera lo que falte de carga si no terminó).

## Instalación (una sola vez, sirve para todos tus proyectos)

Registrás el MCP **a nivel usuario**, no por proyecto — se instala una vez y queda disponible
en cualquier sesión de Claude Code en la máquina, sin tener que apuntarlo a un corpus fijo.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# demonio de cola (recomendado si vas a usar más de un cliente MCP a la vez)
cp mcp_server/doc-copiloto-daemon.service.example ~/.config/systemd/user/doc-copiloto-daemon.service
# editá /ruta/a/este/repo en ese archivo
systemctl --user daemon-reload
systemctl --user enable --now doc-copiloto-daemon.service

# registrar el MCP en Claude Code (a nivel usuario, disponible en cualquier sesión/proyecto)
claude mcp add doc-copiloto -s user -- $(pwd)/.venv/bin/python $(pwd)/mcp_server/server.py
```

Listo — desde cualquier proyecto, `preguntar_docs(...)` indexa ese proyecto automáticamente
la primera vez que lo consultás (todo el markdown, excluyendo `node_modules/`, `.git/`, etc.),
guardando su índice aparte en `~/.cache/doc-copiloto/indices/`. Cada proyecto queda con su
propio índice — no hay un corpus fijo compartido entre todos.

Si los defaults de indexado no te sirven para un proyecto puntual (traen basura, o falta algo
fuera de markdown), creá un `.doc-copiloto.yaml` en su raíz — mismas claves que
`corpus_index/config.example.yaml`, sin `corpus_root` (se infiere solo). Si editaste
documentación y el índice quedó viejo, llamá `reindexar_proyecto()`.

## Tools que expone el MCP

- `preguntar_docs(pregunta, modelo?, top_k?, proyecto?)` — RAG completo con el LLM local, en
  el proyecto actual (auto-detectado; pasá `proyecto` con una ruta absoluta si hace falta
  apuntar a otro).
- `buscar_docs(pregunta, top_k?, proyecto?)` — solo retrieval, sin resumir (para citar textual).
- `reindexar_proyecto(proyecto?)` — reconstruye el índice de un proyecto que ya tenía uno.
- `precalentar_modelo(modelo?)` — carga el modelo a VRAM sin preguntarle nada todavía.
- `consultar_estado(job_id)` — si tu pedido quedó encolado por estar el demonio ocupado.
- `estado_cola()` — mirar si hay algo corriendo/encolado y qué está cargado en VRAM ahora.

## Benchmark propio

Para comparar modelos con un corpus fijo y reproducible (no el modo multi-proyecto del MCP en
uso real), armá un `config.yaml` explícito:

```bash
cp corpus_index/config.example.yaml corpus_index/config.yaml
# editá corpus_root e include/exclude para apuntar al corpus que quieras usar de referencia
.venv/bin/python corpus_index/build_index.py --config corpus_index/config.yaml

cp bench/preguntas.example.yaml bench/preguntas.yaml
# escribí preguntas reales sobre TU corpus, con la fuente exacta de la respuesta correcta

MODELOS="modelo-a,modelo-b" .venv/bin/python bench/run_bench.py
```

Los resultados (`bench/resultados/`) quedan en JSON para revisar fidelidad a mano contra
`preguntas.yaml`. No se versiona ningún resultado ni corpus real en este repo — cada quien
corre el torneo contra su propia documentación.
