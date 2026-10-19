# Embedded AI Copilot

This is an AI assistant for understanding embedded C/C++ codebases. You send a question to a small FastAPI app. The app reads the fixed sample project, puts those source files into one prompt, and asks Gemini for an answer.

## Status

Phase 1.5 complete.

## Request path

```mermaid
flowchart LR
    client --> main["main (FastAPI)"]
    main --> ai_service
    ai_service --> config
    ai_service --> project_loader
    ai_service --> prompts
    ai_service --> Gemini
```

## Modules

- `app/main.py` is the HTTP layer. It serves `GET /` and `POST /ask`, checks the question, and maps service errors to status codes. It does not read the project or call Gemini.
- `app/ai_service.py` orchestrates one answer: load files, build the prompt, call Gemini, and raise `ProjectLoadError`, `UpstreamRejectedError`, `EmptyUpstreamResponseError` (subclass of `UpstreamRejectedError`, maps to 502), or `UpstreamUnavailableError`. It does not define routes or scan the disk itself.
- `app/project_loader.py` reads the filesystem. It does not build prompts or call Gemini.
- `app/prompts.py` builds the prompt from `app/prompts/assistant.md`, the file text, and the question. It does not load the tree or call Gemini.
- `app/config.py` loads `.env` and checks `GEMINI_API_KEY`, `GEMINI_MODEL`, and that `sample_project/` exists. It returns a frozen `Settings` dataclass. It does not call Gemini.
- `app/models.py` holds the `ProjectFile` dataclass. The HTTP body is `AskRequest` in `main.py`.
- `app/logging_config.py` sets up logging. It does not choose the log messages.

## Setup

Python 3.14 (>=3.14,<3.15). From the repo root:

```bash
uv python install 3.14
uv python pin 3.14
uv sync
```

Pin CPython 3.14. A newer interpreter can fail to build the locked `pydantic-core`.

Create `.env` in the repo root, or export the same variables:

```
GEMINI_API_KEY=your-key
GEMINI_MODEL=your-model
```

Startup fails if either value is missing or empty (`RuntimeError: GEMINI_API_KEY is not set.` or `RuntimeError: GEMINI_MODEL is not set.`).

The running app reads the fixed `sample_project/` directory. It loads only `.c`, `.h`, `.cpp`, and `.hpp` files, including files in subdirectories. A non-UTF-8 file is skipped with a warning (`Skipping non-UTF-8 file: ...`).

## Run

From the repo root:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Logs go to `logs/app.log` and to the console. The `logs/` directory is created when the app starts.

## API

`GET /`

```bash
curl http://127.0.0.1:8000/
```

```json
{"message":"Embedded AI Copilot is running"}
```

`POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is the answer initialized?"}'
```

Request body: `{"question":"<non-empty string>"}`.

Response body: `{"answer":"<text>"}`.

## Errors

| Status | When | Body |
| --- | --- | --- |
| 200 | Gemini returned text | `{"answer":"<text>"}` |
| 422 | Invalid, empty, or whitespace-only `question` | validation `detail` list, example below |
| 500 | Project files could not be loaded | `{"detail":"Project files could not be loaded."}` |
| 500 | Unexpected error | `Internal Server Error` |
| 502 | Gemini rejected the request, or returned empty text | `{"detail":"AI provider rejected the request."}` |
| 503 | Gemini is unavailable | `{"detail":"AI provider is unavailable."}` |

Empty question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":""}'
```

```json
{"detail":[{"type":"string_too_short","loc":["body","question"],"msg":"String should have at least 1 character","input":"","ctx":{"min_length":1}}]}
```

A missing `question` or a non-string `question` also returns 422, with a different `detail` list. An unexpected exception returns the plain text `Internal Server Error`.

## Tests

```bash
uv run pytest
```

The tests run offline. Gemini is mocked.

## Design

Each module has one job. Domain data is a dataclass (`ProjectFile`, `Settings`). Pydantic validates the HTTP request (`AskRequest`). The loader, the prompt builder, and the Gemini client are called directly.

There is no RAG, agents, vector database, LangChain, or tree-sitter. The sample project is sent in one prompt. Those tools get added when this design hits a real limit.

## Roadmap

1. Phase 1: simple assistant (done)
2. Phase 1.5: engineering quality (done)
3. Phase 2: smarter project loading
4. Phase 3: tree-sitter / code structure
5. Phase 4: RAG / embeddings / ChromaDB
6. Phase 5: agentic AI

## Layout

```
.
├── app/
│   ├── __init__.py
│   ├── ai_service.py
│   ├── config.py
│   ├── logging_config.py
│   ├── main.py
│   ├── models.py
│   ├── project_loader.py
│   ├── prompts.py
│   └── prompts/
│       └── assistant.md
├── sample_project/
│   ├── inc/
│   │   ├── app.h
│   │   ├── can.h
│   │   └── gpio.h
│   └── src/
│       ├── app.c
│       ├── can.c
│       ├── gpio.c
│       └── main.c
├── tests/
│   ├── conftest.py
│   ├── test_ai_service.py
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_project_loader.py
│   └── test_prompts.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```
