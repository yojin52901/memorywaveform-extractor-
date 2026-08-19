# Memory Waveform Extractor

Memory Waveform Extractor is a local-first tool for turning clean memory
datasheet timing diagrams into an auditable relationship graph. It will expose
the same canonical JSON result through a Python package, CLI, FastAPI service,
and React review interface.

## MVP scope

- Accept clear English PNG, JPG, and GIF timing diagrams.
- Support `vision` and grounded `hybrid` extraction modes; `hybrid` is the
  default.
- Return signal rows, event anchors, timing parameters, relations, confidence,
  warnings, and an annotated image.
- Keep images, artifacts, and model execution local by default.

## Repository layout

- `backend/`: extraction domain, API, CLI, artifacts, local model providers,
  and tests.
- `frontend/`: React review interface.
- `docs/`: approved design and implementation plan.

## Quick start with Docker

The backend connects only to a model server you run locally. Start Ollama (or
your MLX/vLLM-compatible server) on the host first, then configure its model:

```bash
cp .env.example .env
# Edit .env and replace VISION_MODEL with an installed local vision model.
docker compose up --build
```

Open the review UI at <http://localhost:5173>. The API is available at
<http://localhost:8000>; generated input images, JSON results, annotations,
and SQLite job metadata stay in `./artifacts/`.

For Ollama, keep `VISION_PROVIDER=ollama` and set `VISION_MODEL`. In Docker,
the default `OLLAMA_BASE_URL=http://host.docker.internal:11434` reaches an
Ollama instance on the host. For a local MLX or vLLM server, set
`VISION_PROVIDER=openai-compatible`, `OPENAI_COMPATIBLE_BASE_URL`, and
`OPENAI_COMPATIBLE_MODEL` instead. Neither option requires a cloud model API.

`hybrid` mode additionally needs Tesseract. The backend Docker image includes
it; for native development install it with `brew install tesseract` on macOS
or `sudo apt install tesseract-ocr` on Debian/Ubuntu.

Relations below `RELATION_CONFIDENCE_THRESHOLD` (default `0.7`) remain in the
result with a structured review warning and a `partial` job status. The current
hybrid baseline emits a multi-signal relation only when the detected arrow row
maps uniquely to an OCR signal row; otherwise it returns an explicit unresolved
warning rather than assigning the arrow to an arbitrary signal row.

## Native development

The repository targets Python 3.11+ and Node.js 20+.

```bash
cd backend && uv sync --group dev
cd ../frontend && npm install
```

In separate terminals, start the backend and UI:

```bash
cd backend
VISION_PROVIDER=ollama VISION_MODEL=your-local-vision-model \
  uv run uvicorn memorywaveform_extractor.api.app:create_app --factory --reload
```

```bash
cd frontend
npm run dev
```

## Interfaces

The web UI submits the same job lifecycle used by the CLI and Python API.

```bash
cd backend
VISION_PROVIDER=ollama VISION_MODEL=your-local-vision-model \
  uv run memorywaveform-extract extract ./wave.png --mode hybrid --output .artifacts
```

```python
from pathlib import Path

from memorywaveform_extractor import ExtractionMode, extract_file

result = extract_file(Path("wave.png"), ExtractionMode.HYBRID)
print(result.model_dump_json(indent=2))
```

HTTP routes:

- `POST /v1/extractions` — multipart fields `image` and optional `mode`.
- `GET /v1/extractions/{id}` — job status, warnings, and canonical result when ready.
- `GET /v1/extractions/{id}/artifacts/annotated-image` — the auditable PNG;
  add `?focus=<timing-parameter-id>` to emphasize one timing span.

## Benchmarking

Place only images you are allowed to use in `samples/local/`; it is ignored by
Git. Copy the example manifest and expected result schema from `samples/`, then
run:

```bash
cd backend
uv run memorywaveform-benchmark ../samples/local --mode hybrid \
  --output ../.artifacts/benchmark-report.json
```

The report records signal-label, timing-label, and relation F1 plus the count
of low-confidence timing relations. Benchmark tests use fakes and do not need
a GPU, downloaded model, or Tesseract binary.

## Verification

```bash
cd backend && uv run pytest -q && uv run ruff check . && uv run mypy src
cd ../frontend && npm test -- --run && npm run lint && npm run typecheck
```
