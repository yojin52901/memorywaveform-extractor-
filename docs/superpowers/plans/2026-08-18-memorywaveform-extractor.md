# Memory Waveform Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first timing-diagram extractor that returns signal-to-timing-parameter relationships as canonical JSON and an annotated image through web, CLI, and Python interfaces.

**Architecture:** A FastAPI backend owns one canonical extraction contract, job lifecycle, local artifacts, and selectable `vision` and `hybrid` strategies. A React frontend submits jobs and renders the same contract. Local model and OCR capabilities are accessed only through provider interfaces.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, Pillow, OpenCV, pytesseract, Typer, pytest, React, TypeScript, Vite, Vitest, Docker Compose, Ollama, MLX/vLLM-compatible local endpoints.

**Spec:** `docs/superpowers/specs/2026-08-18-memorywaveform-extractor-design.md`

## Global Constraints

- Python version is 3.11 or later.
- TypeScript runs in strict mode.
- No application path requires a cloud model API.
- The default extraction mode is `hybrid`.
- Inputs and artifacts remain under a configurable local artifact root.
- Models and credentials are configured only through environment variables.
- Every parser result uses the canonical Pydantic result model.
- Animated GIF input uses only frame zero and creates a warning.
- Tests run without a GPU, a model download, or an OCR binary by using fakes.

---

## Planned File Structure

| Path | Responsibility |
|---|---|
| `backend/src/memorywaveform_extractor/domain/models.py` | Canonical Pydantic models and enums. |
| `backend/src/memorywaveform_extractor/application/extract.py` | One extraction use case and strategy selection. |
| `backend/src/memorywaveform_extractor/application/jobs.py` | Job lifecycle orchestration. |
| `backend/src/memorywaveform_extractor/domain/ports.py` | Protocols for OCR, vision, jobs, and artifacts. |
| `backend/src/memorywaveform_extractor/infrastructure/images.py` | Input decoding and GIF-frame handling. |
| `backend/src/memorywaveform_extractor/infrastructure/artifacts.py` | Local filesystem artifact store. |
| `backend/src/memorywaveform_extractor/infrastructure/sqlite_jobs.py` | SQLite job repository. |
| `backend/src/memorywaveform_extractor/providers/ollama.py` | Local Ollama vision adapter. |
| `backend/src/memorywaveform_extractor/providers/openai_compatible.py` | MLX/vLLM local HTTP adapter. |
| `backend/src/memorywaveform_extractor/providers/tesseract.py` | OCR adapter. |
| `backend/src/memorywaveform_extractor/strategies/vision.py` | Schema-grounded vision-only extraction. |
| `backend/src/memorywaveform_extractor/strategies/hybrid.py` | OCR, geometry, and semantic-normalization extraction. |
| `backend/src/memorywaveform_extractor/strategies/geometry.py` | Line, arrow, and anchor geometry. |
| `backend/src/memorywaveform_extractor/annotation/render.py` | Annotated PNG rendering. |
| `backend/src/memorywaveform_extractor/api/app.py` | FastAPI application and routes. |
| `backend/src/memorywaveform_extractor/cli.py` | CLI entry point. |
| `frontend/src/*` | React upload, job polling, results, and API client. |
| `backend/tests/*` | Python unit, API, and golden tests. |
| `frontend/src/*.test.tsx` | Frontend component tests. |
| `samples/*` | Copyright-safe local test fixtures and hand-reviewed expected JSON. |

### Task 1: Bootstrap the monorepo and canonical result contract

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/memorywaveform_extractor/domain/models.py`
- Create: `backend/tests/domain/test_models.py`
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `.gitignore`
- Create: `README.md`

**Interfaces:**
- Produces `ExtractionMode`, `JobStatus`, `BoundingBox`, `Signal`, `Event`,
  `TimingParameter`, `Relation`, `Warning`, and `ExtractionResult`.
- `ExtractionResult` is the only result type returned by strategies, CLI, and
  HTTP routes.

- [ ] **Step 1: Write the failing canonical-model tests**

```python
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult


def test_result_rejects_timing_parameter_without_two_event_ids() -> None:
    payload = {
        "document": {"title": "Write timing", "mode": "hybrid", "image_size": [10, 10]},
        "signals": [],
        "events": [],
        "timing_parameters": [{"id": "tp_twc", "name": "tWC", "from_event_id": "a"}],
        "relations": [],
        "warnings": [],
    }

    assert ExtractionResult.model_validate(payload)  # replaced by a validation error assertion
```

- [ ] **Step 2: Run the test and verify model imports fail before implementation**

Run: `cd backend && python -m pytest tests/domain/test_models.py -v`

Expected: FAIL because `memorywaveform_extractor.domain.models` does not exist.

- [ ] **Step 3: Implement the Pydantic contract and correct the failing assertion**

```python
class ExtractionMode(str, Enum):
    VISION = "vision"
    HYBRID = "hybrid"


class TimingParameter(BaseModel):
    id: str
    name: str
    from_event_id: str
    to_event_id: str
    participant_signal_ids: list[str]
    meaning: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: TimingEvidence
```

Use `pytest.raises(ValidationError)` in the test and add one valid-result
fixture that verifies JSON serialization.

- [ ] **Step 4: Run focused model tests**

Run: `cd backend && python -m pytest tests/domain/test_models.py -v`

Expected: PASS with validation and serialization coverage.

- [ ] **Step 5: Initialize Python and TypeScript quality commands**

Add `pytest`, `ruff`, and `mypy` scripts for the backend plus `test`, `lint`,
and `typecheck` scripts for the frontend. Keep the starter React app to a
single root component that renders `Memory Waveform Extractor`.

- [ ] **Step 6: Run baseline quality commands**

Run: `cd backend && python -m pytest -q && ruff check . && mypy src`

Run: `cd frontend && npm test -- --run && npm run lint && npm run typecheck`

Expected: all commands pass.

- [ ] **Step 7: Commit the bootstrap**

```bash
git add .gitignore README.md backend frontend
git commit -m "chore: bootstrap waveform extractor monorepo"
```

### Task 2: Add local inputs, artifacts, and the job lifecycle

**Files:**
- Create: `backend/src/memorywaveform_extractor/domain/ports.py`
- Create: `backend/src/memorywaveform_extractor/infrastructure/images.py`
- Create: `backend/src/memorywaveform_extractor/infrastructure/artifacts.py`
- Create: `backend/src/memorywaveform_extractor/infrastructure/sqlite_jobs.py`
- Create: `backend/src/memorywaveform_extractor/application/jobs.py`
- Create: `backend/tests/infrastructure/test_images.py`
- Create: `backend/tests/application/test_jobs.py`

**Interfaces:**
- `decode_image(source: bytes, filename: str) -> DecodedImage`
- `ArtifactStore.save_input(job_id: UUID, image: DecodedImage) -> Path`
- `JobRepository.create(mode: ExtractionMode, input_path: Path) -> ExtractionJob`
- `JobCoordinator.run(job_id: UUID) -> ExtractionJob`

- [ ] **Step 1: Write failing image-decoding tests**

```python
def test_decode_gif_uses_first_frame_and_warns() -> None:
    decoded = decode_image(animated_gif_bytes(), "wave.gif")

    assert decoded.format == "GIF"
    assert decoded.frame_index == 0
    assert decoded.warnings == ["Animated GIF detected; only frame 0 was analyzed."]
```

- [ ] **Step 2: Run image tests before implementation**

Run: `cd backend && python -m pytest tests/infrastructure/test_images.py -v`

Expected: FAIL because `decode_image` is not implemented.

- [ ] **Step 3: Implement image decoding and local artifact persistence**

```python
def decode_image(source: bytes, filename: str) -> DecodedImage:
    with Image.open(BytesIO(source)) as image:
        is_animated = bool(getattr(image, "is_animated", False))
        image.seek(0)
        raster = image.convert("RGB").copy()
    return DecodedImage(
        raster=raster,
        format=Path(filename).suffix.lstrip(".").upper(),
        frame_index=0,
        warnings=["Animated GIF detected; only frame 0 was analyzed."] if is_animated else [],
    )
```

Create an artifact directory named after the UUID and store `input.png`,
`result.json`, and `annotated.png` there.

- [ ] **Step 4: Write failing job-lifecycle tests**

```python
def test_job_transitions_from_queued_to_completed(tmp_path: Path) -> None:
    job = coordinator.submit(image_bytes(), "wave.png", ExtractionMode.HYBRID)
    completed = coordinator.run(job.id)

    assert completed.status is JobStatus.COMPLETED
    assert completed.result_path is not None
```

- [ ] **Step 5: Implement SQLite metadata and a synchronous local coordinator**

The coordinator persists `queued`, changes the row to `running`, invokes an
injected extraction service, then persists either `completed`, `partial`, or
`failed`. Keep work synchronous inside the coordinator; FastAPI runs it in a
background task in Task 5.

- [ ] **Step 6: Run the Task 2 tests and static checks**

Run: `cd backend && python -m pytest tests/infrastructure/test_images.py tests/application/test_jobs.py -v && ruff check . && mypy src`

Expected: PASS.

- [ ] **Step 7: Commit local lifecycle support**

```bash
git add backend
git commit -m "feat: add local extraction job lifecycle"
```

### Task 3: Implement local vision providers and vision mode

**Files:**
- Create: `backend/src/memorywaveform_extractor/providers/ollama.py`
- Create: `backend/src/memorywaveform_extractor/providers/openai_compatible.py`
- Create: `backend/src/memorywaveform_extractor/providers/fakes.py`
- Create: `backend/src/memorywaveform_extractor/strategies/vision.py`
- Create: `backend/src/memorywaveform_extractor/application/extract.py`
- Create: `backend/tests/strategies/test_vision.py`
- Create: `backend/tests/providers/test_openai_compatible.py`

**Interfaces:**
- `VisionProvider.extract(image: bytes, schema: dict[str, object]) -> dict[str, object]`
- `ExtractionService.extract(image: DecodedImage, mode: ExtractionMode) -> ExtractionResult`
- `VisionStrategy.extract(image: DecodedImage) -> ExtractionResult`

- [ ] **Step 1: Write the failing vision-strategy test using a fake provider**

```python
def test_vision_strategy_validates_provider_payload(sample_image: DecodedImage) -> None:
    provider = FakeVisionProvider(payload=valid_result_payload(mode="vision"))
    result = VisionStrategy(provider).extract(sample_image)

    assert result.document.mode is ExtractionMode.VISION
    assert result.timing_parameters[0].name == "tWC"
```

- [ ] **Step 2: Run the test before implementation**

Run: `cd backend && python -m pytest tests/strategies/test_vision.py -v`

Expected: FAIL because `VisionStrategy` is not implemented.

- [ ] **Step 3: Implement schema-grounded local provider calls**

```python
class VisionProvider(Protocol):
    def extract(self, image: bytes, schema: dict[str, object]) -> dict[str, object]: ...


class VisionStrategy:
    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    def extract(self, image: DecodedImage) -> ExtractionResult:
        payload = self._provider.extract(image.png_bytes, ExtractionResult.model_json_schema())
        payload["document"]["mode"] = ExtractionMode.VISION.value
        return ExtractionResult.model_validate(payload)
```

The Ollama adapter sends the image as base64 to the configured local endpoint.
The OpenAI-compatible adapter sends the image as a data URL to the configured
local endpoint. Both adapters read endpoint and model IDs from settings.

- [ ] **Step 4: Add provider HTTP contract tests**

Use `respx` or `httpx.MockTransport` to assert that Ollama receives
`model`, `messages`, and `images`, and the OpenAI-compatible endpoint receives
an image-url content part. Do not contact a real model server.

- [ ] **Step 5: Run Task 3 tests and quality checks**

Run: `cd backend && python -m pytest tests/strategies/test_vision.py tests/providers/test_openai_compatible.py -v && ruff check . && mypy src`

Expected: PASS.

- [ ] **Step 6: Commit vision mode**

```bash
git add backend
git commit -m "feat: add local vision extraction mode"
```

### Task 4: Implement OCR, geometry, and hybrid mode

**Files:**
- Create: `backend/src/memorywaveform_extractor/providers/tesseract.py`
- Create: `backend/src/memorywaveform_extractor/strategies/geometry.py`
- Create: `backend/src/memorywaveform_extractor/strategies/hybrid.py`
- Create: `backend/tests/strategies/test_geometry.py`
- Create: `backend/tests/strategies/test_hybrid.py`

**Interfaces:**
- `OcrProvider.read(image: Image.Image) -> list[OcrToken]`
- `detect_vertical_anchors(image: Image.Image) -> list[int]`
- `detect_timing_arrows(image: Image.Image, tokens: list[OcrToken]) -> list[ArrowEvidence]`
- `HybridStrategy.extract(image: DecodedImage) -> ExtractionResult`

- [ ] **Step 1: Write a failing arrow-to-anchor test**

```python
def test_arrow_endpoints_snap_to_nearest_vertical_anchors() -> None:
    arrow = ArrowEvidence(label="tWC", start_x=101, end_x=301, label_bbox=BoundingBox(180, 5, 220, 25))

    relation = snap_arrow_to_anchors(arrow, anchors=[100, 300])

    assert relation.start_anchor_x == 100
    assert relation.end_anchor_x == 300
```

- [ ] **Step 2: Run geometry tests before implementation**

Run: `cd backend && python -m pytest tests/strategies/test_geometry.py -v`

Expected: FAIL because geometry functions do not exist.

- [ ] **Step 3: Implement deterministic evidence extraction**

Use grayscale conversion, adaptive thresholding, Canny edges, and Hough line
detection. Treat vertical lines longer than the configured minimum height as
event anchors. Match horizontal arrow spans to the nearest anchors. Normalize
OCR text by stripping whitespace and preserving alphanumeric characters plus
timing symbols.

```python
def snap_arrow_to_anchors(arrow: ArrowEvidence, anchors: list[int]) -> SnappedArrow:
    start = min(anchors, key=lambda x: abs(x - arrow.start_x))
    end = min(anchors, key=lambda x: abs(x - arrow.end_x))
    return SnappedArrow(label=arrow.label, start_anchor_x=start, end_anchor_x=end, label_bbox=arrow.label_bbox)
```

- [ ] **Step 4: Write a failing hybrid-strategy test using fakes**

```python
def test_hybrid_result_contains_geometry_evidence(sample_image: DecodedImage) -> None:
    result = HybridStrategy(fake_ocr, fake_vision, fake_geometry).extract(sample_image)

    assert result.document.mode is ExtractionMode.HYBRID
    assert result.timing_parameters[0].evidence.arrow_start_x == 100
    assert result.timing_parameters[0].evidence.arrow_end_x == 300
```

- [ ] **Step 5: Implement hybrid semantic normalization**

Build candidate signals from left-side OCR labels, events from anchors and
state regions, and parameters from `t`-prefixed OCR labels. Pass this grounded
candidate graph to the vision provider. Merge only provider relations whose
IDs exist in the candidate graph; otherwise add an `UNGROUNDED_RELATION`
warning and exclude them.

- [ ] **Step 6: Run Task 4 tests and quality checks**

Run: `cd backend && python -m pytest tests/strategies/test_geometry.py tests/strategies/test_hybrid.py -v && ruff check . && mypy src`

Expected: PASS.

- [ ] **Step 7: Commit hybrid mode**

```bash
git add backend
git commit -m "feat: add grounded hybrid extraction mode"
```

### Task 5: Render annotations and expose the backend, CLI, and Python API

**Files:**
- Create: `backend/src/memorywaveform_extractor/annotation/render.py`
- Create: `backend/src/memorywaveform_extractor/api/app.py`
- Create: `backend/src/memorywaveform_extractor/api/dependencies.py`
- Create: `backend/src/memorywaveform_extractor/cli.py`
- Create: `backend/tests/annotation/test_render.py`
- Create: `backend/tests/api/test_extractions.py`
- Create: `backend/tests/test_cli.py`

**Interfaces:**
- `render_annotation(image: Image.Image, result: ExtractionResult) -> bytes`
- `POST /v1/extractions` accepts multipart `image` and `mode`.
- `GET /v1/extractions/{job_id}` returns `ExtractionJob`.
- `GET /v1/extractions/{job_id}/artifacts/annotated-image` returns `image/png`.
- `memorywaveform-extract extract INPUT --mode hybrid --output OUTPUT_DIR`

- [ ] **Step 1: Write failing annotation-render tests**

```python
def test_renderer_draws_parameter_label_box(sample_image: Image.Image, result: ExtractionResult) -> None:
    png = render_annotation(sample_image, result)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
```

- [ ] **Step 2: Implement deterministic overlay rendering**

Draw signal-row boxes in blue, event anchors in orange, timing-arrow spans in
green, and warning evidence in red. Include a legend only when at least one
warning is present. Render the result to PNG regardless of input format.

- [ ] **Step 3: Write failing API lifecycle tests**

```python
def test_upload_then_read_completed_job(client: TestClient, sample_png: bytes) -> None:
    created = client.post("/v1/extractions", files={"image": ("wave.png", sample_png, "image/png")}, data={"mode": "hybrid"})
    job_id = created.json()["id"]

    response = client.get(f"/v1/extractions/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] in {"queued", "running", "completed", "partial"}
```

- [ ] **Step 4: Implement FastAPI routes, CLI, and public package export**

The POST route persists the input and schedules `JobCoordinator.run` in a
FastAPI background task. The CLI uses the same service factory and writes the
same artifact names. Export `extract_file(path: Path, mode: ExtractionMode,
settings: Settings) -> ExtractionResult` from the package root.

- [ ] **Step 5: Run Task 5 tests and quality checks**

Run: `cd backend && python -m pytest tests/annotation/test_render.py tests/api/test_extractions.py tests/test_cli.py -v && ruff check . && mypy src`

Expected: PASS.

- [ ] **Step 6: Commit backend interfaces**

```bash
git add backend
git commit -m "feat: add extraction API cli and annotations"
```

### Task 6: Build the React upload and result-review workflow

**Files:**
- Create: `frontend/src/api/extractions.ts`
- Create: `frontend/src/components/UploadPanel.tsx`
- Create: `frontend/src/components/ResultViewer.tsx`
- Create: `frontend/src/components/RelationTable.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/components/UploadPanel.test.tsx`
- Create: `frontend/src/components/RelationTable.test.tsx`

**Interfaces:**
- `createExtraction(file: File, mode: "vision" | "hybrid"): Promise<Job>`
- `getExtraction(id: string): Promise<Job>`
- `UploadPanel` emits a selected file and mode.
- `RelationTable` emits a selected timing-parameter ID to `ResultViewer`.

- [ ] **Step 1: Write a failing upload-mode test**

```tsx
it("submits the selected hybrid mode", async () => {
  const createExtraction = vi.fn().mockResolvedValue({ id: "job-1", status: "queued" });
  render(<UploadPanel createExtraction={createExtraction} />);

  await userEvent.click(screen.getByLabelText("Hybrid"));
  await userEvent.upload(screen.getByLabelText("Timing diagram"), new File(["image"], "wave.png", { type: "image/png" }));
  await userEvent.click(screen.getByRole("button", { name: "Extract" }));

  expect(createExtraction).toHaveBeenCalledWith(expect.any(File), "hybrid");
});
```

- [ ] **Step 2: Run frontend test before implementation**

Run: `cd frontend && npm test -- --run src/components/UploadPanel.test.tsx`

Expected: FAIL because `UploadPanel` does not exist.

- [ ] **Step 3: Implement upload, polling, and result presentation**

The app polls every second only while a job is `queued` or `running`. It stops
polling at `completed`, `partial`, or `failed`. The result viewer switches
between original and annotated images. Selecting a relation sends its evidence
ID to the image layer and visually emphasizes the matching overlay.

- [ ] **Step 4: Write and implement relation-table test coverage**

```tsx
it("reports the selected timing parameter", async () => {
  const onSelect = vi.fn();
  render(<RelationTable parameters={[parameter("tp_twc", "tWC")]} onSelect={onSelect} />);

  await userEvent.click(screen.getByRole("button", { name: "tWC" }));
  expect(onSelect).toHaveBeenCalledWith("tp_twc");
});
```

- [ ] **Step 5: Run frontend quality checks**

Run: `cd frontend && npm test -- --run && npm run lint && npm run typecheck`

Expected: PASS.

- [ ] **Step 6: Commit the UI**

```bash
git add frontend
git commit -m "feat: add waveform extraction review UI"
```

### Task 7: Add golden samples, benchmark reporting, Docker, and documentation

**Files:**
- Create: `backend/src/memorywaveform_extractor/benchmark.py`
- Create: `backend/tests/golden/test_samples.py`
- Create: `samples/README.md`
- Create: `docker-compose.yml`
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`
- Modify: `README.md`
- Create: `.env.example`

**Interfaces:**
- `run_benchmark(samples_root: Path, mode: ExtractionMode) -> BenchmarkReport`
- `BenchmarkReport` includes signal-label F1, timing-label F1, relation F1,
  and low-confidence relation count.
- `docker compose up --build` starts frontend and backend while local model
  configuration remains host-controlled.

- [ ] **Step 1: Write a failing benchmark-metric test**

```python
def test_relation_f1_counts_matching_parameter_and_event_pair() -> None:
    report = score_relations(
        predicted={("tWC", "evt_1", "evt_2")},
        expected={("tWC", "evt_1", "evt_2"), ("tAW", "evt_3", "evt_4")},
    )

    assert report.precision == 1.0
    assert report.recall == 0.5
    assert round(report.f1, 2) == 0.67
```

- [ ] **Step 2: Run the test before implementation**

Run: `cd backend && python -m pytest tests/golden/test_samples.py -v`

Expected: FAIL because benchmark scoring is not implemented.

- [ ] **Step 3: Implement benchmark scoring and sample manifest handling**

Use one JSON expected-result file per sample. The sample manifest names the
image, expected JSON, and allowed relation IDs. The benchmark command records
both mode, configured model ID, and score values in one report JSON file.

- [ ] **Step 4: Add container definitions and user documentation**

Document these exact setup paths:

```bash
cp .env.example .env
docker compose up --build
```

Explain how to start Ollama locally, set `VISION_PROVIDER=ollama`, select
`VISION_MODEL`, or point `OPENAI_COMPATIBLE_BASE_URL` to a local MLX/vLLM
server. Include the Tesseract installation requirement for `hybrid` mode.

- [ ] **Step 5: Run the full backend and frontend suite**

Run: `cd backend && python -m pytest -q && ruff check . && mypy src`

Run: `cd frontend && npm test -- --run && npm run lint && npm run typecheck`

Expected: all checks pass without a real model server or GPU.

- [ ] **Step 6: Commit delivery material**

```bash
git add README.md .env.example docker-compose.yml backend frontend samples
git commit -m "docs: add local deployment and benchmark workflow"
```

## Plan Self-Review

- Spec coverage: Tasks 1-7 cover the canonical contract, local provider
  abstraction, image-only inputs, both modes, evidence, artifacts, web/API/
  CLI/Python interfaces, local deployment, golden samples, and benchmark
  reporting.
- Placeholder scan: this plan contains no future-work placeholders; all
  interfaces, test commands, and commit boundaries are specified.
- Type consistency: every strategy returns `ExtractionResult`; every interface
  uses `ExtractionMode`; every job path is mediated by `JobCoordinator`.
