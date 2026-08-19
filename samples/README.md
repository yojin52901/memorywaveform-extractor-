# Local benchmark samples

This public repository intentionally does not include private or third-party
datasheet images. Keep your licensed waveform diagrams in `samples/local/`,
which is ignored by Git, then create a `manifest.json` there from
`manifest.example.json`.

Each expected JSON file must validate against the canonical `ExtractionResult`
contract. `allowed_relation_ids` narrows scoring to the timing-parameter IDs
you have hand-reviewed; leave it empty to score every extracted parameter.

```json
{
  "samples": [
    {
      "id": "write-cycle",
      "image": "write-cycle.png",
      "expected": "write-cycle.expected.json",
      "allowed_relation_ids": ["tp_twc"]
    }
  ]
}
```

Run the benchmark with a configured local model:

```bash
cd backend
uv run memorywaveform-benchmark ../samples/local --mode hybrid \
  --output ../.artifacts/benchmark-report.json
```

The report records the selected mode, configured model ID, signal-label F1,
timing-label F1, relation F1, and the count of relations below the review
confidence threshold.
