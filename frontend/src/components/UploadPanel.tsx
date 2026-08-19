import { type FormEvent, useState } from "react";

import type { ExtractionJob, ExtractionMode } from "../api/extractions";


export interface UploadPanelProps {
  createExtraction: (file: File, mode: ExtractionMode) => Promise<ExtractionJob>;
  onSubmitted: (job: ExtractionJob, file: File) => void;
}

export function UploadPanel({ createExtraction, onSubmitted }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<ExtractionMode>("hybrid");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (file === null) {
      setError("Choose a PNG, JPG, or GIF timing diagram first.");
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const job = await createExtraction(file, mode);
      onSubmitted(job, file);
    } catch (submissionError) {
      setError(messageFor(submissionError));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="panel upload-panel" aria-labelledby="upload-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">New extraction</p>
          <h2 id="upload-heading">Upload a timing diagram</h2>
        </div>
        <p>Images are sent only to the local backend and configured local model.</p>
      </div>

      <form onSubmit={submit}>
        <label className="file-picker" htmlFor="timing-diagram">
          <span>Timing diagram</span>
          <input
            id="timing-diagram"
            name="timing-diagram"
            type="file"
            accept=".png,.jpg,.jpeg,.gif,image/png,image/jpeg,image/gif"
            onChange={(event) => setFile(event.currentTarget.files?.[0] ?? null)}
          />
          <small>{file?.name ?? "PNG, JPG, JPEG, or GIF"}</small>
        </label>

        <fieldset>
          <legend>Extraction mode</legend>
          <label className="mode-option">
            <input
              checked={mode === "hybrid"}
              name="mode"
              type="radio"
              value="hybrid"
              onChange={() => setMode("hybrid")}
            />
            <span>
              <strong>Hybrid</strong>
              <small>OCR + geometry evidence, then local semantic normalization.</small>
            </span>
          </label>
          <label className="mode-option">
            <input
              checked={mode === "vision"}
              name="mode"
              type="radio"
              value="vision"
              onChange={() => setMode("vision")}
            />
            <span>
              <strong>Vision</strong>
              <small>Local vision model extracts the canonical result directly.</small>
            </span>
          </label>
        </fieldset>

        {error !== null && <p className="form-error" role="alert">{error}</p>}

        <button className="primary-button" disabled={isSubmitting} type="submit">
          {isSubmitting ? "Submitting…" : "Extract"}
        </button>
      </form>
    </section>
  );
}

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The extraction could not be submitted.";
}
