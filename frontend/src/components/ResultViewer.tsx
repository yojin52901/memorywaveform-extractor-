import { useEffect, useMemo, useState } from "react";

import { annotationImageUrl, type ExtractionJob } from "../api/extractions";


export interface ResultViewerProps {
  job: ExtractionJob;
  originalImageUrl: string | null;
  selectedParameterId: string | null;
}

type ImageView = "original" | "annotated";

export function ResultViewer({
  job,
  originalImageUrl,
  selectedParameterId,
}: ResultViewerProps) {
  const [imageView, setImageView] = useState<ImageView>("annotated");
  const result = job.result;

  useEffect(() => {
    setImageView("annotated");
  }, [job.id]);

  useEffect(() => {
    if (originalImageUrl === null && imageView === "original") {
      setImageView("annotated");
    }
  }, [imageView, originalImageUrl]);

  const annotatedImage = useMemo(
    () => annotationImageUrl(job.id, selectedParameterId),
    [job.id, selectedParameterId],
  );
  const imageSource = imageView === "original" ? originalImageUrl : annotatedImage;

  if (result === null || result === undefined) {
    return (
      <section className="panel result-panel" aria-labelledby="result-heading">
        <p className="eyebrow">Extraction job</p>
        <h2 id="result-heading">{statusHeading(job.status)}</h2>
        {job.error_message ? <p className="form-error">{job.error_message}</p> : null}
        <p>{job.warnings.join(" ")}</p>
      </section>
    );
  }

  const jsonDownload = `data:application/json;charset=utf-8,${encodeURIComponent(
    JSON.stringify(result, null, 2),
  )}`;

  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Extraction result</p>
          <h2 id="result-heading">{result.document.title}</h2>
        </div>
        <span className={`status-pill status-${job.status}`}>{job.status}</span>
      </div>

      <div className="image-toggle" role="radiogroup" aria-label="Diagram image">
        <label>
          <input
            checked={imageView === "annotated"}
            name="image-view"
            type="radio"
            value="annotated"
            onChange={() => setImageView("annotated")}
          />
          Annotated
        </label>
        <label>
          <input
            checked={imageView === "original"}
            disabled={originalImageUrl === null}
            name="image-view"
            type="radio"
            value="original"
            onChange={() => setImageView("original")}
          />
          Original
        </label>
      </div>

      {imageSource !== null ? (
        <img
          className="result-image"
          src={imageSource}
          alt={imageView === "annotated" ? "Annotated timing diagram" : "Original timing diagram"}
        />
      ) : (
        <p>The original image is available only during this browser session.</p>
      )}

      {selectedParameterId ? (
        <p className="focus-note">
          Highlighting evidence for <code>{selectedParameterId}</code>.
        </p>
      ) : null}

      {result.warnings.length > 0 ? (
        <aside className="warnings" aria-label="Extraction warnings">
          <strong>Review warnings</strong>
          <ul>
            {result.warnings.map((warning) => (
              <li key={`${warning.code}-${warning.message}`}>{warning.message}</li>
            ))}
          </ul>
        </aside>
      ) : null}

      <details>
        <summary>Canonical JSON</summary>
        <a className="json-download" download={`${job.id}.json`} href={jsonDownload}>
          Download JSON
        </a>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </section>
  );
}

function statusHeading(status: ExtractionJob["status"]): string {
  if (status === "failed") {
    return "Extraction failed";
  }
  if (status === "partial") {
    return "Partial extraction ready";
  }
  if (status === "completed") {
    return "Extraction ready";
  }
  return "Extracting timing relationships";
}
