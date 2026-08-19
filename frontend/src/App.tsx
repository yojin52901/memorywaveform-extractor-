import { useEffect, useRef, useState } from "react";

import {
  createExtraction,
  getExtraction,
  type ExtractionJob,
  type JobStatus,
} from "./api/extractions";
import { RelationTable } from "./components/RelationTable";
import { ResultViewer } from "./components/ResultViewer";
import { UploadPanel } from "./components/UploadPanel";


const terminalStatuses: JobStatus[] = ["completed", "partial", "failed"];

export function App() {
  const [job, setJob] = useState<ExtractionJob | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [selectedParameterId, setSelectedParameterId] = useState<string | null>(null);
  const [pollingError, setPollingError] = useState<string | null>(null);
  const originalImageUrlRef = useRef<string | null>(null);
  const jobId = job?.id;
  const jobStatus = job?.status;

  useEffect(() => {
    return () => {
      if (originalImageUrlRef.current !== null) {
        URL.revokeObjectURL(originalImageUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (
      jobId === undefined ||
      jobStatus === undefined ||
      terminalStatuses.includes(jobStatus)
    ) {
      return undefined;
    }

    let cancelled = false;
    const intervalId = window.setInterval(() => {
      void getExtraction(jobId)
        .then((updatedJob) => {
          if (!cancelled) {
            setJob(updatedJob);
            setPollingError(null);
          }
        })
        .catch((error: unknown) => {
          if (!cancelled) {
            setPollingError(errorMessage(error));
          }
        });
    }, 1_000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [jobId, jobStatus]);

  function handleSubmitted(submittedJob: ExtractionJob, file: File) {
    if (originalImageUrlRef.current !== null) {
      URL.revokeObjectURL(originalImageUrlRef.current);
    }
    const imageUrl = URL.createObjectURL(file);
    originalImageUrlRef.current = imageUrl;
    setOriginalImageUrl(imageUrl);
    setJob(submittedJob);
    setSelectedParameterId(null);
    setPollingError(null);
  }

  return (
    <main className="app-shell">
      <header className="app-header" aria-labelledby="application-title">
        <p className="eyebrow">Local-first timing analysis</p>
        <h1 id="application-title">Memory Waveform Extractor</h1>
        <p>
          Turn a memory-datasheet waveform into grounded signals, events, and timing
          relationships without sending the image to a cloud model.
        </p>
      </header>

      <UploadPanel createExtraction={createExtraction} onSubmitted={handleSubmitted} />

      {job !== null ? (
        <section className="job-area" aria-live="polite">
          <div className="job-summary">
            <span className={`status-pill status-${job.status}`}>{job.status}</span>
            <span>Job {job.id}</span>
          </div>
          {pollingError ? <p className="form-error">{pollingError}</p> : null}
          <ResultViewer
            job={job}
            originalImageUrl={originalImageUrl}
            selectedParameterId={selectedParameterId}
          />
          {job.result ? (
            <RelationTable
              parameters={job.result.timing_parameters}
              relations={job.result.relations}
              warnings={job.result.warnings}
              selectedParameterId={selectedParameterId}
              onSelect={setSelectedParameterId}
            />
          ) : null}
        </section>
      ) : null}
    </main>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not refresh the extraction job.";
}
