export type ExtractionMode = "vision" | "hybrid";
export type JobStatus = "queued" | "running" | "completed" | "partial" | "failed";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Signal {
  id: string;
  name: string;
  row: number;
  states: string[];
  confidence?: number | null;
  evidence?: { bbox: BoundingBox } | null;
}

export interface Event {
  id: string;
  signal_id: string;
  type: string;
  x: number;
  from_state?: string | null;
  to_state?: string | null;
  confidence?: number | null;
  evidence?: {
    anchor_x: number;
    bbox?: BoundingBox | null;
  } | null;
}

export interface TimingParameter {
  id: string;
  name: string;
  from_event_id: string;
  to_event_id: string;
  participant_signal_ids: string[];
  meaning: string;
  confidence: number;
  evidence?: {
    label_bbox?: BoundingBox | null;
    arrow_start_x?: number | null;
    arrow_end_x?: number | null;
  } | null;
}

export interface Relation {
  timing_parameter_id: string;
  signal_id: string;
  role: string;
  confidence?: number | null;
  evidence?: TimingParameter["evidence"];
}

export interface ExtractionWarning {
  code: string;
  message: string;
  related_ids: string[];
  evidence?: BoundingBox | null;
}

export interface ExtractionResult {
  document: {
    title: string;
    mode: ExtractionMode;
    image_size: { width: number; height: number };
    source_filename?: string | null;
  };
  signals: Signal[];
  events: Event[];
  timing_parameters: TimingParameter[];
  relations: Relation[];
  warnings: ExtractionWarning[];
  annotated_image?: string | null;
}

export interface ExtractionJob {
  id: string;
  mode: ExtractionMode;
  status: JobStatus;
  input_path?: string;
  source_filename?: string;
  result_path?: string | null;
  annotated_image_path?: string | null;
  warnings: string[];
  error_code?: string | null;
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
  result?: ExtractionResult | null;
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export async function createExtraction(
  file: File,
  mode: ExtractionMode,
): Promise<ExtractionJob> {
  const formData = new FormData();
  formData.set("image", file);
  formData.set("mode", mode);
  const response = await fetch(`${apiBaseUrl}/v1/extractions`, {
    method: "POST",
    body: formData,
  });
  return parseJob(response);
}

export async function getExtraction(id: string): Promise<ExtractionJob> {
  const response = await fetch(`${apiBaseUrl}/v1/extractions/${encodeURIComponent(id)}`);
  return parseJob(response);
}

export function annotationImageUrl(jobId: string, focus?: string | null): string {
  const url = new URL(
    `${apiBaseUrl}/v1/extractions/${encodeURIComponent(jobId)}/artifacts/annotated-image`,
  );
  if (focus) {
    url.searchParams.set("focus", focus);
  }
  return url.toString();
}

async function parseJob(response: Response): Promise<ExtractionJob> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}.`);
  }
  return (await response.json()) as ExtractionJob;
}
