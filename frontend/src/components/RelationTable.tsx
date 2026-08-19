import type {
  ExtractionWarning,
  Relation,
  TimingParameter,
} from "../api/extractions";


export interface RelationTableProps {
  parameters: TimingParameter[];
  relations: Relation[];
  warnings?: ExtractionWarning[];
  selectedParameterId?: string | null;
  onSelect: (parameterId: string) => void;
}

export function RelationTable({
  parameters,
  relations,
  warnings = [],
  selectedParameterId,
  onSelect,
}: RelationTableProps) {
  if (parameters.length === 0) {
    return (
      <section className="panel relation-panel" aria-labelledby="relations-heading">
        <h2 id="relations-heading">Timing relationships</h2>
        <p>No grounded timing relationships were found in this diagram.</p>
      </section>
    );
  }

  const reviewOnlyIds = new Set(
    warnings
      .filter((warning) => warning.code === "LOW_CONFIDENCE_RELATION")
      .flatMap((warning) => warning.related_ids),
  );

  return (
    <section className="panel relation-panel" aria-labelledby="relations-heading">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Review</p>
          <h2 id="relations-heading">Timing relationships</h2>
        </div>
        <p>Choose a parameter to emphasize its image evidence.</p>
      </div>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Parameter</th>
              <th scope="col">Events</th>
              <th scope="col">Signals</th>
              <th scope="col">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {parameters.map((parameter) => (
              <tr
                className={rowClassName(parameter.id, selectedParameterId, reviewOnlyIds)}
                key={parameter.id}
              >
                <td>
                  <button
                    aria-pressed={parameter.id === selectedParameterId}
                    className="parameter-button"
                    type="button"
                    onClick={() => onSelect(parameter.id)}
                  >
                    {parameter.name}
                  </button>
                  <small>{parameter.meaning}</small>
                  {reviewOnlyIds.has(parameter.id) ? (
                    <span className="review-badge">Review required</span>
                  ) : null}
                </td>
                <td>
                  <code>{parameter.from_event_id}</code>
                  <span aria-hidden="true"> → </span>
                  <code>{parameter.to_event_id}</code>
                </td>
                <td>{signalSummary(parameter, relations)}</td>
                <td>{Math.round(parameter.confidence * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function rowClassName(
  parameterId: string,
  selectedParameterId: string | null | undefined,
  reviewOnlyIds: Set<string>,
): string | undefined {
  const classes = [
    parameterId === selectedParameterId ? "selected-row" : null,
    reviewOnlyIds.has(parameterId) ? "review-row" : null,
  ].filter((className): className is string => className !== null);
  return classes.length > 0 ? classes.join(" ") : undefined;
}

function signalSummary(parameter: TimingParameter, relations: Relation[]): string {
  const relatedSignalIds = relations
    .filter((relation) => relation.timing_parameter_id === parameter.id)
    .map((relation) => relation.signal_id);
  const signalIds =
    relatedSignalIds.length > 0 ? relatedSignalIds : parameter.participant_signal_ids;
  return signalIds.join(", ");
}
