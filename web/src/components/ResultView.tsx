import type { PredictionResponse } from "../types/api";

interface ResultViewProps {
  result: PredictionResponse | null;
}

export function ResultView({ result }: ResultViewProps) {
  if (!result) {
    return (
      <section className="card result-card result-empty" aria-labelledby="result-title">
        <h2 id="result-title">Prediction Result</h2>
        <p className="muted-text">
          Submit machine sensor readings on the left to view the predicted failure class,
          failure probability, and active model version.
        </p>
      </section>
    );
  }

  const isFailure = result.failure_class === 1;
  const probabilityPercent = (result.failure_probability * 100).toFixed(2);
  const thresholdPercent = (result.threshold * 100).toFixed(0);

  return (
    <section
      className={`card result-card ${isFailure ? "result-danger" : "result-safe"}`}
      aria-labelledby="result-title"
      aria-live="polite"
    >
      <div className="card-header">
        <h2 id="result-title">Prediction Result</h2>
        <span className={`badge ${isFailure ? "badge-danger" : "badge-safe"}`}>
          {isFailure ? "Class 1 - Failure Risk" : "Class 0 - Normal"}
        </span>
      </div>

      <div className="metrics-row">
        <div className="metric-box">
          <span className="metric-label">Predicted Class</span>
          <strong className="metric-value">
            {result.failure_class} ({isFailure ? "Failure" : "No Failure"})
          </strong>
        </div>

        <div className="metric-box">
          <span className="metric-label">Failure Probability</span>
          <strong className="metric-value">{probabilityPercent}%</strong>
        </div>

        <div className="metric-box">
          <span className="metric-label">Decision Threshold</span>
          <strong className="metric-value">{thresholdPercent}%</strong>
        </div>

        <div className="metric-box">
          <span className="metric-label">Model Version</span>
          <strong className="metric-value">v{result.model_version}</strong>
        </div>
      </div>

      <div className="probability-meter" aria-hidden="true">
        <div
          className={`probability-fill ${isFailure ? "fill-danger" : "fill-safe"}`}
          style={{ width: `${Math.min(100, Math.max(2, result.failure_probability * 100))}%` }}
        />
      </div>

      <p className="disclaimer-note">{result.disclaimer}</p>
    </section>
  );
}
