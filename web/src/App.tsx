import { useCallback, useEffect, useState } from "react";
import {
  ApiClientError,
  fetchHealth,
  fetchModelInfo,
  predictFailure,
} from "./api/client";
import { ResultView } from "./components/ResultView";
import { SensorForm } from "./components/SensorForm";
import { StatusBanner, type ConnectionState } from "./components/StatusBanner";
import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "./types/api";

export function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [apiError, setApiError] = useState<ApiClientError | null>(null);

  const [simulateBadKey, setSimulateBadKey] = useState<boolean>(false);

  const checkBackendConnection = useCallback(async () => {
    setConnectionState("checking");
    setConnectionError(null);
    try {
      const [healthData, infoData] = await Promise.all([
        fetchHealth(),
        fetchModelInfo(),
      ]);
      setHealth(healthData);
      setModelInfo(infoData);
      setConnectionState("connected");
    } catch (err) {
      setHealth(null);
      setModelInfo(null);
      setConnectionState("unreachable");
      if (err instanceof ApiClientError) {
        setConnectionError(err.message);
      } else {
        setConnectionError("Unexpected error checking backend status.");
      }
    }
  }, []);

  useEffect(() => {
    void checkBackendConnection();
  }, [checkBackendConnection]);

  async function handlePredictValid(payload: PredictionRequest) {
    setIsSubmitting(true);
    setApiError(null);
    try {
      const response = await predictFailure(
        payload,
        simulateBadKey ? "invalid-demo-api-key" : undefined
      );
      setPredictionResult(response);
    } catch (err) {
      if (err instanceof ApiClientError) {
        setApiError(err);
      } else {
        setApiError(
          new ApiClientError("NETWORK_ERROR", "Unexpected error sending prediction request.")
        );
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSendRawPayload(rawPayload: Record<string, unknown>) {
    setIsSubmitting(true);
    setApiError(null);
    try {
      const response = await predictFailure(
        rawPayload,
        simulateBadKey ? "invalid-demo-api-key" : undefined
      );
      setPredictionResult(response);
    } catch (err) {
      if (err instanceof ApiClientError) {
        setApiError(err);
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>CloudGuard AI</h1>
          <p className="subtitle">
            Predictive Maintenance Telemetry Classifier (UCI AI4I 2020 Synthetic Dataset)
          </p>
        </div>
        <span className="edu-pill">Educational Demo - Not for Industrial Use</span>
      </header>

      <StatusBanner
        connectionState={connectionState}
        health={health}
        connectionError={connectionError}
        onRetryConnection={() => void checkBackendConnection()}
        apiError={apiError}
        onDismissError={() => setApiError(null)}
      />

      <main className="dashboard-grid">
        <SensorForm
          isLoading={isSubmitting}
          onSubmitValid={handlePredictValid}
          onSendRawPayload={handleSendRawPayload}
        />

        <div className="right-column">
          <ResultView result={predictionResult} />

          <section className="card" aria-labelledby="model-overview-title">
            <div className="card-header">
              <h2 id="model-overview-title">Model &amp; Security Diagnostics</h2>
            </div>

            <div className="diagnostic-box">
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={simulateBadKey}
                  onChange={(e) => setSimulateBadKey(e.target.checked)}
                />
                <span>Simulate Invalid API Key (Test HTTP 401 Error)</span>
              </label>

              <p className="muted-text">
                <strong>Browser Key Notice:</strong> <code>VITE_API_KEY</code> is loaded from{" "}
                <code>web/.env</code>, so no secret is hardcoded in source files - however, any{" "}
                <code>VITE_*</code> variable is bundled into browser JS and visible in DevTools.
              </p>

              {modelInfo && (
                <ul className="model-meta-list">
                  <li>
                    <strong>Algorithm:</strong> {modelInfo.algorithm} (v{modelInfo.model_version})
                  </li>
                  <li>
                    <strong>Test PR-AUC / Recall:</strong>{" "}
                    {modelInfo.test_metrics.pr_auc ?? "N/A"} /{" "}
                    {modelInfo.test_metrics.recall ?? "N/A"}
                  </li>
                  <li>
                    <strong>Synthetic Dataset:</strong>{" "}
                    {modelInfo.is_synthetic_data ? "Yes (UCI AI4I 2020)" : "No"}
                  </li>
                </ul>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
