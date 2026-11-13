import type { ApiClientError } from "../api/client";
import { envConfig } from "../config/env";
import type { HealthResponse } from "../types/api";

export type ConnectionState = "checking" | "connected" | "unreachable";

interface StatusBannerProps {
  connectionState: ConnectionState;
  health: HealthResponse | null;
  connectionError: string | null;
  onRetryConnection: () => void;
  apiError: ApiClientError | null;
  onDismissError: () => void;
}

export function StatusBanner({
  connectionState,
  health,
  connectionError,
  onRetryConnection,
  apiError,
  onDismissError,
}: StatusBannerProps) {
  return (
    <div className="status-stack">
      <div
        className={`connection-bar connection-${connectionState}`}
        role="status"
        aria-live="polite"
      >
        <div className="connection-info">
          <span className="status-dot" aria-hidden="true" />
          {connectionState === "checking" && (
            <span>Connecting to backend at <code>{envConfig.apiUrl}</code>...</span>
          )}
          {connectionState === "connected" && health && (
            <span>
              Backend Connected (<code>{envConfig.apiUrl}</code>) - Model{" "}
              <strong>v{health.model_version ?? "unknown"}</strong> Ready
            </span>
          )}
          {connectionState === "unreachable" && (
            <span>
              <strong>Backend Unreachable:</strong>{" "}
              {connectionError ?? `Unable to connect to ${envConfig.apiUrl}`}
            </span>
          )}
        </div>

        <button
          type="button"
          className="btn btn-small btn-secondary"
          onClick={onRetryConnection}
          disabled={connectionState === "checking"}
        >
          {connectionState === "checking" ? "Checking..." : "Refresh Status"}
        </button>
      </div>

      {apiError && (
        <div className="error-banner" role="alert">
          <div className="error-banner-header">
            <strong>
              {apiError.category === "NETWORK_ERROR" && "Connection Error (Backend Unreachable)"}
              {apiError.category === "UNAUTHORIZED" && "Authentication Error (HTTP 401 - Bad API Key)"}
              {apiError.category === "VALIDATION_ERROR" && "Backend Validation Error (HTTP 422 - Invalid Input)"}
              {apiError.category === "HTTP_ERROR" && `Server Error (HTTP ${apiError.statusCode ?? "Unknown"})`}
            </strong>
            <button
              type="button"
              className="btn btn-small btn-ghost"
              onClick={onDismissError}
            >
              Dismiss
            </button>
          </div>

          <p className="error-message">{apiError.message}</p>

          {apiError.details && apiError.details.length > 0 && (
            <ul className="error-detail-list">
              {apiError.details.map((item, idx) => (
                <li key={`${item.field}-${idx}`}>
                  <code>{item.field}</code>: {item.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
