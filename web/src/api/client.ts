import { envConfig } from "../config/env";
import type {
  ApiErrorDetailItem,
  ApiErrorPayload,
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/api";

export type ApiErrorCategory =
  | "NETWORK_ERROR"
  | "UNAUTHORIZED"
  | "VALIDATION_ERROR"
  | "HTTP_ERROR";

export class ApiClientError extends Error {
  public readonly category: ApiErrorCategory;
  public readonly statusCode: number | null;
  public readonly details: ApiErrorDetailItem[] | null;

  constructor(
    category: ApiErrorCategory,
    message: string,
    statusCode: number | null = null,
    details: ApiErrorDetailItem[] | null = null
  ) {
    super(message);
    this.name = "ApiClientError";
    this.category = category;
    this.statusCode = statusCode;
    this.details = details;
  }
}

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs = 8000
): Promise<T> {
  const controller = new AbortController();
  const timerId = window.setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${envConfig.apiUrl}${path}`, {
      ...options,
      signal: controller.signal,
    });
  } catch (err) {
    window.clearTimeout(timerId);
    const isTimeout = err instanceof DOMException && err.name === "AbortError";
    throw new ApiClientError(
      "NETWORK_ERROR",
      isTimeout
        ? `Request to ${envConfig.apiUrl}${path} timed out after ${timeoutMs / 1000}s.`
        : `Cannot reach backend at ${envConfig.apiUrl}. Verify the FastAPI server/container is running and CORS allows this origin.`
    );
  } finally {
    window.clearTimeout(timerId);
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const errorEnvelope = body as ApiErrorPayload | null;
    const serverMessage = errorEnvelope?.error?.message;
    const serverDetails = errorEnvelope?.error?.details ?? null;

    if (response.status === 401) {
      throw new ApiClientError(
        "UNAUTHORIZED",
        serverMessage || "401 Unauthorized: Missing or invalid X-API-Key header.",
        401,
        serverDetails
      );
    }

    if (response.status === 422) {
      throw new ApiClientError(
        "VALIDATION_ERROR",
        serverMessage || "422 Unprocessable Entity: Sensor payload failed backend validation.",
        422,
        serverDetails
      );
    }

    throw new ApiClientError(
      "HTTP_ERROR",
      serverMessage || `Backend returned HTTP ${response.status}.`,
      response.status,
      serverDetails
    );
  }

  return body as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health", { method: "GET" });
}

export function fetchModelInfo(): Promise<ModelInfoResponse> {
  return requestJson<ModelInfoResponse>("/model-info", { method: "GET" });
}

export function predictFailure(
  payload: PredictionRequest | Record<string, unknown>,
  apiKeyOverride?: string
): Promise<PredictionResponse> {
  const activeKey = apiKeyOverride !== undefined ? apiKeyOverride : envConfig.apiKey;
  return requestJson<PredictionResponse>("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": activeKey,
    },
    body: JSON.stringify(payload),
  });
}
