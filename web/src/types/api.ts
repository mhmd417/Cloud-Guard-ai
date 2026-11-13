export type MachineType = "L" | "M" | "H";

export interface PredictionRequest {
  type: MachineType;
  air_temperature_k: number;
  process_temperature_k: number;
  rotational_speed_rpm: number;
  torque_nm: number;
  tool_wear_min: number;
}

export interface PredictionResponse {
  failure_class: 0 | 1;
  failure_probability: number;
  threshold: number;
  model_version: string;
  disclaimer: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string | null;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  algorithm: string;
  is_synthetic_data: boolean;
  features: string[];
  decision_threshold: number;
  test_metrics: {
    pr_auc?: number;
    roc_auc?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
  };
  disclaimer: string;
}

export interface ApiErrorDetailItem {
  field: string;
  message: string;
  type: string;
}

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
    details: ApiErrorDetailItem[] | null;
  };
}
