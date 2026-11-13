import { useState, type FormEvent } from "react";
import type { PredictionRequest, ProductType } from "../types/api";
import {
  HIGH_RISK_SAMPLE,
  NORMAL_SAMPLE,
  validateSensorForm,
  type SensorFormErrors,
  type SensorFormValues,
} from "../utils/validation";

interface SensorFormProps {
  isLoading: boolean;
  onSubmitValid: (payload: PredictionRequest) => Promise<void>;
  onSendRawPayload?: (rawPayload: Record<string, unknown>) => Promise<void>;
}

export function SensorForm({
  isLoading,
  onSubmitValid,
  onSendRawPayload,
}: SensorFormProps) {
  const [values, setValues] = useState<SensorFormValues>(NORMAL_SAMPLE);
  const [errors, setErrors] = useState<SensorFormErrors>({});

  function handleFieldChange<K extends keyof SensorFormValues>(
    field: K,
    value: SensorFormValues[K]
  ) {
    setValues((prev) => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }
  }

  function applyPreset(preset: SensorFormValues) {
    setValues(preset);
    setErrors({});
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const result = validateSensorForm(values);
    setErrors(result.errors);
    if (!result.isValid || !result.payload) {
      return;
    }
    await onSubmitValid(result.payload);
  }

  async function handleTriggerBackend422() {
    if (!onSendRawPayload) return;
    await onSendRawPayload({
      type: values.type,
      air_temperature_k: -50,
      process_temperature_k: Number(values.process_temperature_k),
      rotational_speed_rpm: Number(values.rotational_speed_rpm),
      torque_nm: Number(values.torque_nm),
      tool_wear_min: Number(values.tool_wear_min),
    });
  }

  return (
    <section className="card" aria-labelledby="sensor-form-title">
      <div className="card-header">
        <h2 id="sensor-form-title">Machine Sensor Readings</h2>
        <div className="preset-actions">
          <button
            type="button"
            className="btn btn-secondary btn-small"
            onClick={() => applyPreset(NORMAL_SAMPLE)}
            disabled={isLoading}
          >
            Load Normal Sample
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-small"
            onClick={() => applyPreset(HIGH_RISK_SAMPLE)}
            disabled={isLoading}
          >
            Load High-Risk Sample
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="type">Product Quality Type</label>
            <select
              id="type"
              value={values.type}
              onChange={(e) => handleFieldChange("type", e.target.value as ProductType)}
              disabled={isLoading}
            >
              <option value="L">L - Low Quality Variant (60% of fleet)</option>
              <option value="M">M - Medium Quality Variant (30% of fleet)</option>
              <option value="H">H - High Quality Variant (10% of fleet)</option>
            </select>
            {errors.type && <span className="field-error">{errors.type}</span>}
          </div>

          <div className="form-field">
            <label htmlFor="air_temperature_k">Air Temperature [K] (250-350)</label>
            <input
              id="air_temperature_k"
              type="number"
              step="0.1"
              value={values.air_temperature_k}
              onChange={(e) => handleFieldChange("air_temperature_k", e.target.value)}
              disabled={isLoading}
              className={errors.air_temperature_k ? "input-invalid" : ""}
            />
            {errors.air_temperature_k && (
              <span className="field-error">{errors.air_temperature_k}</span>
            )}
          </div>

          <div className="form-field">
            <label htmlFor="process_temperature_k">
              Process Temperature [K] (250-400)
            </label>
            <input
              id="process_temperature_k"
              type="number"
              step="0.1"
              value={values.process_temperature_k}
              onChange={(e) => handleFieldChange("process_temperature_k", e.target.value)}
              disabled={isLoading}
              className={errors.process_temperature_k ? "input-invalid" : ""}
            />
            {errors.process_temperature_k && (
              <span className="field-error">{errors.process_temperature_k}</span>
            )}
          </div>

          <div className="form-field">
            <label htmlFor="rotational_speed_rpm">
              Rotational Speed [rpm] (500-5000)
            </label>
            <input
              id="rotational_speed_rpm"
              type="number"
              step="1"
              value={values.rotational_speed_rpm}
              onChange={(e) => handleFieldChange("rotational_speed_rpm", e.target.value)}
              disabled={isLoading}
              className={errors.rotational_speed_rpm ? "input-invalid" : ""}
            />
            {errors.rotational_speed_rpm && (
              <span className="field-error">{errors.rotational_speed_rpm}</span>
            )}
          </div>

          <div className="form-field">
            <label htmlFor="torque_nm">Torque [Nm] (0-150)</label>
            <input
              id="torque_nm"
              type="number"
              step="0.1"
              value={values.torque_nm}
              onChange={(e) => handleFieldChange("torque_nm", e.target.value)}
              disabled={isLoading}
              className={errors.torque_nm ? "input-invalid" : ""}
            />
            {errors.torque_nm && (
              <span className="field-error">{errors.torque_nm}</span>
            )}
          </div>

          <div className="form-field">
            <label htmlFor="tool_wear_min">Tool Wear [min] (0-500)</label>
            <input
              id="tool_wear_min"
              type="number"
              step="1"
              value={values.tool_wear_min}
              onChange={(e) => handleFieldChange("tool_wear_min", e.target.value)}
              disabled={isLoading}
              className={errors.tool_wear_min ? "input-invalid" : ""}
            />
            {errors.tool_wear_min && (
              <span className="field-error">{errors.tool_wear_min}</span>
            )}
          </div>
        </div>

        <div className="submit-row">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? "Running Inference..." : "Predict Failure Risk"}
          </button>

          {onSendRawPayload && (
            <button
              type="button"
              className="btn btn-outline-danger btn-small"
              onClick={handleTriggerBackend422}
              disabled={isLoading}
              title="Bypass frontend range validation to test FastAPI 422 response"
            >
              Test Backend 422 Error
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
