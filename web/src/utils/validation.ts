import type { PredictionRequest, ProductType } from "../types/api";

export interface SensorFormValues {
  type: ProductType;
  air_temperature_k: string;
  process_temperature_k: string;
  rotational_speed_rpm: string;
  torque_nm: string;
  tool_wear_min: string;
}

export type SensorFormErrors = Partial<Record<keyof SensorFormValues, string>>;

export const NORMAL_SAMPLE: SensorFormValues = {
  type: "L",
  air_temperature_k: "298.1",
  process_temperature_k: "308.6",
  rotational_speed_rpm: "1551",
  torque_nm: "42.8",
  tool_wear_min: "0",
};

export const HIGH_RISK_SAMPLE: SensorFormValues = {
  type: "L",
  air_temperature_k: "304.0",
  process_temperature_k: "313.2",
  rotational_speed_rpm: "1350",
  torque_nm: "68.5",
  tool_wear_min: "220",
};

export function validateSensorForm(values: SensorFormValues): {
  isValid: boolean;
  errors: SensorFormErrors;
  payload?: PredictionRequest;
} {
  const errors: SensorFormErrors = {};

  if (!["L", "M", "H"].includes(values.type)) {
    errors.type = "Select product type L, M, or H.";
  }

  const airTemp = Number(values.air_temperature_k);
  if (values.air_temperature_k.trim() === "" || Number.isNaN(airTemp) || airTemp < 250 || airTemp > 350) {
    errors.air_temperature_k = "Must be between 250 K and 350 K.";
  }

  const procTemp = Number(values.process_temperature_k);
  if (values.process_temperature_k.trim() === "" || Number.isNaN(procTemp) || procTemp < 250 || procTemp > 400) {
    errors.process_temperature_k = "Must be between 250 K and 400 K.";
  }

  const rpm = Number(values.rotational_speed_rpm);
  if (values.rotational_speed_rpm.trim() === "" || Number.isNaN(rpm) || rpm < 500 || rpm > 5000) {
    errors.rotational_speed_rpm = "Must be between 500 rpm and 5000 rpm.";
  }

  const torque = Number(values.torque_nm);
  if (values.torque_nm.trim() === "" || Number.isNaN(torque) || torque < 0 || torque > 150) {
    errors.torque_nm = "Must be between 0 Nm and 150 Nm.";
  }

  const wear = Number(values.tool_wear_min);
  if (values.tool_wear_min.trim() === "" || Number.isNaN(wear) || wear < 0 || wear > 500) {
    errors.tool_wear_min = "Must be between 0 min and 500 min.";
  }

  if (Object.keys(errors).length > 0) {
    return { isValid: false, errors };
  }

  return {
    isValid: true,
    errors: {},
    payload: {
      type: values.type,
      air_temperature_k: airTemp,
      process_temperature_k: procTemp,
      rotational_speed_rpm: rpm,
      torque_nm: torque,
      tool_wear_min: wear,
    },
  };
}
