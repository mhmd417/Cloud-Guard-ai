export interface AppEnvConfig {
  apiUrl: string;
  apiKey: string;
}

function getRequiredEnv(name: "VITE_API_URL" | "VITE_API_KEY"): string {
  const raw = import.meta.env[name];
  if (!raw || typeof raw !== "string" || raw.trim() === "") {
    throw new Error(
      `Missing required environment variable ${name}. Check web/.env (see web/.env.example).`
    );
  }
  return raw.trim();
}

export const envConfig: AppEnvConfig = {
  // Remove any trailing slash so `${apiUrl}/predict` never becomes `//predict`
  apiUrl: getRequiredEnv("VITE_API_URL").replace(/\/+$/, ""),
  apiKey: getRequiredEnv("VITE_API_KEY"),
};
