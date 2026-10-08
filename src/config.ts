export type InferenceMode = 'python' | 'browser';

/**
 * Read Vite env values through a captured object so both inference modes stay
 * in the production bundle even when no .env file is present.
 */
const env = import.meta.env as unknown as Record<string, string | undefined>;

/** Base URL of the FastAPI backend (Python mode). */
export const API_URL: string = (env['VITE_API_URL'] ?? 'http://127.0.0.1:8000').replace(/\/+$/, '');

/** Which engine the frontend uses: "python" (HTTP API) or "browser" (bundled model). */
export const INFERENCE_MODE: InferenceMode = env['VITE_INFERENCE_MODE'] === 'browser' ? 'browser' : 'python';

export const IS_BROWSER_MODE = INFERENCE_MODE === 'browser';

export const TEXT_TIMEOUT_MS = 30_000;
export const IMAGE_TIMEOUT_MS = 300_000;
