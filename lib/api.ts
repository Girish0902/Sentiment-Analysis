/**
 * API client and inference-mode dispatch.
 *
 * In "python" mode every request goes to the FastAPI backend. In "browser"
 * mode the same exported model runs locally through lib/sentiment.ts.
 */

import {
  analyzeItems,
  evaluateItems,
  InputError,
  MAX_CSV_BYTES,
  type AnalysisResult,
  type EvaluationMetrics,
  type ModelArtifact,
  type TextItem,
} from './sentiment';
import {
  API_URL,
  IMAGE_TIMEOUT_MS,
  INFERENCE_MODE,
  IS_BROWSER_MODE,
  TEXT_TIMEOUT_MS,
} from '../src/config';

export interface ImageMatch {
  label: string;
  score: number;
}

export interface ImageAnalysisResult {
  label: 'positive' | 'neutral' | 'negative';
  unclear: boolean;
  probabilities: Record<string, number>;
  matches: ImageMatch[];
  model: string;
  engine: 'python' | 'browser';
}

let modelPromise: Promise<ModelArtifact> | null = null;

export function loadModel(): Promise<ModelArtifact> {
  if (!modelPromise) {
    modelPromise = fetch(`${import.meta.env.BASE_URL}model.json`)
      .then((response) => {
        if (!response.ok) throw new Error('The sentiment model could not be loaded.');
        return response.json() as Promise<ModelArtifact>;
      })
      .catch((error) => {
        modelPromise = null;
        throw error;
      });
  }
  return modelPromise;
}

class ApiError extends Error {}

async function request<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    });
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === 'object' && 'detail' in payload
          ? String((payload as { detail: unknown }).detail)
          : `Request failed with status ${response.status}.`;
      throw new ApiError(detail);
    }
    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('The request timed out. Check that the Python backend is running.');
    }
    if (error instanceof TypeError) {
      throw new ApiError(
        'Cannot reach the Python API. Start the backend or switch to Browser preview mode.'
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function fetchHealth(): Promise<{ status: string; modelVersion: string; engine: string }> {
  return request('/api/health', { method: 'GET' }, TEXT_TIMEOUT_MS);
}

export async function fetchModelInfo(): Promise<Record<string, unknown>> {
  return request('/api/model', { method: 'GET' }, TEXT_TIMEOUT_MS);
}

export async function analyzeTexts(items: TextItem[]): Promise<{ results: AnalysisResult[] }> {
  if (IS_BROWSER_MODE) {
    const model = await loadModel();
    return { results: analyzeItems(items, model) };
  }
  return request(
    '/api/analyze',
    { method: 'POST', body: JSON.stringify({ items }) },
    TEXT_TIMEOUT_MS
  );
}

export async function evaluateLabels(items: TextItem[]): Promise<{
  results: AnalysisResult[];
  metrics: EvaluationMetrics;
}> {
  if (IS_BROWSER_MODE) {
    const model = await loadModel();
    return evaluateItems(items, model);
  }
  return request(
    '/api/evaluate',
    { method: 'POST', body: JSON.stringify({ items }) },
    TEXT_TIMEOUT_MS
  );
}

export async function analyzeImage(bytes: Blob, signal?: AbortSignal): Promise<ImageAnalysisResult> {
  if (IS_BROWSER_MODE) {
    const { analyzeImageInWorker } = await import('./image-sentiment');
    return analyzeImageInWorker(bytes, signal);
  }
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), IMAGE_TIMEOUT_MS);
  const abortFromCaller = () => controller.abort();
  signal?.addEventListener('abort', abortFromCaller);
  try {
    const response = await fetch(`${API_URL}/api/analyze-image`, {
      method: 'POST',
      headers: { 'Content-Type': bytes.type || 'image/jpeg' },
      body: bytes,
      signal: controller.signal,
    });
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === 'object' && 'detail' in payload
          ? String((payload as { detail: unknown }).detail)
          : `Image analysis failed with status ${response.status}.`;
      throw new ApiError(detail);
    }
    return (payload as { result: ImageAnalysisResult }).result;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('Image analysis was cancelled or timed out.');
    }
    if (error instanceof TypeError) {
      throw new ApiError('Cannot reach the Python API for image analysis.');
    }
    throw error;
  } finally {
    signal?.removeEventListener('abort', abortFromCaller);
    window.clearTimeout(timer);
  }
}

export { ApiError, InputError, MAX_CSV_BYTES, INFERENCE_MODE };
