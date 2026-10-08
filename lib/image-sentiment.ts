import type { ImageAnalysisResult } from './api';
import workerUrl from './image-worker?worker&url';

export interface ImageProgress {
  stage: string;
  progress: number | null;
}

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 36_000_000;
export const MAX_ASPECT_RATIO = 20;
const SUPPORTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function validateImageFile(file: File): void {
  if (!SUPPORTED_TYPES.includes(file.type)) {
    throw new Error('Unsupported file type. Use a JPG, PNG or WebP image.');
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error('The image is larger than the 10 MB limit.');
  }
}

export async function readImageMeta(file: File): Promise<{ width: number; height: number }> {
  try {
    const bitmap = await createImageBitmap(file);
    const { width, height } = bitmap;
    bitmap.close();
    if (width * height > MAX_IMAGE_PIXELS) {
      throw new Error('The image exceeds the 36 megapixel limit.');
    }
    const aspect = Math.max(width, height) / Math.max(1, Math.min(width, height));
    if (aspect > MAX_ASPECT_RATIO) {
      throw new Error('The image aspect ratio is greater than 20:1.');
    }
    return { width, height };
  } catch (error) {
    if (error instanceof Error && error.message.includes('limit')) throw error;
    throw new Error('The image could not be decoded.');
  }
}

/**
 * Prepare an image for analysis: flatten onto white, cap the longest edge at
 * 1600px and export RGB JPEG bytes.
 */
export async function prepareImage(file: File): Promise<{
  blob: Blob;
  width: number;
  height: number;
}> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    bitmap.close();
    throw new Error('Canvas is not available in this browser.');
  }
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, 'image/jpeg', 0.92)
  );
  if (!blob) throw new Error('The image could not be converted.');
  return { blob, width, height };
}

export function analyzeImageInWorker(
  bytes: Blob,
  signal?: AbortSignal,
  onProgress?: (update: ImageProgress) => void
): Promise<ImageAnalysisResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(workerUrl, window.location.href), { type: 'module' });
    let settled = false;

    const finish = (action: () => void) => {
      if (settled) return;
      settled = true;
      worker.terminate();
      signal?.removeEventListener('abort', onAbort);
      action();
    };

    const onAbort = () => finish(() => reject(new Error('Image analysis was cancelled.')));
    signal?.addEventListener('abort', onAbort);

    worker.onmessage = (event: MessageEvent) => {
      const message = event.data as
        | { type: 'progress'; stage: string; progress: number | null }
        | { type: 'result'; payload: ImageAnalysisResult }
        | { type: 'error'; message: string };
      if (message.type === 'progress') {
        onProgress?.({ stage: message.stage, progress: message.progress });
      } else if (message.type === 'result') {
        finish(() => resolve(message.payload));
      } else {
        finish(() => reject(new Error(message.message)));
      }
    };
    worker.onerror = (event) => {
      finish(() => reject(new Error(event.message || 'The image worker failed.')));
    };
    worker.postMessage({ image: bytes });
  });
}
