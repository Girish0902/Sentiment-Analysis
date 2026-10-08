import { useRef, useState } from 'react';
import { analyzeImage, type ImageAnalysisResult } from '../lib/api';
import {
  prepareImage,
  readImageMeta,
  validateImageFile,
  MAX_IMAGE_BYTES,
} from '../lib/image-sentiment';
import { Spinner } from '../components/ui/progress';

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function ImageWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ width: number; height: number } | null>(null);
  const [result, setResult] = useState<ImageAnalysisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  function reset() {
    abortRef.current?.abort();
    setFile(null);
    setPreview(null);
    setMeta(null);
    setResult(null);
    setError(null);
    setCancelled(false);
  }

  async function chooseFile(chosen: File) {
    setError(null);
    setResult(null);
    setCancelled(false);
    try {
      validateImageFile(chosen);
      const dimensions = await readImageMeta(chosen);
      if (preview) URL.revokeObjectURL(preview);
      setFile(chosen);
      setMeta(dimensions);
      setPreview(URL.createObjectURL(chosen));
    } catch (err) {
      setFile(null);
      setPreview(null);
      setMeta(null);
      setError(err instanceof Error ? err.message : 'The image could not be loaded.');
    }
  }

  async function handleAnalyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setCancelled(false);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const { blob } = await prepareImage(file);
      const outcome = await analyzeImage(blob, controller.signal);
      setResult(outcome);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Image analysis failed.';
      if (message.toLowerCase().includes('cancel')) setCancelled(true);
      else setError(message);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="workspace">
      <div className="workspace-grid">
        <div className="panel">
          <h2>Your image</h2>
          <div className="image-zone">
            {preview ? (
              <img className="image-preview" src={preview} alt="Selected preview" />
            ) : (
              <div
                className="dropzone"
                role="button"
                tabIndex={0}
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  const dropped = event.dataTransfer.files[0];
                  if (dropped) void chooseFile(dropped);
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click();
                }}
              >
                Drop a JPG, PNG or WebP here or click to browse
                <div className="file-meta">Max 10 MB · 36 megapixels · aspect ratio up to 20:1</div>
              </div>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              hidden
              onChange={(event) => {
                const chosen = event.target.files?.[0];
                if (chosen) void chooseFile(chosen);
                event.target.value = '';
              }}
            />
            {file && meta ? (
              <div className="file-meta">
                <strong>{file.name}</strong> · {meta.width}×{meta.height} ·{' '}
                {(file.size / 1024).toFixed(0)} KB
              </div>
            ) : null}
            <div className="btn-row">
              <button
                type="button"
                className="btn btn-primary"
                disabled={!file || busy}
                onClick={handleAnalyze}
              >
                {busy ? 'Analyzing…' : 'Analyze image'}
              </button>
              <button type="button" className="btn btn-secondary" disabled={!file && !busy} onClick={reset}>
                {file ? 'Change image' : 'Remove image'}
              </button>
              {busy ? (
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => abortRef.current?.abort()}
                >
                  Cancel
                </button>
              ) : null}
            </div>
            <p className="file-meta">
              The image is processed on your device in Browser preview mode, or sent to the Python
              backend in Python API mode. Uploads are never stored.
            </p>
            {busy ? <Spinner label="Running the image model… first use may download ~160 MB." /> : null}
            {cancelled ? (
              <div className="banner banner-warn" role="status">
                Analysis was cancelled.
              </div>
            ) : null}
            {error ? (
              <div className="banner banner-error" role="alert">
                {error}
              </div>
            ) : null}
          </div>
        </div>

        <div className="panel">
          <h2>Visual tone</h2>
          {!result ? (
            <div className="empty">
              {busy
                ? 'Analyzing…'
                : 'Choose an image and click Analyze image to estimate its visual tone.'}
            </div>
          ) : (
            <>
              <div className="result-head">
                <span
                  className={`badge badge-lg ${
                    result.unclear ? 'badge-warn' : `badge-${result.label}`
                  }`}
                >
                  {result.unclear ? 'Mixed / unclear' : result.label}
                </span>
                <span className="muted small">
                  engine: {result.engine} · model: {result.model}
                </span>
              </div>
              <div className="score-list">
                {(['positive', 'neutral', 'negative'] as const).map((key) => (
                  <div className="score-row" key={key}>
                    <span>{key}</span>
                    <div className="score-track">
                      <div
                        className={`score-fill ${key}`}
                        style={{ width: `${(result.probabilities[key] ?? 0) * 100}%` }}
                      />
                    </div>
                    <span className="score-value">{pct(result.probabilities[key] ?? 0)}</span>
                  </div>
                ))}
              </div>
              <h3 className="section-gap">Top description matches</h3>
              <ul className="match-list">
                {result.matches.map((match) => (
                  <li key={match.label}>
                    <span>{match.label}</span>
                    <span className="muted">{pct(match.score)}</span>
                  </li>
                ))}
              </ul>
              <p className="interp">
                Results below 50%, or within 10 percentage points of the runner-up, are labeled
                Mixed / unclear. These are relative comparisons, not calibrated sentiment
                probabilities, and the matches are suggested descriptions — not detected facts.
              </p>
            </>
          )}
        </div>
      </div>

      <div className="detail-grid section-gap">
        <div className="panel">
          <h2>How the image is analyzed</h2>
          <ol className="muted small">
            <li>The file is validated (type, size, dimensions, aspect ratio).</li>
            <li>It is flattened onto white and reduced to at most 1600px on its longest edge.</li>
            <li>A 224×224 center crop is compared against nine fixed descriptions with CLIP.</li>
            <li>Softmax similarities are summed per class: three descriptions per class.</li>
          </ol>
        </div>
        <div className="panel">
          <h2>Read the result with context</h2>
          <p className="muted small">
            This feature estimates the subjective tone of a scene. It does not determine a person's
            internal emotions, and it may misread text-heavy screenshots, memes, irony or complex
            scenes. Image content near the edges can be missed by the center crop.
          </p>
          <p className="muted small">
            Limits: {MAX_IMAGE_BYTES / (1024 * 1024)} MB, 36 megapixels, aspect ratio no greater
            than 20:1.
          </p>
        </div>
      </div>
    </div>
  );
}
