import { useEffect, useMemo, useRef, useState } from 'react';
import demoCsv from '../data/test.csv?raw';
import { analyzeTexts, evaluateLabels } from '../lib/api';
import {
  itemsFromCsv,
  MAX_CHARS,
  resultsToCsv,
  type AnalysisResult,
  type EvaluationMetrics,
  type TextItem,
} from '../lib/sentiment';
import { Spinner, WorkflowSteps } from '../components/ui/progress';
import { DataTable } from '../components/ui/tables';

export type WorkspaceTab = 'text' | 'image' | 'evaluation';

const EXAMPLES = [
  'Great value and the delivery arrived early!',
  'The app crashes every time I open the settings menu.',
  'The meeting starts at nine.',
];

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

function Badge({ label, large }: { label: string; large?: boolean }) {
  return <span className={`badge badge-${label}${large ? ' badge-lg' : ''}`}>{label}</span>;
}

function ScoreBars({ result }: { result: AnalysisResult }) {
  return (
    <div className="score-list">
      {(['positive', 'neutral', 'negative'] as const).map((label) => (
        <div className="score-row" key={label}>
          <span>{label}</span>
          <div className="score-track">
            <div
              className={`score-fill ${label}`}
              style={{ width: `${result.probabilities[label] * 100}%` }}
            />
          </div>
          <span className="score-value">{pct(result.probabilities[label])}</span>
        </div>
      ))}
    </div>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong.';
}

interface EvalState {
  results: AnalysisResult[];
  metrics: EvaluationMetrics;
}

export default function Workspace({
  tab,
  onTabChange,
}: {
  tab: WorkspaceTab;
  onTabChange: (t: WorkspaceTab) => void;
}) {
  const [inputMode, setInputMode] = useState<'text' | 'csv'>('text');
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stale, setStale] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [csvContent, setCsvContent] = useState<string | null>(null);
  const [csvItems, setCsvItems] = useState<TextItem[] | null>(null);
  const [csvName, setCsvName] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [batch, setBatch] = useState<AnalysisResult[] | null>(null);

  const [evalDataset, setEvalDataset] = useState<'demo' | 'uploaded'>('demo');
  const [demoEval, setDemoEval] = useState<EvalState | null>(null);
  const [uploadedEval, setUploadedEval] = useState<EvalState | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const snapshotRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const activeEval = evalDataset === 'demo' ? demoEval : uploadedEval;
  const step = tab === 'evaluation' ? 5 : result || (batch && batch.length > 1) ? 4 : busy ? 3 : 1;

  useEffect(() => {
    if (tab === 'evaluation' && evalDataset === 'demo' && !demoEval && !evalBusy) {
      runDemoEval();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, evalDataset, demoEval, evalBusy]);

  async function runDemoEval() {
    setEvalBusy(true);
    setEvalError(null);
    try {
      const items = itemsFromCsv(demoCsv, true);
      const { results, metrics } = await evaluateLabels(items);
      setDemoEval({ results, metrics });
    } catch (err) {
      setEvalError(errorMessage(err));
    } finally {
      setEvalBusy(false);
    }
  }

  async function handleAnalyzeText() {
    setBusy(true);
    setError(null);
    try {
      const { results } = await analyzeTexts([{ text }]);
      setResult(results[0]);
      setStale(false);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleAnalyzeCsv() {
    if (!csvContent) return;
    setBusy(true);
    setError(null);
    try {
      const items = itemsFromCsv(csvContent, false);
      const { results } = await analyzeTexts(items);
      setBatch(results);
      setResult(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleEvaluateLabels() {
    if (!csvContent) return;
    setBusy(true);
    setError(null);
    try {
      const items = itemsFromCsv(csvContent, true);
      const { results, metrics } = await evaluateLabels(items);
      setUploadedEval({ results, metrics });
      setEvalDataset('uploaded');
      onTabChange('evaluation');
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  function loadFile(file: File) {
    setCsvError(null);
    if (file.size > 1_048_576) {
      setCsvError('The CSV file is larger than the 1 MB limit.');
      return;
    }
    file.text().then((content) => {
      setCsvContent(content);
      setCsvName(file.name);
      setBatch(null);
      try {
        setCsvItems(itemsFromCsv(content, false));
      } catch (err) {
        setCsvItems(null);
        setCsvError(errorMessage(err));
      }
    });
  }

  function inspect(item: AnalysisResult) {
    setInputMode('text');
    setText(item.text);
    setResult(item);
    setStale(false);
    setError(null);
    window.setTimeout(() => {
      snapshotRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
  }

  function exportCsv(results: AnalysisResult[]) {
    const blob = new Blob([resultsToCsv(results)], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'predictions.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const counts = useMemo(() => {
    if (!batch) return null;
    const map: Record<string, number> = { positive: 0, neutral: 0, negative: 0 };
    for (const item of batch) map[item.label] += 1;
    return map;
  }, [batch]);

  return (
    <div className="workspace">
      <WorkflowSteps active={step} />

      {tab === 'evaluation' ? (
        <section className="workspace" id="panel-evaluation" role="tabpanel" aria-labelledby="tab-evaluation">
          <div className="panel">
            <div className="panel-title-row">
              <h2>Text evaluation</h2>
              <button type="button" className="btn btn-secondary" onClick={() => onTabChange('text')}>
                Use your labeled CSV
              </button>
            </div>
            <div className="segmented" role="group" aria-label="Dataset">
              <button
                type="button"
                className={evalDataset === 'demo' ? 'active' : ''}
                onClick={() => setEvalDataset('demo')}
              >
                Demo test set
              </button>
              <button
                type="button"
                className={evalDataset === 'uploaded' ? 'active' : ''}
                onClick={() => setEvalDataset('uploaded')}
              >
                Your dataset
              </button>
            </div>

            {evalError ? <div className="banner banner-error" role="alert">{evalError}</div> : null}

            {evalBusy ? (
              <Spinner label="Evaluating demo test set…" />
            ) : !activeEval ? (
              <div className="empty">
                Upload a labeled CSV and select <strong>Evaluate labels</strong> in the Text
                analysis workspace.
              </div>
            ) : (
              <>
                <div className="metric-grid">
                  <div className="metric-card">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value">{pct(activeEval.metrics.accuracy)}</div>
                  </div>
                  <div className="metric-card pos">
                    <div className="metric-label">Macro precision</div>
                    <div className="metric-value">{pct(activeEval.metrics.macroPrecision)}</div>
                  </div>
                  <div className="metric-card neu">
                    <div className="metric-label">Macro recall</div>
                    <div className="metric-value">{pct(activeEval.metrics.macroRecall)}</div>
                  </div>
                  <div className="metric-card neg">
                    <div className="metric-label">Macro F1</div>
                    <div className="metric-value">{pct(activeEval.metrics.macroF1)}</div>
                  </div>
                </div>

                <div className="eval-grid section-gap">
                  <div className="panel">
                    <h3>Confusion matrix</h3>
                    <DataTable
                      headers={['actual \\ predicted', ...activeEval.metrics.confusionMatrix.labels]}
                      rows={activeEval.metrics.confusionMatrix.labels.map((label, rowIndex) => [
                        <strong key={label}>{label}</strong>,
                        ...activeEval.metrics.confusionMatrix.rows[rowIndex].map(
                          (count, colIndex) => (
                            <span
                              key={colIndex}
                              className={rowIndex === colIndex ? 'mono' : undefined}
                            >
                              {count}
                            </span>
                          )
                        ),
                      ])}
                    />
                    <p className="muted small">Rows are actual labels, columns are predicted.</p>
                  </div>
                  <div className="panel">
                    <h3>Per-class results</h3>
                    <DataTable
                      headers={['Class', 'Precision', 'Recall', 'F1', 'Support']}
                      rows={(['positive', 'neutral', 'negative'] as const).map((label) => {
                        const m = activeEval.metrics.perClass[label];
                        return [
                          <Badge key={label} label={label} />,
                          pct(m.precision),
                          pct(m.recall),
                          pct(m.f1),
                          String(m.support),
                        ];
                      })}
                    />
                  </div>
                </div>

                <div className="section-gap">
                  <h3>
                    {evalDataset === 'demo' ? 'Demo test texts' : 'Dataset rows'} ·{' '}
                    {activeEval.results.length}
                  </h3>
                  <DataTable
                    headers={['#', 'Text', 'Actual', 'Predicted', 'Result']}
                    rows={activeEval.results.map((item, index) => [
                      String(index + 1),
                      <span className="cell-truncate" title={item.text} key={index}>
                        {item.text}
                      </span>,
                      item.actual ? <Badge key={`a${index}`} label={item.actual} /> : '—',
                      <Badge key={`p${index}`} label={item.label} />,
                      item.correct ? '✓' : '✗',
                    ])}
                  />
                </div>
              </>
            )}
          </div>
        </section>
      ) : (
        <section className="workspace" id="panel-text" role="tabpanel" aria-labelledby="tab-text">
          <div className="workspace-grid">
            <div className="panel">
              <h2>Your text, your insights</h2>
              <div className="segmented" role="group" aria-label="Input type">
                <button
                  type="button"
                  className={inputMode === 'text' ? 'active' : ''}
                  onClick={() => setInputMode('text')}
                >
                  Enter text
                </button>
                <button
                  type="button"
                  className={inputMode === 'csv' ? 'active' : ''}
                  onClick={() => setInputMode('csv')}
                >
                  CSV dataset
                </button>
              </div>

              {inputMode === 'text' ? (
                <>
                  <textarea
                    className="text-input"
                    value={text}
                    maxLength={MAX_CHARS + 200}
                    placeholder="Paste a review, comment or message…"
                    onChange={(event) => {
                      setText(event.target.value);
                      if (result) setStale(true);
                      setError(null);
                    }}
                    aria-label="Text to analyze"
                  />
                  <div className={`char-count${text.length > MAX_CHARS ? ' over' : ''}`}>
                    {text.length} / {MAX_CHARS}
                  </div>
                  <div className="examples">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example}
                        type="button"
                        className="example-btn"
                        onClick={() => {
                          setText(example);
                          if (result) setStale(true);
                          setError(null);
                        }}
                      >
                        {example.length > 42 ? `${example.slice(0, 42)}…` : example}
                      </button>
                    ))}
                  </div>
                  <div className="btn-row">
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={busy || !text.trim() || text.length > MAX_CHARS}
                      onClick={handleAnalyzeText}
                    >
                      {busy ? 'Analyzing…' : 'Analyze sentiment'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => {
                        setText('');
                        setResult(null);
                        setStale(false);
                        setError(null);
                      }}
                    >
                      Clear text
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div
                    className={`dropzone${csvContent ? '' : ''}`}
                    onClick={() => fileRef.current?.click()}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      const file = event.dataTransfer.files[0];
                      if (file) loadFile(file);
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') fileRef.current?.click();
                    }}
                  >
                    {csvName ? (
                      <>
                        <strong>{csvName}</strong>
                        <div className="file-meta">
                          {csvItems ? `${csvItems.length} rows ready` : 'Could not read rows'}
                        </div>
                      </>
                    ) : (
                      <>
                        Drop a CSV here or click to browse
                        <div className="file-meta">
                          Requires a <code>text</code> column · optional <code>label</code> ·
                          max 200 rows, 5,000 chars each, 1 MB
                        </div>
                      </>
                    )}
                  </div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".csv,text/csv"
                    hidden
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) loadFile(file);
                      event.target.value = '';
                    }}
                  />
                  <div className="file-meta">
                    <a href={`${import.meta.env.BASE_URL}sample.csv`} download>
                      Download sample CSV
                    </a>
                  </div>
                  {csvError ? (
                    <div className="banner banner-error" role="alert">
                      {csvError}
                    </div>
                  ) : null}
                  <div className="btn-row">
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={busy || !csvItems}
                      onClick={handleAnalyzeCsv}
                    >
                      {busy ? 'Analyzing…' : 'Analyze texts'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      disabled={busy || !csvContent}
                      onClick={handleEvaluateLabels}
                    >
                      Evaluate labels
                    </button>
                  </div>
                </>
              )}

              {error ? (
                <div className="banner banner-error" role="alert">
                  {error}
                </div>
              ) : null}
            </div>

            <div className="panel" id="sentiment-snapshot" ref={snapshotRef}>
              <h2>Sentiment snapshot</h2>
              {busy && !batch ? (
                <Spinner label="Analyzing…" />
              ) : !result ? (
                <div className="empty">Run an analysis to see the sentiment snapshot.</div>
              ) : (
                <>
                  {stale ? (
                    <div className="banner banner-warn" role="status">
                      Input changed — this result is from the previous text.
                    </div>
                  ) : null}
                  <div className="result-head" style={{ marginTop: stale ? 12 : 0 }}>
                    {result.insufficient ? (
                      <span className="badge badge-warn badge-lg">Insufficient signal</span>
                    ) : (
                      <Badge label={result.label} large />
                    )}
                    <span className="muted small">
                      {result.coverage.known} of {result.coverage.total} features known
                    </span>
                  </div>
                  <ScoreBars result={result} />
                  {result.warnings.length > 0 ? (
                    <div className="banner banner-warn" role="status">
                      <ul>
                        {result.warnings.map((warning) => (
                          <li key={warning}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  <p className="interp">
                    Model-score interpretation: the predicted class has the highest softmax score.
                    Scores are relative comparisons, not calibrated confidence estimates.
                  </p>
                </>
              )}
            </div>
          </div>

          {batch && batch.length > 1 && counts ? (
            <div className="panel section-gap">
              <div className="panel-title-row">
                <h2>Batch results · {batch.length}</h2>
                <button type="button" className="btn btn-secondary" onClick={() => exportCsv(batch)}>
                  Export CSV
                </button>
              </div>
              <div className="counts-row">
                {(['positive', 'neutral', 'negative'] as const).map((label) => (
                  <span key={label} className="chip">
                    {label} {counts[label]}
                  </span>
                ))}
              </div>
              <DataTable
                headers={['Text', 'Sentiment', 'Score', 'Actual', 'Details']}
                rows={batch.map((item, index) => [
                  <span className="cell-truncate" title={item.text} key={index}>
                    {item.text}
                  </span>,
                  <Badge key={`b${index}`} label={item.label} />,
                  pct(Math.max(...['positive', 'neutral', 'negative'].map(
                    (label) => item.probabilities[label as 'positive' | 'neutral' | 'negative']
                  ))),
                  item.actual ? <Badge key={`a${index}`} label={item.actual} /> : '—',
                  <button
                    key={`d${index}`}
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => inspect(item)}
                  >
                    Inspect
                  </button>,
                ])}
              />
            </div>
          ) : null}

          {result ? (
            <div className="detail-grid section-gap">
              <div className="panel">
                <h2>Text preprocessing</h2>
                <h3>Cleaned text</h3>
                <p className="mono">{result.cleaned || '—'}</p>
                <h3 className="section-gap">
                  Removed stop words · {result.removedStopWords.length}
                </h3>
                <div className="chips">
                  {result.removedStopWords.length === 0 ? (
                    <span className="muted small">None</span>
                  ) : (
                    result.removedStopWords.map((word, index) => (
                      <span className="chip plain" key={`${word}-${index}`}>
                        {word}
                      </span>
                    ))
                  )}
                </div>
                <h3 className="section-gap">Tokens after stemming · {result.tokens.length}</h3>
                <div className="chips">
                  {result.tokens.map((token, index) => (
                    <span className="chip" key={`${token}-${index}`}>
                      {token}
                    </span>
                  ))}
                </div>
              </div>
              <div className="panel">
                <h2>Feature extraction</h2>
                <p className="muted small">
                  Vocabulary coverage: {result.coverage.known} of {result.coverage.total} features
                  known · ranked by TF-IDF weight
                </p>
                {result.features.length === 0 ? (
                  <div className="empty">No features matched the model vocabulary.</div>
                ) : (
                  <ul className="feature-list">
                    {result.features.map((feature) => (
                      <li key={feature.term}>
                        <span className="term">{feature.term}</span>
                        <span className="weight">
                          {feature.weight.toFixed(4)} · ×{feature.count}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ) : null}
        </section>
      )}
    </div>
  );
}
