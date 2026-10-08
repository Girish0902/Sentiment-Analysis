export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="spinner-row" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ProgressBar({ label, value }: { label: string; value: number | null }) {
  const pct = value === null ? 100 : Math.round(value * 100);
  return (
    <div className="progress-row" role="status">
      <div className="progress-track">
        <div
          className={`progress-fill${value === null ? ' indeterminate' : ''}`}
          style={{ width: `${value === null ? 40 : pct}%` }}
        />
      </div>
      <span className="progress-label">{value === null ? label : `${label} ${pct}%`}</span>
    </div>
  );
}

const STEPS = ['Collect', 'Preprocess', 'Extract features', 'Classify', 'Evaluate'];

export function WorkflowSteps({ active }: { active: number }) {
  return (
    <ol className="stepper" aria-label="Analysis workflow">
      {STEPS.map((step, index) => {
        const position = index + 1;
        const state = position < active ? 'done' : position === active ? 'active' : 'todo';
        return (
          <li key={step} className={`step ${state}`}>
            <span className="step-dot">{position}</span>
            <span className="step-label">{step}</span>
          </li>
        );
      })}
    </ol>
  );
}
