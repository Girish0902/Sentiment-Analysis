import { useRef } from 'react';

export function Tabs({
  tabs,
  active,
  onChange,
  ariaLabel,
}: {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
  ariaLabel: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const move = (index: number) => {
    const next = (index + tabs.length) % tabs.length;
    onChange(tabs[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div className="tabs" role="tablist" aria-label={ariaLabel}>
      {tabs.map((tab, index) => (
        <button
          key={tab.id}
          ref={(el) => {
            refs.current[index] = el;
          }}
          type="button"
          role="tab"
          id={`tab-${tab.id}`}
          aria-selected={active === tab.id}
          aria-controls={`panel-${tab.id}`}
          tabIndex={active === tab.id ? 0 : -1}
          className={`tab${active === tab.id ? ' active' : ''}`}
          onClick={() => onChange(tab.id)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowRight') {
              event.preventDefault();
              move(index + 1);
            } else if (event.key === 'ArrowLeft') {
              event.preventDefault();
              move(index - 1);
            } else if (event.key === 'Home') {
              event.preventDefault();
              move(0);
            } else if (event.key === 'End') {
              event.preventDefault();
              move(tabs.length - 1);
            }
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
