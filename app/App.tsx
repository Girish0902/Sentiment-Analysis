import { useState } from 'react';
import Workspace, { type WorkspaceTab } from './workspace';
import ImageWorkspace from './image-workspace';
import { Tabs } from '../components/ui/tabs';
import { IS_BROWSER_MODE } from '../src/config';

const TABS = [
  { id: 'text', label: 'Text analysis' },
  { id: 'image', label: 'Image analysis' },
  { id: 'evaluation', label: 'Text evaluation' },
];

export default function App() {
  const [tab, setTab] = useState<WorkspaceTab>('text');

  return (
    <>
      <header className="app-header">
        <div className="shell">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true" />
            Sentiment Lab
          </div>
          <span
            className={`mode-badge ${IS_BROWSER_MODE ? 'browser' : 'python'}`}
            title={
              IS_BROWSER_MODE
                ? 'Self-contained preview: the bundled model runs in your browser.'
                : 'Results come from the local Python FastAPI backend.'
            }
          >
            {IS_BROWSER_MODE ? 'Browser preview' : 'Python API'}
          </span>
        </div>
      </header>

      <main className="shell">
        <section className="intro">
          <h1>Sentiment analysis you can inspect</h1>
          <p>
            Analyze English reviews, comments and messages, watch how stop-word removal, stemming
            and TF-IDF features feed a logistic-regression classifier, then evaluate the model
            against labeled data — and explore the visual tone of images.
          </p>
          <div className="model-line">
            Text model: TF-IDF (unigrams + bigrams) + multinomial logistic regression · v1.1.0
          </div>
        </section>

        <Tabs
          tabs={TABS}
          active={tab}
          onChange={(id) => setTab(id as WorkspaceTab)}
          ariaLabel="Workspaces"
        />

        <div hidden={tab !== 'text' && tab !== 'evaluation'}>
          <Workspace tab={tab} onTabChange={setTab} />
        </div>
        <div hidden={tab !== 'image'}>
          <ImageWorkspace />
        </div>
      </main>

      <footer className="app-footer">
        <div className="shell">
          Sentiment Lab · educational project · model scores are softmax probabilities, not
          calibrated confidence estimates · {new Date().getFullYear()}
        </div>
      </footer>
    </>
  );
}
