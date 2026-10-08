import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from '../app/App';
import '../app/globals.css';

export { API_URL, INFERENCE_MODE, IS_BROWSER_MODE } from './config';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
