import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const base: string = process.env.VITE_BASE ?? '/';

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    port: 5173,
  },
  preview: {
    port: 5173,
  },
  worker: {
    format: 'es',
  },
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
  },
});
