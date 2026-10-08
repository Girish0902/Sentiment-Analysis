/**
 * Verify the built image worker stays on the site's own origin and points at
 * an included file (prevents file:/// worker errors after deployment).
 *
 * Usage: node scripts/check-image-worker.mjs dist
 */

import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const distDir = resolve(process.argv[2] ?? 'dist');

if (!existsSync(distDir) || !statSync(distDir).isDirectory()) {
  console.error(`FAIL: build directory ${distDir} not found. Run npm run build first.`);
  process.exit(1);
}

const htmlPath = join(distDir, 'index.html');
if (!existsSync(htmlPath)) {
  console.error('FAIL: dist/index.html is missing.');
  process.exit(1);
}

const assetsDir = join(distDir, 'assets');
const assets = readdirSync(assetsDir);
const workerAssets = assets.filter((name) => /^image-worker-.*\.js$/.test(name));
if (workerAssets.length === 0) {
  console.error('FAIL: no emitted image-worker-*.js asset found in dist/assets.');
  process.exit(1);
}
for (const name of workerAssets) {
  if (statSync(join(assetsDir, name)).size === 0) {
    console.error(`FAIL: worker asset ${name} is empty.`);
    process.exit(1);
  }
}

let failures = 0;
const bundle = assets
  .filter((name) => /^index-.*\.js$/.test(name))
  .map((name) => readFileSync(join(assetsDir, name), 'utf8'))
  .join('\n');

if (!bundle.includes('image-worker-')) {
  console.error('FAIL: main bundle does not reference the image worker asset.');
  failures += 1;
}
if (bundle.includes('file:///')) {
  console.error('FAIL: main bundle contains a file:/// reference.');
  failures += 1;
}
const workerReference = bundle.match(/["'`]([^"'`]*image-worker-[^"'`]*\.js)["'`]/);
if (workerReference && !workerReference[1].startsWith('/') && !workerReference[1].startsWith('./')) {
  console.error(`FAIL: worker URL is not root-relative: ${workerReference[1]}`);
  failures += 1;
}

if (failures > 0) process.exit(1);
console.log(
  `OK: worker asset ${workerAssets[0]} is emitted on the site origin and referenced by the bundle.`
);
