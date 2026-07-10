import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const html = await readFile(new URL('../web/index.html', import.meta.url), 'utf8');

assert.match(
  html,
  /googletagmanager\.com\/gtag\/js\?id=G-4Z9774J59Y/,
  'loads the Neurodesk GA4 measurement ID',
);
assert.match(
  html,
  /gtag\(['"]config['"],\s*['"]G-4Z9774J59Y['"]\)/,
  'configures the correct GA4 measurement ID',
);
assert.match(
  html,
  /navigator\.doNotTrack\s*\|\|\s*window\.doNotTrack\s*\|\|\s*navigator\.msDoNotTrack/,
  'checks browser Do Not Track settings',
);
assert.match(
  html,
  /doNotTrack\s*=\s*dnt\s*==\s*["']1["']\s*\|\|\s*dnt\s*==\s*["']yes["']/,
  'recognizes the same Do Not Track values as neurodesk.org',
);
assert.match(
  html,
  /if\s*\(!doNotTrack\)\s*\{[\s\S]*?gtag\(['"]config['"],\s*['"]G-4Z9774J59Y['"]\);?[\s\S]*?\}/,
  'guards GA4 configuration with the Do Not Track check',
);

assert.doesNotMatch(html, /cloudflareinsights|data-cf-beacon|Cloudflare Web Analytics/i);
assert.match(html, /Google Analytics collects page usage metrics/);
assert.match(html, /disabled when your browser sends a Do Not Track value of "1" or "yes"/);

console.log('Analytics and privacy checks passed.');
