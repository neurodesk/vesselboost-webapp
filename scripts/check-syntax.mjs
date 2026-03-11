#!/usr/bin/env node
/**
 * Pre-deploy syntax checker for web JS files.
 * Parses each .js file as an ES module using acorn to catch syntax errors
 * (e.g. await in non-async functions) before they break the webapp.
 */
import { parse } from 'acorn';
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative } from 'path';

const WEB_DIR = new URL('../web', import.meta.url).pathname;

function collectJsFiles(dir) {
  const files = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (entry === 'node_modules' || entry === 'ort') continue;
    if (statSync(full).isDirectory()) {
      files.push(...collectJsFiles(full));
    } else if (entry.endsWith('.js')) {
      files.push(full);
    }
  }
  return files;
}

const files = collectJsFiles(WEB_DIR);
let failed = 0;

for (const file of files) {
  const rel = relative(WEB_DIR, file);
  const code = readFileSync(file, 'utf8');

  // Detect if the file uses ES module syntax (import/export)
  const isModule = /\b(import|export)\s/.test(code);

  try {
    parse(code, {
      ecmaVersion: 'latest',
      sourceType: isModule ? 'module' : 'script',
      locations: true,
    });
  } catch (err) {
    failed++;
    console.error(`\x1b[31mSYNTAX ERROR\x1b[0m ${rel}:${err.loc.line}:${err.loc.column} — ${err.message}`);
  }
}

if (failed > 0) {
  console.error(`\n${failed} file(s) with syntax errors.`);
  process.exit(1);
} else {
  console.log(`Checked ${files.length} JS files — all OK.`);
}
