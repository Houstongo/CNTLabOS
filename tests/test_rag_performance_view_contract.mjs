import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

assert.match(html, /id="rag-tab-performance-btn"/);
assert.match(html, /id="rag-subpage-performance"/);
assert.match(html, /id="rag-performance-summary"/);
assert.match(html, /id="rag-performance-list"/);
assert.match(html, /function renderRagPerformanceView\(chain,\s*query\)/);
assert.match(html, /renderRagPerformanceView\(chain,\s*query\)/);

console.log('rag performance view contract ok');
