import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

test('main data page exposes batch toolbar', () => {
  assert.match(html, /id="data-batch-toolbar"/);
});

test('main data page exposes select-all checkbox', () => {
  assert.match(html, /id="data-select-all"/);
});

test('main data page exposes batch analyze and delete actions', () => {
  assert.match(html, /id="data-batch-analyze-btn"/);
  assert.match(html, /id="data-batch-delete-btn"/);
});
