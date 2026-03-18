import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

test('data cleaning nav entry exists', () => {
  assert.match(html, /数据清洗/);
});

test('data cleaning page containers exist', () => {
  assert.match(html, /id="clean-page"/);
  assert.match(html, /id="clean-filter-bar"/);
  assert.match(html, /id="clean-viewer-stage"/);
  assert.match(html, /id="clean-step-strip"/);
  assert.match(html, /id="clean-review-panel"/);
  assert.match(html, /id="clean-sample-list"/);
  assert.match(html, /id="clean-open-original"/);
  assert.match(html, /id="clean-lightbox"/);
});
