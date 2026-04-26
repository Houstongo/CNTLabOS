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
  assert.match(html, /id="clean-sample-list"/);
  assert.match(html, /id="clean-review-content"/);
  assert.match(html, /id="clean-open-original"/);
  assert.match(html, /id="clean-lightbox"/);
});

test('data cleaning exposes a recycle-bin entry point', () => {
  assert.match(html, /id="clean-trash-toggle"/);
});

test('data cleaning page switch closes legacy overlay panels', () => {
  assert.match(
    html,
    /else if \(name === 'clean'\) \{[\s\S]*closeInterpretPanel\(\);[\s\S]*closeDetails\(\);/
  );
  assert.match(
    html,
    /function closeInterpretPanel\(\) \{[\s\S]*const panel = document\.getElementById\('interpret-panel'\);[\s\S]*panel\.classList\.remove\('open'\);[\s\S]*panel\.style\.display = 'none';[\s\S]*\}/
  );
  assert.match(
    html,
    /function closeDetails\(\) \{[\s\S]*const panel = document\.getElementById\('details-panel'\);[\s\S]*panel\.classList\.remove\('open'\);[\s\S]*panel\.style\.display = ''[\s\S]*\}/
  );
});
