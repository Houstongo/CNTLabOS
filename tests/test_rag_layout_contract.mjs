import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

const ragIndex = html.indexOf('id="rag-page"');
const lightboxIndex = html.indexOf('id="clean-lightbox"');

assert.notEqual(ragIndex, -1, 'rag-page should exist');
assert.notEqual(lightboxIndex, -1, 'clean-lightbox should exist');
assert.ok(
  ragIndex < lightboxIndex,
  'rag-page must be declared before clean-lightbox so it stays inside the main content container',
);

console.log('rag layout contract ok');
