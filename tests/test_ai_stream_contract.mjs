import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

assert.match(html, /async function consumeSseJsonStream\(response, onMessage\)/);
assert.match(html, /m\.type === 'error'/);
assert.match(html, /await consumeSseJsonStream\(res, async \(m\) =>/);

console.log('ai stream contract ok');
