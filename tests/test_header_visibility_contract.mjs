import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

assert.match(html, /if \(name === 'clean'\)[\s\S]*mainHeader\.classList\.add\('hidden'\)/);
assert.match(html, /else \{[\s\S]*mainHeader\.classList\.remove\('hidden'\)/);

console.log('header visibility contract ok');
