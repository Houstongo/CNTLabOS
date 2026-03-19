import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

assert.match(html, /async function openAlgoVisualization\(\)/);
assert.match(html, /if \(!visualizationSteps\)[\s\S]*await reanalyzeImage\(\)/);

console.log('algo visualization contract ok');
