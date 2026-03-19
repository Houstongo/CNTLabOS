import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const indexPath = path.resolve('index.html');
const html = fs.readFileSync(indexPath, 'utf8');

[
  'rag-tab-overview-btn',
  'rag-tab-graph-btn',
  'rag-tab-performance-btn',
  'rag-tab-manage-btn',
  'rag-subpage-overview',
  'rag-subpage-graph',
  'rag-subpage-performance',
  'rag-subpage-manage',
  'rag-overview-more-p2m',
  'rag-overview-more-m2p',
  'rag-overview-more-p2p',
  'rag-overview-more-mech',
  'rag-graph-filter-all',
  'rag-graph-filter-process_to_morphology',
  'rag-graph-filter-morphology_to_performance',
  'rag-graph-filter-process_to_performance',
  'rag-graph-filter-mechanism_evidence',
].forEach((id) => {
  assert.match(html, new RegExp(`id="${id}"`), `${id} should exist`);
});

assert.match(html, /switchRagSubPage\('overview'\)/);
assert.match(html, /switchRagSubPage\('graph'\)/);
assert.match(html, /switchRagSubPage\('performance'\)/);
assert.match(html, /switchRagSubPage\('manage'\)/);
assert.match(html, /renderChainList\('rag-chain-p2m',\s*chain\.process_to_morphology \|\| \[\],\s*2\)/);
assert.match(html, /renderChainList\('rag-chain-m2p',\s*chain\.morphology_to_performance \|\| \[\],\s*2\)/);
assert.match(html, /renderChainList\('rag-chain-p2p',\s*chain\.process_to_performance \|\| \[\],\s*2\)/);
assert.match(html, /renderChainList\('rag-chain-mech',\s*chain\.mechanism_evidence \|\| \[\],\s*2\)/);
assert.match(html, /function setRagGraphFilter\(filter\)/);
assert.match(html, /function getFilteredRagGraphItems\(chain,\s*filter\)/);

console.log('rag workspace contract ok');
