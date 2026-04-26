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
  'rag-search-fallback',
  'rag-graph-filter-all',
  'rag-graph-filter-process_to_morphology',
  'rag-graph-filter-morphology_to_performance',
  'rag-graph-filter-process_to_performance',
  'rag-graph-filter-mechanism_evidence',
  'rag-graph-filter-count-all',
  'rag-graph-filter-count-process_to_morphology',
  'rag-graph-filter-count-morphology_to_performance',
  'rag-graph-filter-count-process_to_performance',
  'rag-graph-filter-count-mechanism_evidence',
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
assert.match(html, /function renderRagGraphFilterState\(\)/);
assert.match(html, /function renderRagGraphFilterCounts\(chain\)/);
assert.match(html, /function formatRagNodeLabel\(nodeId,\s*withCategory = true\)/);
assert.match(html, /function ragNodeShortLabel\(nodeId\)/);
assert.match(html, /function effectDirectionLabel\(value\)/);
assert.match(html, /function renderRagSearchFallback\(items,\s*query\)/);
assert.match(html, /renderRagSearchFallback\(ragState\.currentItems,\s*query\)/);
assert.match(html, /const displayName = d\.title \|\| d\.filename \|\| \(\(d\.file_path \|\| ''\)\.split\(/);
assert.match(html, /formatRagNodeLabel\(it\.source_node\)\} → \$\{formatRagNodeLabel\(it\.target_node\)\}/);
assert.match(html, /name: ragNodeShortLabel\(source\)/);
assert.match(html, /name: ragNodeShortLabel\(target\)/);

const switchBlock = html.match(/function switchRagSubPage\(page\)\s*\{[\s\S]*?\n        \}/);
assert.ok(switchBlock, 'switchRagSubPage block should exist');
assert.doesNotMatch(
  switchBlock[0],
  /loadRagLinks\(\)/,
  'switchRagSubPage should not auto-run query search; it should default to global summary',
);

console.log('rag workspace contract ok');
