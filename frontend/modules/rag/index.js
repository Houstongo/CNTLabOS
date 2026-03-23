// RAG 模块 - 文献知识库管理

import { getEl } from '../../utils/dom.js';
import { api } from '../../utils/api.js';
import { getState, setState } from '../../core/store.js';
import { emit, Events } from '../../core/events.js';
import { API_BASE } from '../../core/constants.js';

// RAG 关系类型标签
const RAG_RELATION_TYPE_LABELS = {
    process_to_morphology: '工艺→形貌',
    morphology_to_performance: '形貌→性能',
    process_to_performance: '工艺→性能',
    process_to_mechanism: '工艺→机理',
    mechanism_to_morphology: '机理→形貌',
    mechanism_evidence: '机理证据',
};

// RAG 方向标签
const RAG_DIRECTION_LABELS = {
    increase: '增强',
    decrease: '降低',
    improve: '改善',
    reduced: '降低',
    improved: '改善',
    nonlinear: '非线性',
    conditional: '条件相关',
    uncertain: '趋势不确定',
    unknown: '未说明',
};

// RAG 节点类别标签
const RAG_NODE_CATEGORY_LABELS = {
    process: '工艺参数',
    morphology: '形貌特征',
    performance: '材料性能',
    mechanism: '生长机理',
    evidence: '文献证据',
};

// RAG 节点标签
const RAG_NODE_LABELS = {
    alignment: '取向度',
    density: '面密度',
    diameter: '表观直径',
    curvature: '曲率',
    tortuosity: '波曲度',
    waviness: '波曲度',
    conductivity: '导电性',
    resistivity: '电阻率',
    sheet_resistance: '方阻',
    tensile_strength: '拉伸强度',
    modulus: '弹性模量',
    strength: '强度',
    growth_temp: '生长温度',
    growth_time: '生长时间',
    anneal_temp: '退火温度',
    anneal_time: '退火时间',
    fe_thickness: 'Fe厚度',
    al2o3_thickness: 'Al2O3厚度',
    h2_flow: 'H2流量',
    ar_flow: 'Ar流量',
    c2h4_flow: 'C2H4流量',
    catalyst_deactivation: '催化剂失活',
    catalyst_agglomeration: '催化剂团聚',
    boundary_layer_effect: '边界层效应',
    diffusion_limitation: '扩散受限',
    carbon_supply_imbalance: '碳源供给失衡',
    stress_induced_bending: '应力诱导弯曲',
    ostwald_ripening: '奥斯特瓦尔德熟化',
    growth_mechanism: '生长机理',
    evidence: '文献证据',
    unknown: '未知节点',
};

/**
 * 关系类型标签
 */
export function relationTypeLabel(type) {
    return RAG_RELATION_TYPE_LABELS[type] || type || '关联';
}

/**
 * 效应方向标签
 */
export function effectDirectionLabel(value) {
    const raw = String(value || '').trim().toLowerCase();
    if (!raw) return '未说明';
    return RAG_DIRECTION_LABELS[raw] || value;
}

/**
 * 标准化 RAG 节点 ID
 */
function normalizeRagNodeId(nodeId) {
    const raw = String(nodeId || '').trim();
    if (!raw) return { category: '', key: '' };
    const [category, ...rest] = raw.split(':');
    if (rest.length > 0) {
        return { category, key: rest.join(':').trim().toLowerCase() };
    }
    return { category: '', key: raw.toLowerCase() };
}

/**
 * 获取 RAG 节点标签（不含类别）
 */
export function ragNodeLabelOnly(nodeId) {
    const { key } = normalizeRagNodeId(nodeId);
    if (!key) return '--';
    return RAG_NODE_LABELS[key] || key.replace(/_/g, ' ');
}

/**
 * 格式化 RAG 节点标签
 */
export function formatRagNodeLabel(nodeId, withCategory = true) {
    const { category } = normalizeRagNodeId(nodeId);
    const label = ragNodeLabelOnly(nodeId);
    if (!withCategory || !category) return label;
    const prefix = RAG_NODE_CATEGORY_LABELS[category];
    return prefix ? `${prefix}：${label}` : label;
}

/**
 * RAG 节点短标签
 */
export function ragNodeShortLabel(nodeId) {
    return formatRagNodeLabel(nodeId, false);
}

/**
 * RAG 因子标签
 */
export function ragFactorLabel(value) {
    return formatRagNodeLabel(value, false);
}

/**
 * 根据节点 ID 获取类别索引
 */
function nodeCategoryById(nodeId) {
    const id = String(nodeId || '');
    if (id.startsWith('process:')) return 0;
    if (id.startsWith('morphology:')) return 1;
    if (id.startsWith('performance:')) return 2;
    return 3;
}

/**
 * 获取过滤后的 RAG 图谱项
 */
function getFilteredRagGraphItems(chain, filter) {
    const buckets = {
        process_to_morphology: chain?.process_to_morphology || [],
        morphology_to_performance: chain?.morphology_to_performance || [],
        process_to_performance: chain?.process_to_performance || [],
        mechanism_evidence: chain?.mechanism_evidence || [],
    };
    if (!filter || filter === 'all') {
        return [
            ...buckets.process_to_morphology,
            ...buckets.morphology_to_performance,
            ...buckets.process_to_performance,
            ...buckets.mechanism_evidence,
        ];
    }
    return buckets[filter] || [];
}

/**
 * 渲染 RAG 图谱过滤状态
 */
function renderRagGraphFilterState() {
    const ragState = getState('rag') || {};
    const graphFilter = ragState.graphFilter || 'all';

    ['all', 'process_to_morphology', 'morphology_to_performance', 'process_to_performance', 'mechanism_evidence'].forEach((filter) => {
        const btn = getEl(`rag-graph-filter-${filter}`);
        if (btn) {
            btn.classList.toggle('active', graphFilter === filter);
        }
    });
}

/**
 * 设置 RAG 图谱过滤器
 */
export function setRagGraphFilter(filter) {
    setState('rag.graphFilter', filter || 'all');
    renderRagGraphFilterState();

    const ragState = getState('rag') || {};
    if (ragState.currentChain) {
        renderRagGraph(ragState.currentChain, ragState.currentQuery || '');
    }
}

/**
 * 渲染 RAG 图谱过滤计数
 */
function renderRagGraphFilterCounts(chain) {
    const counts = {
        all: getFilteredRagGraphItems(chain, 'all').length,
        process_to_morphology: (chain?.process_to_morphology || []).length,
        morphology_to_performance: (chain?.morphology_to_performance || []).length,
        process_to_performance: (chain?.process_to_performance || []).length,
        mechanism_evidence: (chain?.mechanism_evidence || []).length,
    };

    Object.entries(counts).forEach(([key, count]) => {
        const el = getEl(`rag-graph-filter-count-${key}`);
        if (el) el.textContent = String(count);
    });
}

/**
 * 构建 RAG 链接摘要标记
 */
function buildRagLinkSummaryMarkup(chain) {
    const p2m = (chain?.process_to_morphology || []).length;
    const m2p = (chain?.morphology_to_performance || []).length;
    const p2p = (chain?.process_to_performance || []).length;
    const mech = (chain?.mechanism_evidence || []).length;

    return `
        <div class="bg-indigo-50 border border-indigo-100 rounded-xl p-3">
            <div class="text-slate-500">工艺→形貌</div>
            <div class="text-indigo-700 font-black text-lg mt-1">${p2m}</div>
        </div>
        <div class="bg-emerald-50 border border-emerald-100 rounded-xl p-3">
            <div class="text-slate-500">形貌→性能</div>
            <div class="text-emerald-700 font-black text-lg mt-1">${m2p}</div>
        </div>
        <div class="bg-amber-50 border border-amber-100 rounded-xl p-3">
            <div class="text-slate-500">工艺→性能</div>
            <div class="text-amber-700 font-black text-lg mt-1">${p2p}</div>
        </div>
        <div class="bg-slate-100 border border-slate-200 rounded-xl p-3">
            <div class="text-slate-500">机理证据</div>
            <div class="text-slate-700 font-black text-lg mt-1">${mech}</div>
        </div>
    `;
}

/**
 * 渲染链接摘要
 */
function renderLinkSummary(chain) {
    renderRagGraphFilterCounts(chain);
    const markup = buildRagLinkSummaryMarkup(chain);

    ['rag-toolbar-summary', 'rag-overview-link-summary', 'rag-graph-link-summary'].forEach(id => {
        const summaryEl = getEl(id);
        if (summaryEl) summaryEl.innerHTML = markup;
    });
}

/**
 * 渲染链路列表
 */
function renderChainList(containerId, items, limit = 8) {
    const el = getEl(containerId);
    if (!el) return;

    if (!items || items.length === 0) {
        el.innerHTML = '<div class="text-slate-300">暂无关联</div>';
        return;
    }

    el.innerHTML = items.slice(0, limit).map(it => `
        <div class="bg-white border border-slate-200 rounded-lg px-2.5 py-2">
            <div class="font-bold text-slate-700">${formatRagNodeLabel(it.source_node)} → ${formatRagNodeLabel(it.target_node)}</div>
            <div class="text-[10px] text-slate-500 mt-1">
                方向：${effectDirectionLabel(it.effect_direction)} · 置信度 ${(Number(it.confidence || 0)).toFixed(2)}
            </div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" title="${(it.title || '').replace(/"/g, '&quot;')}">
                ${it.title || '--'}
            </div>
        </div>
    `).join('');
}

/**
 * 加载 RAG 文档列表
 */
export async function loadRagDocs() {
    const listEl = getEl('rag-doc-list');
    if (!listEl) return;

    try {
        const data = await api.rag.documents();
        const docs = data.documents || [];

        setState('rag.documents', docs);

        if (docs.length === 0) {
            listEl.innerHTML = '<div class="text-slate-400 text-sm p-4">暂无文献，请上传 PDF 文件</div>';
            return;
        }

        listEl.innerHTML = docs.map(doc => `
            <div class="flex items-center justify-between p-3 bg-white border border-slate-200 rounded-lg">
                <div class="flex-1 min-w-0">
                    <div class="font-bold text-slate-700 truncate text-sm">${doc.filename || doc.id}</div>
                    <div class="text-[10px] text-slate-400 mt-1">
                        ${doc.chunk_count || 0} 块 · ${doc.uploaded_at || ''}
                    </div>
                </div>
                <button onclick="window.dispatchEvent(new CustomEvent('rag-delete-doc', { detail: { id: ${doc.id}, filename: '${doc.filename || ''}' }}))"
                    class="ml-3 px-3 py-1.5 text-red-500 hover:bg-red-50 rounded-lg text-xs font-bold transition">
                    删除
                </button>
            </div>
        `).join('');

        emit(Events.RAG_DOCS_LOADED, { count: docs.length });
    } catch (err) {
        console.error('Failed to load RAG docs:', err);
        listEl.innerHTML = `<div class="text-red-400 text-sm p-4">加载失败: ${err.message}</div>`;
    }
}

/**
 * 上传 PDF 文件
 */
export async function uploadPDF(input) {
    const file = input.files[0];
    if (!file) return;

    const listEl = getEl('rag-doc-list');
    if (listEl) {
        listEl.innerHTML = `<span class="text-blue-400"><i class="fas fa-spinner fa-spin mr-1"></i>正在处理 "${file.name}"...</span>`;
    }

    try {
        const data = await api.rag.upload(file);
        if (data.status === 'success') {
            alert(`文献导入成功：${data.chunk_count} 个文本块已建立索引`);
            await loadRagDocs();
            emit(Events.RAG_DOCS_LOADED, { count: data.chunk_count });
        } else {
            alert('上传失败: ' + (data.detail || '未知错误'));
        }
    } catch (err) {
        alert('上传失败: ' + err.message);
    } finally {
        input.value = '';
    }
}

/**
 * 删除 RAG 文档
 */
export async function deleteRagDoc(id, filename) {
    if (!confirm(`确认删除文献"${filename}"及其所有索引？`)) return;

    try {
        await api.rag.deleteDoc(id);
        await loadRagDocs();
        emit(Events.RAG_DOCS_LOADED, { action: 'deleted', id });
    } catch (err) {
        alert('删除失败: ' + err.message);
    }
}

/**
 * 加载 RAG 链接
 */
export async function loadRagLinks() {
    const queryInput = getEl('rag-link-query');
    const query = (queryInput?.value || '').trim();

    if (!query) {
        alert('请输入检索词');
        return;
    }

    try {
        const data = await api.rag.links({ query, top_k: 20 });
        const chain = data.chain || {};
        const subgraph = data.subgraph || null;
        const constrainedChain = data.constrained_chain || null;
        const themeAggregation = data.theme_aggregation || null;

        setState('rag.currentChain', chain);
        setState('rag.currentQuery', query);
        setState('rag.currentSubgraph', subgraph);
        setState('rag.currentConstrainedChain', constrainedChain);
        setState('rag.currentThemeAggregation', themeAggregation);

        renderLinkSummary(chain);
        renderChainList('rag-chain-p2m', chain.process_to_morphology || [], 2);
        renderChainList('rag-chain-m2p', chain.morphology_to_performance || [], 2);
        renderChainList('rag-chain-p2p', chain.process_to_performance || [], 2);
        renderChainList('rag-chain-mech', chain.mechanism_evidence || [], 2);

        renderRagGraphAdvanced(subgraph, constrainedChain, query);
        renderRagThemeAggregation(themeAggregation);
        renderRagPerformanceView(chain, query);
        renderRagGraph(chain, query);

        emit(Events.RAG_LINKS_LOADED, { query, chain });
    } catch (err) {
        ['rag-chain-p2m', 'rag-chain-m2p', 'rag-chain-p2p', 'rag-chain-mech', 'rag-performance-list', 'rag-performance-summary', 'rag-graph-insights', 'rag-graph-glance', 'rag-subgraph-summary', 'rag-constrained-chain', 'rag-theme-aggregation'].forEach(id => {
            const el = getEl(id);
            if (el) el.innerHTML = `<div class="text-red-300">加载失败: ${err.message}</div>`;
        });
        ['rag-toolbar-summary', 'rag-overview-link-summary', 'rag-graph-link-summary'].forEach(id => {
            const el = getEl(id);
            if (el) el.innerHTML = `<div class="col-span-2 lg:col-span-4 text-red-300">加载失败: ${err.message}</div>`;
        });

        const chartHost = getEl('rag-graph-chart');
        if (chartHost) {
            const chart = echarts?.getInstanceByDom(chartHost);
            if (chart) {
                chart.clear();
                chart.setOption({
                    title: {
                        text: `图谱加载失败: ${err.message}`,
                        left: 'center',
                        top: 'middle',
                        textStyle: { color: '#ef4444', fontSize: 12, fontWeight: 700 },
                    },
                });
            }
        }
    }
}

/**
 * 加载 RAG 成功统计
 */
async function loadRagSuccessStats() {
    const el = getEl('rag-success-kpis');
    if (!el) return;

    try {
        const stats = await api.rag.stats();
        setState('rag.allStats', stats);

        el.innerHTML = `
            <div class="bg-white border border-slate-200 rounded-xl p-3">
                <div class="text-[11px] text-slate-500">文献数</div>
                <div class="text-xl font-black text-slate-800 mt-1">${stats.document_count || 0}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-3">
                <div class="text-[11px] text-slate-500">文本块</div>
                <div class="text-xl font-black text-slate-800 mt-1">${stats.chunk_count || 0}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-3">
                <div class="text-[11px] text-slate-500">关系条目</div>
                <div class="text-xl font-black text-indigo-700 mt-1">${stats.link_count || 0}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-3">
                <div class="text-[11px] text-slate-500">核心文献</div>
                <div class="text-xl font-black text-emerald-700 mt-1">${stats.core_document_count || 0}</div>
            </div>
        `;

        const ragState = getState('rag') || {};
        if (!ragState.currentChain) {
            renderGlobalRagSummary(stats);
            renderRagDefaultState();
        }
    } catch (err) {
        el.innerHTML = `<div class="col-span-4 text-red-400 text-sm">统计加载失败: ${err.message}</div>`;
    }
}

/**
 * 渲染全局 RAG 摘要
 */
function renderGlobalRagSummary(stats) {
    const el = getEl('rag-global-summary');
    if (!el) return;

    el.innerHTML = `
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-[11px] text-slate-500">文献数</div>
            <div class="text-xl font-black text-slate-800 mt-1">${stats.document_count || 0}</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-[11px] text-slate-500">文本块</div>
            <div class="text-xl font-black text-slate-800 mt-1">${stats.chunk_count || 0}</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-[11px] text-slate-500">关系条目</div>
            <div class="text-xl font-black text-indigo-700 mt-1">${stats.link_count || 0}</div>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-[11px] text-slate-500">核心文献</div>
            <div class="text-xl font-black text-emerald-700 mt-1">${stats.core_document_count || 0}</div>
        </div>
    `;
}

/**
 * 渲染 RAG 默认状态
 */
function renderRagDefaultState() {
    ['rag-chain-p2m', 'rag-chain-m2p', 'rag-chain-p2p', 'rag-chain-mech'].forEach((id) => {
        const el = getEl(id);
        if (el) el.innerHTML = '<div class="text-slate-300">点击"查询"后显示当前结果</div>';
    });

    const perfSummaryEl = getEl('rag-performance-summary');
    if (perfSummaryEl) {
        perfSummaryEl.innerHTML = `
            <div class="col-span-2 bg-white border border-slate-200 rounded-xl p-3 text-slate-400">
                当前为全库统计，点击"查询"后显示性能关联详情
            </div>
        `;
    }

    const perfListEl = getEl('rag-performance-list');
    if (perfListEl) {
        perfListEl.innerHTML = '<div class="text-slate-300">点击"查询"后显示性能证据链</div>';
    }

    const subgraphEl = getEl('rag-subgraph-summary');
    if (subgraphEl) {
        subgraphEl.innerHTML = '<div class="col-span-3 text-slate-300">默认展示全库统计，查询后显示局部子图检索结果</div>';
    }

    const constrainedEl = getEl('rag-constrained-chain');
    if (constrainedEl) {
        constrainedEl.innerHTML = '<div class="text-slate-300">点击"查询"后显示约束证据链</div>';
    }

    const themeAggEl = getEl('rag-theme-aggregation');
    if (themeAggEl) {
        themeAggEl.innerHTML = '<div class="text-slate-300">点击"查询"后显示主题聚合摘要</div>';
    }
}

/**
 * 切换 RAG 子页面
 */
export function switchRagSubPage(page) {
    const allowed = ['overview', 'graph', 'performance', 'manage'];
    const target = allowed.includes(page) ? page : 'overview';

    allowed.forEach(name => {
        const panel = getEl(`rag-subpage-${name}`);
        const btn = getEl(`rag-tab-${name}-btn`);
        if (panel) panel.classList.toggle('hidden', name !== target);
        if (btn) btn.classList.toggle('active', name === target);
    });

    if (target === 'manage') {
        loadRagDocs();
        return;
    }

    const ragState = getState('rag') || {};
    if (target === 'graph') {
        renderRagGraphFilterState();
        if (ragState.currentChain) {
            renderRagGraphAdvanced(
                ragState.currentSubgraph,
                ragState.currentConstrainedChain,
                ragState.currentQuery || '',
            );
            renderRagGraph(ragState.currentChain, ragState.currentQuery || '');
        }
        return;
    }

    if (target === 'performance' && ragState.currentChain) {
        renderRagThemeAggregation(ragState.currentThemeAggregation);
        renderRagPerformanceView(ragState.currentChain, ragState.currentQuery || '');
        return;
    }

    loadRagSuccessStats();
}

/**
 * 渲染 RAG 性能视图
 */
function renderRagPerformanceView(chain, query) {
    const summaryEl = getEl('rag-performance-summary');
    const listEl = getEl('rag-performance-list');
    if (!summaryEl || !listEl) return;

    const morphItems = chain?.morphology_to_performance || [];
    const processItems = chain?.process_to_performance || [];
    const perfItems = [...morphItems, ...processItems];

    if (perfItems.length === 0) {
        summaryEl.innerHTML = `
            <div class="col-span-2 bg-white border border-slate-200 rounded-xl p-3 text-slate-300">
                当前查询暂无性能关联统计
            </div>
        `;
        listEl.innerHTML = `<div class="text-slate-300">没有检索到与"${query}"相关的性能证据链</div>`;
        return;
    }

    const factorCounts = {};
    perfItems.forEach(item => {
        const factor = item.performance_factor || (item.target_node || '').split(':').pop() || 'unknown';
        factorCounts[factor] = (factorCounts[factor] || 0) + 1;
    });
    const topFactors = Object.entries(factorCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

    summaryEl.innerHTML = `
        <div class="bg-white border border-emerald-100 rounded-xl p-3">
            <div class="text-slate-500">性能关系数</div>
            <div class="text-emerald-700 font-black text-lg mt-1">${perfItems.length}</div>
        </div>
        <div class="bg-white border border-amber-100 rounded-xl p-3">
            <div class="text-slate-500">性能指标数</div>
            <div class="text-amber-700 font-black text-lg mt-1">${Object.keys(factorCounts).length}</div>
        </div>
        <div class="bg-white border border-indigo-100 rounded-xl p-3">
            <div class="text-slate-500">形貌驱动</div>
            <div class="text-indigo-700 font-black text-lg mt-1">${morphItems.length}</div>
        </div>
        <div class="bg-white border border-sky-100 rounded-xl p-3">
            <div class="text-slate-500">工艺直达</div>
            <div class="text-sky-700 font-black text-lg mt-1">${processItems.length}</div>
        </div>
        <div class="col-span-2 bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-slate-500 mb-2">当前性能热点</div>
            <div class="flex flex-wrap gap-2">
                ${topFactors.map(([factor, count]) => `
                    <span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-bold">
                        ${ragFactorLabel(factor)}
                        <span class="text-[10px] text-slate-400">${count}</span>
                    </span>
                `).join('')}
            </div>
        </div>
    `;

    listEl.innerHTML = perfItems.slice(0, 8).map(item => `
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="flex items-center justify-between gap-2">
                <div class="font-bold text-slate-700">
                    ${ragFactorLabel(item.source_node)} → ${ragFactorLabel(item.target_node)}
                </div>
                <span class="px-2 py-0.5 rounded-full text-[10px] font-black ${item.relation_type === 'morphology_to_performance' ? 'bg-emerald-50 text-emerald-700' : 'bg-sky-50 text-sky-700'}">
                    ${relationTypeLabel(item.relation_type)}
                </span>
            </div>
            <div class="text-[10px] text-slate-500 mt-1">
                方向：${effectDirectionLabel(item.effect_direction)} · 置信度 ${(Number(item.confidence || 0)).toFixed(2)}
            </div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" title="${(item.title || '').replace(/"/g, '&quot;')}">
                ${item.title || '--'}
            </div>
        </div>
    `).join('');
}

// 导出默认对象
export default {
    loadRagDocs,
    uploadPDF,
    deleteRagDoc,
    loadRagLinks,
    switchRagSubPage,
    setRagGraphFilter,
    relationTypeLabel,
    effectDirectionLabel,
    formatRagNodeLabel,
    ragNodeShortLabel,
    ragFactorLabel,
};
