// Data Clean 模块 - 数据清洗与审查

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { API_BASE } from '../../core/constants.js';
import { fmtMl } from '../../utils/format.js';
import * as Viz from '../visualization/index.js';

// ── 清洗评估 ──────────────────────────────────────────

/**
 * 计算清洗评估
 */
export function computeCleanAssessment(item) {
    const reasons = [];
    let score = 100;
    const density = Number(item.density);
    const alignment = Number(item.alignment);
    const diameter = Number(item.diameter);
    const curvature = Number(item.curvature);
    const tortuosity = Number(item.tortuosity);
    const mag = Number(item.magnification);

    if (!Number.isFinite(density)) {
        score -= 35;
        reasons.push({ level: 'bad', text: '缺少 density，无法判断前景面积比例。' });
    } else {
        if (density <= 0 || density > 95) {
            score -= 40;
            reasons.push({ level: 'bad', text: `density=${fmtMl(density, 1)}%，明显异常，疑似二值化失败。` });
        } else if (density < 5 || density > 85) {
            score -= 18;
            reasons.push({ level: 'warn', text: `density=${fmtMl(density, 1)}%，处于警戒区。` });
        }
    }

    if (!Number.isFinite(alignment)) {
        score -= 20;
        reasons.push({ level: 'bad', text: '缺少 alignment，无法判断取向。' });
    } else if (alignment < -0.5 || alignment > 1.0) {
        score -= 30;
        reasons.push({ level: 'bad', text: `alignment=${fmtMl(alignment, 3)} 超出理论范围。` });
    } else if (alignment < 0) {
        score -= 12;
        reasons.push({ level: 'warn', text: `alignment=${fmtMl(alignment, 3)} 偏低，可能为杂乱网络或算法偏差。` });
    }

    if (!Number.isFinite(diameter)) {
        score -= 15;
        reasons.push({ level: 'warn', text: '缺少 diameter，可能倍率不足或直径提取失败。' });
    } else if (diameter < 5 || diameter > 200) {
        score -= 30;
        reasons.push({ level: 'bad', text: `diameter=${fmtMl(diameter, 1)} nm 不合理，可能测到噪声或束宽。` });
    } else if (diameter > 120) {
        score -= 12;
        reasons.push({ level: 'warn', text: `diameter=${fmtMl(diameter, 1)} nm 偏大，更像束宽而非单管直径。` });
    }

    if (!Number.isFinite(curvature)) {
        score -= 18;
        reasons.push({ level: 'warn', text: '缺少 curvature，局部弯曲强度不可用。' });
    } else if (curvature < 0 || curvature > 100.0) {
        score -= 28;
        reasons.push({ level: 'bad', text: `curvature=${fmtMl(curvature, 3)} um^-1 超出推荐范围。` });
    } else if (curvature > 30.0) {
        score -= 10;
        reasons.push({ level: 'warn', text: `curvature=${fmtMl(curvature, 3)} um^-1 偏高，需检查骨架是否抖动。` });
    }

    if (!Number.isFinite(tortuosity)) {
        score -= 10;
        reasons.push({ level: 'warn', text: '缺少 tortuosity，整体绕曲度无法参考。' });
    } else if (tortuosity < 1.0 || tortuosity > 3.0) {
        score -= 25;
        reasons.push({ level: 'bad', text: `tortuosity=${fmtMl(tortuosity, 3)} 不合理，可能路径追踪异常。` });
    } else if (tortuosity > 2.2) {
        score -= 10;
        reasons.push({ level: 'warn', text: `tortuosity=${fmtMl(tortuosity, 3)} 偏高，需确认是否真实卷曲。` });
    }

    const wavinessRatio = Number(item.waviness_ratio);
    if (!Number.isFinite(wavinessRatio)) {
        score -= 8;
        reasons.push({ level: 'warn', text: '缺少 waviness_ratio，波曲度无法参考。' });
    } else if (wavinessRatio < 0 || wavinessRatio > 5.0) {
        score -= 20;
        reasons.push({ level: 'bad', text: `waviness_ratio=${fmtMl(wavinessRatio, 3)} 超出合理范围。` });
    } else if (wavinessRatio > 2.0) {
        score -= 10;
        reasons.push({ level: 'warn', text: `waviness_ratio=${fmtMl(wavinessRatio, 3)} 偏高，波形可能失真。` });
    }

    if (Number.isFinite(mag) && mag < 20000) {
        score -= 14;
        reasons.push({ level: 'warn', text: `倍率 ${mag}x 偏低，直径与曲率结果天然低可信。` });
    }

    let confidence = 'high';
    let label = '高可信';
    if (score < 55) {
        confidence = 'low';
        label = '低可信';
    } else if (score < 80) {
        confidence = 'medium';
        label = '可参考';
    }

    if (!reasons.length) {
        reasons.push({ level: 'ok', text: '当前样品各字段都在推荐范围内，可先作为高可信样品查看。' });
    }

    return { score: Math.max(0, Math.round(score)), confidence, label, reasons };
}

/**
 * 标准化清洗项
 */
export function normalizeCleanItem(item) {
    const assessment = computeCleanAssessment(item);
    return { ...item, assessment, is_deleted: Number(item.is_deleted || 0) };
}

// ── 列表渲染 ──────────────────────────────────────────

/**
 * 获取过滤后的清洗项
 */
export function getFilteredCleanItems() {
    const cleanState = getState('clean') || {};
    let rows = [...(cleanState.items || [])];
    const confidence = getEl('clean-confidence-filter')?.value || '';
    const processed = getEl('clean-processed-filter')?.value || '';
    const keyword = (getEl('clean-keyword-filter')?.value || '').trim().toLowerCase();

    // confidence 过滤已移除（不再计算 assessment）
    if (confidence) { /* noop */ }
    if (processed !== '') rows = rows.filter(x => String(Number(x.processed || 0)) === processed);
    if (keyword) {
        rows = rows.filter(x => {
            const text = [
                x.sample_id, x.source, x.file_path,
                x.position_label, x.horizontal_pos,
            ].filter(Boolean).join(' ').toLowerCase();
            return text.includes(keyword);
        });
    }

    return rows;
}

/**
 * 获取当前选中的清洗项
 * 兼容 inline JS 的 window.cleanState、ES module store、以及全局 currentItem
 */
export function getActiveCleanItem() {
    // 优先从 ES store / inline 全局变量读取
    const store = getState('clean') || {};
    const inline = window.cleanState || {};
    const selectedId = store.selectedId || inline.selectedId;
    if (selectedId) {
        const items = (store.items || inline.items || []);
        const found = items.find(x => x.id === selectedId);
        if (found) return found;
    }
    // 回退到当前查看的图像
    return window.currentItem || null;
}

/**
 * 选中清洗项
 */
export function selectCleanItem(id) {
    const cleanState = getState('clean') || {};
    cleanState.selectedId = id;
    setState('clean', cleanState);
    renderCleanList();
}

/**
 * 渲染清洗列表
 */
export function renderCleanList() {
    const list = getEl('clean-sample-list');
    const meta = getEl('clean-list-meta');
    if (!list) return;

    const cleanState = getState('clean') || {};
    const rows = getFilteredCleanItems();
    cleanState.filteredItems = rows;

    if (meta) {
        const scopeText = cleanState.view === 'deleted' ? '回收站显示' : '显示';
        meta.innerText = `${scopeText} ${rows.length} / ${cleanState.items.length}`;
    }

    if (!rows.length) {
        list.innerHTML = `<div class="text-sm text-slate-400 font-bold py-8 text-center">${cleanState.view === 'deleted' ? '回收站中没有符合筛选条件的样品' : '没有符合筛选条件的样品'}</div>`;
        return;
    }

    list.innerHTML = rows.map(item => `
        <button type="button" onclick="window.dispatchEvent(new CustomEvent('clean-select-item', { detail: { id: ${item.id} } }))" class="clean-list-row ${cleanState.selectedId === item.id ? 'active' : ''} w-full text-left mb-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 transition p-3">
            <div class="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-2">
                <div class="min-w-0">
                    <div class="font-black text-slate-800 text-sm truncate">${escapeHtml(String(item.sample_id || item.file_path || `ID-${item.id}`))}</div>
                    <div class="text-[11px] text-slate-500 mt-1 truncate">${escapeHtml(String(item.file_path || ''))}</div>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                    ${item.is_deleted ? '<span class="bg-red-100 text-red-700 text-xs px-2 py-1 rounded">已删除</span>' : ''}
                    ${item.processed ? '<span class="bg-emerald-100 text-emerald-700 text-xs px-2 py-1 rounded">已提取</span>' : '<span class="bg-slate-100 text-slate-500 text-xs px-2 py-1 rounded">待处理</span>'}
                </div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-6 gap-2 mt-3 text-[11px] text-slate-500">
                <div>density <span class="font-black text-slate-700">${fmtMl(item.density, 1)}</span></div>
                <div>alignment <span class="font-black text-slate-700">${fmtMl(item.alignment, 3)}</span></div>
                <div>diameter <span class="font-black text-slate-700">${fmtMl(item.diameter, 1)}</span></div>
                <div>curvature (um^-1) <span class="font-black text-slate-700">${fmtMl(item.curvature, 3)}</span></div>
                <div>tortuosity <span class="font-black text-slate-700">${fmtMl(item.tortuosity, 3)}</span></div>
                <div>waviness_ratio <span class="font-black text-slate-700">${fmtMl(item.waviness_ratio, 3)}</span></div>
            </div>
        </button>
    `).join('');
}

function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// ── 算法可视化（清洗模块内）────────────────────────────

/**
 * 切换后端并重新加载
 */
export function switchCleanBackend() {
    // 空壳兼容
}

/**
 * 显示算法可视化面板（清洗模块入口）
 */
export async function showAlgorithmVisualization() {
    // 多种方式获取当前样品
    let activeItem = getActiveCleanItem();
    if (!activeItem?.id) {
        const items = (window.cleanState?.items || []);
        activeItem = items[0] || window.currentItem || null;
    }
    if (!activeItem?.id) {
        alert('请先选择一个样品');
        return;
    }

    const mainLayout = getEl('clean-main-layout');
    const algoPanel = getEl('clean-algo-panel');
    if (!mainLayout || !algoPanel) return;

    mainLayout.classList.add('hidden');
    algoPanel.classList.remove('hidden');

    const contentDiv = getEl('clean-algo-content');
    if (contentDiv) {
        contentDiv.innerHTML = '<div class="text-center text-slate-400 py-10">加载中...</div>';
    }

    try {
        const data = await Viz.loadVisualizationData(activeItem.id);
        if (data.steps && data.steps.length > 0) {
            Viz.setSteps(data.steps);
            Viz.renderStepPanel('clean-algo-content');
        } else {
            if (contentDiv) {
                contentDiv.innerHTML = '<div class="text-sm text-slate-400 py-8 text-center">未获取到可视化步骤</div>';
            }
        }
    } catch (error) {
        const message = error?.message || '未知错误';
        console.error('加载清洗模块算法可视化失败:', error);
        if (contentDiv) {
            contentDiv.innerHTML = `
                <div class="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
                    <div class="font-bold mb-2">算法可视化加载失败</div>
                    <div class="break-all">${escapeHtml(message)}</div>
                </div>`;
        }
    }
}

/**
 * 隐藏清洗模块算法可视化面板
 */
export function hideAlgoPanel() {
    getEl('clean-main-layout')?.classList.remove('hidden');
    getEl('clean-algo-panel')?.classList.add('hidden');
}

/**
 * 切换清洗模块算法可视化面板
 */
export function toggleAlgoPanel() {
    const panel = getEl('clean-algo-panel');
    if (panel?.classList.contains('hidden')) {
        showAlgorithmVisualization();
    } else {
        hideAlgoPanel();
    }
}

// ── 模型异常数据清洗 ──────────────────────────────────────

let _modelCleanData = null;
let _selectedAnomalyId = null;

// 有预测值的 4 个 target
const PREDICTED_TARGETS = [
    { key: 'curvature', label: '曲率 (curvature)', unit: 'um⁻¹', dec: 3 },
    { key: 'waviness_ratio', label: '波曲度 (waviness_ratio)', unit: '', dec: 4 },
    { key: 'tortuosity', label: '绕曲度 (tortuosity)', unit: '', dec: 4 },
    { key: 'alignment', label: '取向 (alignment)', unit: '', dec: 4 },
];

/**
 * 加载模型异常数据（供 inline JS 调用）
 */
export async function loadModelAnomalyData() {
    const listEl = getEl('model-anomaly-list');
    const metaEl = getEl('model-clean-meta');
    if (listEl) listEl.innerHTML = '<div class="text-center text-slate-400 py-10"><i class="fas fa-spinner fa-spin text-2xl"></i><div class="text-xs font-black mt-2">加载中...</div></div>';

    try {
        const res = await fetch(`${API_BASE}/api/model-report/anomaly-review`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        _modelCleanData = await res.json();
        _renderModelSummaryBar();
        _renderAnomalyList();
        // 重置详情
        _selectedAnomalyId = null;
        _renderAnomalyDetail(null);
    } catch (err) {
        if (listEl) listEl.innerHTML = `<div class="text-sm text-rose-500 font-bold py-8 text-center">加载失败: ${escapeHtml(err.message)}</div>`;
    }
}

/**
 * 渲染模型概览条（左侧栏顶部 chips）
 */
function _renderModelSummaryBar() {
    const chipsEl = getEl('model-summary-chips');
    if (!chipsEl || !_modelCleanData) return;
    const best = _modelCleanData.model_summary?.best_results_by_target || {};
    const summary = _modelCleanData.model_summary || {};
    const anomaly = _modelCleanData.anomaly_summary || {};

    const metaEl = getEl('model-clean-meta');
    if (metaEl) metaEl.textContent = `${anomaly.candidate_count || 0} 异常 / ${summary.row_count || 0} 总`;

    chipsEl.innerHTML = Object.entries(best).map(([t, info]) => {
        const r2 = Number(info.r2);
        const color = r2 > 0.3 ? 'bg-emerald-100 text-emerald-700' : r2 > 0 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-500';
        return `<span class="text-[10px] ${color} px-2 py-0.5 rounded-full font-bold">${t} R²=${r2.toFixed(2)}</span>`;
    }).join('');
}

/**
 * 渲染异常候选列表（左侧栏）
 */
function _renderAnomalyList() {
    const listEl = getEl('model-anomaly-list');
    if (!listEl || !_modelCleanData) return;
    const candidates = _modelCleanData.candidates || [];

    if (!candidates.length) {
        listEl.innerHTML = '<div class="text-sm text-slate-400 font-bold py-8 text-center">无异常候选</div>';
        return;
    }

    listEl.innerHTML = candidates.map((c, i) => {
        const score = Number(c.anomaly_score || 0);
        const scoreColor = score >= 6 ? 'bg-rose-500' : score >= 4 ? 'bg-amber-500' : score >= 2 ? 'bg-yellow-400' : 'bg-slate-300';
        const isActive = _selectedAnomalyId === Number(c.image_id);
        const reasons = (c.anomaly_reasons || '').split(';').map(r => r.trim()).filter(Boolean);
        const is45 = c.is_45min === 'True';

        return `
        <button type="button" onclick="window.dispatchEvent(new CustomEvent('model-anomaly-select', { detail: { index: ${i} } }))"
                class="w-full text-left mb-1.5 rounded-xl border ${isActive ? 'border-violet-300 bg-violet-50' : 'border-slate-200 bg-white hover:bg-slate-50'} transition p-3">
            <div class="flex items-center justify-between gap-2 mb-1.5">
                <span class="font-black text-slate-800 text-xs truncate">${escapeHtml(c.sample_id || `ID-${c.image_id}`)}</span>
                <span class="${scoreColor} text-white text-[10px] font-black px-2 py-0.5 rounded-full shrink-0">${score}</span>
            </div>
            <div class="grid grid-cols-3 gap-1 text-[10px] text-slate-500 mb-1.5">
                <div>FE <span class="font-black text-slate-700">${c.fe_power}W</span></div>
                <div>厚 <span class="font-black text-slate-700">${c.fe_thickness}nm</span></div>
                <div>退火 <span class="font-black text-slate-700">${Number(c.anneal_time) * 60}min</span></div>
            </div>
            <div class="flex flex-wrap gap-1">
                ${is45 ? '<span class="text-[10px] bg-violet-100 text-violet-600 px-1.5 py-0.5 rounded font-bold">45min</span>' : ''}
                ${reasons.slice(0, 2).map(r => `<span class="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded font-bold truncate max-w-[100px]">${escapeHtml(r)}</span>`).join('')}
                ${reasons.length > 2 ? `<span class="text-[10px] text-slate-400">+${reasons.length - 2}</span>` : ''}
            </div>
        </button>`;
    }).join('');
}

/**
 * 选中异常项并渲染详情
 */
export function selectModelAnomaly(index) {
    if (!_modelCleanData?.candidates?.[index]) return;
    _selectedAnomalyId = Number(_modelCleanData.candidates[index].image_id);
    _renderAnomalyList();
    _renderAnomalyDetail(_modelCleanData.candidates[index]);
}

/**
 * 渲染异常详情（右侧面板）：SEM 图 + 实测vs预测对比表
 */
function _renderAnomalyDetail(c) {
    const caption = getEl('model-detail-caption');
    const actionsEl = getEl('model-detail-actions');
    const contentEl = getEl('model-detail-content');
    if (!contentEl) return;

    if (!c) {
        if (caption) caption.textContent = '选择左侧异常项查看详情';
        if (actionsEl) actionsEl.innerHTML = '';
        contentEl.innerHTML = `
            <div class="text-center text-slate-400 py-20">
                <div class="text-5xl mb-4 opacity-40"><i class="fas fa-flask"></i></div>
                <div class="text-sm font-black">选择左侧异常候选查看 SEM 图像与预测对比</div>
            </div>`;
        return;
    }

    if (caption) caption.textContent = c.sample_id || `ID-${c.image_id}`;

    // 软删除按钮
    if (actionsEl) {
        actionsEl.innerHTML = `<button type="button" onclick="window.dispatchEvent(new CustomEvent('model-anomaly-delete', { detail: { imageId: ${c.image_id}, sampleId: '${escapeHtml(c.sample_id || '')}' } }))" class="px-4 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-xs font-black shadow-lg shadow-rose-200 transition-all flex items-center gap-2"><i class="fas fa-trash-alt"></i> 移入回收站</button>`;
    }

    // SEM 图
    const imgSrc = `${API_BASE}/api/model-report/image/${c.image_id}`;
    const reasons = (c.anomaly_reasons || '').split(';').map(r => r.trim()).filter(Boolean);

    // 实测 vs 预测对比表
    let comparisonRows = PREDICTED_TARGETS.map(t => {
        const actual = Number(c[t.key] || 0);
        const pred = Number(c[`${t.key}_pred`] || 0);
        const resid = Number(c[`${t.key}_resid`] || 0);
        const residZ = Number(c[`${t.key}_resid_z`] || 0);
        const featureZ = Number(c[`${t.key}_z`] || 0);
        const pct = pred !== 0 ? ((actual - pred) / Math.abs(pred) * 100) : 0;

        // 颜色标记
        let residColor = 'text-slate-600';
        let bgColor = '';
        if (Math.abs(residZ) > 3) { residColor = 'text-rose-600 font-black'; bgColor = 'bg-rose-50'; }
        else if (Math.abs(residZ) > 2) { residColor = 'text-amber-600 font-black'; bgColor = 'bg-amber-50'; }

        const arrow = resid > 0 ? '↑' : resid < 0 ? '↓' : '';
        const sign = resid > 0 ? '+' : '';

        return `<tr class="border-t border-slate-100 ${bgColor}">
            <td class="px-4 py-3 text-xs font-black text-slate-700">${t.label}</td>
            <td class="px-4 py-3 text-center text-xs font-black text-slate-800">${actual.toFixed(t.dec)} <span class="text-slate-400">${t.unit}</span></td>
            <td class="px-4 py-3 text-center text-xs font-bold text-slate-500">${pred.toFixed(t.dec)} <span class="text-slate-400">${t.unit}</span></td>
            <td class="px-4 py-3 text-center text-xs ${residColor}">${sign}${resid.toFixed(t.dec)} <span class="text-slate-400">${t.unit}</span> <span class="text-[10px]">${arrow}</span></td>
            <td class="px-4 py-3 text-center text-[11px] font-bold ${residColor}">${sign}${residZ.toFixed(2)}σ</td>
            <td class="px-4 py-3 text-center text-[11px] font-bold ${residColor}">${sign}${pct.toFixed(1)}%</td>
        </tr>`;
    }).join('');

    // 无预测的 target（density, diameter）
    const extraInfo = `
    <tr class="border-t border-slate-200 bg-slate-50">
        <td class="px-4 py-2.5 text-xs font-black text-slate-500">density</td>
        <td class="px-4 py-2.5 text-center text-xs font-bold text-slate-600">${Number(c.density || 0).toFixed(1)} %</td>
        <td class="px-4 py-2.5 text-center text-xs text-slate-400" colspan="4">无预测值</td>
    </tr>
    <tr class="border-t border-slate-200 bg-slate-50">
        <td class="px-4 py-2.5 text-xs font-black text-slate-500">diameter</td>
        <td class="px-4 py-2.5 text-center text-xs font-bold text-slate-600">${Number(c.diameter || 0).toFixed(1)} nm</td>
        <td class="px-4 py-2.5 text-center text-xs text-slate-400" colspan="4">无预测值</td>
    </tr>`;

    contentEl.innerHTML = `
    <!-- SEM 图像 -->
    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="p-4">
            <img src="${imgSrc}" class="w-full max-h-[50vh] object-contain rounded-xl bg-slate-50 border border-slate-100" alt="SEM" />
        </div>
        <div class="px-4 py-3 border-t border-slate-100 bg-slate-50 text-[11px] text-slate-400 font-bold truncate">
            ${escapeHtml(c.file_path || '')}
        </div>
    </div>

    <!-- 工艺参数 -->
    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
        <div class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">工艺参数</div>
        <div class="grid grid-cols-3 md:grid-cols-6 gap-3">
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">Sample</div>
                <div class="text-sm font-black text-slate-800 mt-0.5">${escapeHtml(c.sample_prefix || '--')}</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">FE Power</div>
                <div class="text-sm font-black text-slate-800 mt-0.5">${c.fe_power}W</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">FE Thick.</div>
                <div class="text-sm font-black text-slate-800 mt-0.5">${c.fe_thickness}nm</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">退火时间</div>
                <div class="text-sm font-black text-slate-800 mt-0.5">${Number(c.anneal_time) * 60}min</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">异常分</div>
                <div class="text-sm font-black text-rose-600 mt-0.5">${c.anomaly_score}</div>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl">
                <div class="text-[10px] text-slate-400 font-bold">批次</div>
                <div class="text-sm font-black text-slate-800 mt-0.5">${c.is_45min === 'True' ? '<span class="text-violet-600">45min</span>' : '标准'}</div>
            </div>
        </div>
    </div>

    <!-- 实测 vs 预测对比表 -->
    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div class="px-5 py-4 border-b border-slate-100">
            <div class="text-[10px] font-black text-slate-400 uppercase tracking-widest">实测值 vs 模型预测值</div>
            <div class="text-[11px] text-slate-400 mt-0.5">偏差绝对值及 Z-score，红色为显著偏离（|Z| > 3σ）</div>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-xs">
                <thead>
                    <tr class="bg-slate-50 text-slate-500">
                        <th class="px-4 py-3 text-left font-black">目标</th>
                        <th class="px-4 py-3 text-center font-black">实测值</th>
                        <th class="px-4 py-3 text-center font-black">预测值</th>
                        <th class="px-4 py-3 text-center font-black">偏差</th>
                        <th class="px-4 py-3 text-center font-black">Z-score</th>
                        <th class="px-4 py-3 text-center font-black">相对偏差</th>
                    </tr>
                </thead>
                <tbody>${comparisonRows}${extraInfo}</tbody>
            </table>
        </div>
    </div>

    <!-- 异常原因 -->
    <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
        <div class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">异常判定依据</div>
        <div class="flex flex-wrap gap-2">
            ${reasons.map(r => {
                const isHigh = r.includes('high');
                const isLow = r.includes('low');
                const color = isHigh ? 'bg-rose-100 text-rose-700 border-rose-200' : isLow ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-amber-100 text-amber-700 border-amber-200';
                return `<span class="text-[11px] ${color} px-3 py-1.5 rounded-lg font-bold border">${escapeHtml(r)}</span>`;
            }).join('')}
        </div>
    </div>`;
}

/**
 * 软删除异常项
 */
export async function softDeleteAnomalyItem(imageId, sampleId) {
    if (!confirm(`确认将 ${sampleId || 'ID-' + imageId} 移入回收站？`)) return;
    try {
        const res = await fetch(`${API_BASE}/api/images/${imageId}/delete`, { method: 'PUT' });
        if (res.ok) {
            if (_modelCleanData?.candidates) {
                _modelCleanData.candidates = _modelCleanData.candidates.filter(c => Number(c.image_id) !== imageId);
                _modelCleanData.anomaly_summary.candidate_count = _modelCleanData.candidates.length;
                _renderModelSummaryBar();
                _renderAnomalyList();
                _selectedAnomalyId = null;
                _renderAnomalyDetail(null);
            }
        } else {
            alert('删除失败');
        }
    } catch (err) {
        console.error('删除失败:', err);
        alert('删除失败: ' + err.message);
    }
}

// 导出默认对象
export default {
    computeCleanAssessment,
    normalizeCleanItem,
    getFilteredCleanItems,
    getActiveCleanItem,
    selectCleanItem,
    renderCleanList,
    showAlgorithmVisualization,
    hideAlgoPanel,
    toggleAlgoPanel,
    loadModelAnomalyData,
    selectModelAnomaly,
    softDeleteAnomalyItem,
};
