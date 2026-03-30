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
    } else if (curvature < 0 || curvature > 2.0) {
        score -= 28;
        reasons.push({ level: 'bad', text: `curvature=${fmtMl(curvature, 3)} 超出推荐范围。` });
    } else if (curvature > 0.8) {
        score -= 10;
        reasons.push({ level: 'warn', text: `curvature=${fmtMl(curvature, 3)} 偏高，需检查骨架是否抖动。` });
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

    if (confidence) rows = rows.filter(x => x.assessment.confidence === confidence);
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
                    <span class="clean-chip ${item.assessment.confidence}">${item.assessment.label}</span>
                    <span class="text-[11px] font-black text-slate-500">分数 ${item.assessment.score}</span>
                </div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-2 mt-3 text-[11px] text-slate-500">
                <div>density <span class="font-black text-slate-700">${fmtMl(item.density, 1)}</span></div>
                <div>alignment <span class="font-black text-slate-700">${fmtMl(item.alignment, 3)}</span></div>
                <div>diameter <span class="font-black text-slate-700">${fmtMl(item.diameter, 1)}</span></div>
                <div>curvature <span class="font-black text-slate-700">${fmtMl(item.curvature, 3)}</span></div>
                <div>tortuosity <span class="font-black text-slate-700">${fmtMl(item.tortuosity, 3)}</span></div>
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
    const caption = getEl('clean-algo-caption');
    if (!mainLayout || !algoPanel) return;

    mainLayout.classList.add('hidden');
    algoPanel.classList.remove('hidden');
    if (caption) {
        caption.textContent = `当前样品：${activeItem.sample_id || activeItem.id} · ${activeItem.source || '--'}`;
    }

    const contentDiv = getEl('clean-algo-content');
    if (contentDiv) {
        contentDiv.innerHTML = '<div class="text-center text-slate-400 py-10">算法可视化加载中...</div>';
    }

    try {
        const data = await Viz.loadVisualizationData(activeItem.id);
        if (data.steps && data.steps.length > 0) {
            Viz.setSteps(data.steps);
            const backend = data.backend || 'threshold_fallback';
            Viz.setBackend(backend === 'cldice' ? 'cldice' : 'wcntsegnet');
            renderCleanAlgoView(data);
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
 * 渲染清洗模块的算法可视化（左右分栏布局）
 */
function renderCleanAlgoView(data) {
    const stepsContainer = getEl('clean-algo-steps');
    const detailContainer = getEl('clean-algo-detail');
    const backend = Viz.getBackend();

    if (stepsContainer) {
        Viz.renderStepCards('clean-algo-steps', (idx) => {
            if (detailContainer) Viz.renderStepDetail('clean-algo-detail');
        });
    }
    if (detailContainer) {
        Viz.renderStepDetail('clean-algo-detail');
    }

    // 监听步骤变化更新卡片高亮和详情
    const handler = () => {
        if (stepsContainer) Viz.updateCardHighlight('clean-algo-steps');
        if (detailContainer) Viz.renderStepDetail('clean-algo-detail');
    };
    window.addEventListener('viz-step-changed', handler);
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
};
