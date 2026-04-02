// Details 模块 - 详情面板与操作

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { emit, Events } from '../../core/events.js';
import { api } from '../../utils/api.js';
import { API_BASE } from '../../core/constants.js';
import { formatNumber, formatTemp, formatPosition, formatMagnification, formatTime, formatMinutes, formatThickness } from '../../utils/format.js';
import { aiStorage } from '../../config/local-storage.js';
import * as Viz from '../visualization/index.js';

function formatDiameterDisplay(features) {
    if (features?.diameter_mean != null) {
        return `${formatNumber(features.diameter_mean, 1)}±${features.diameter_std != null ? formatNumber(features.diameter_std, 1) : '--'}`;
    }
    if (features?.diameter != null) {
        return formatNumber(features.diameter, 1);
    }
    return '--';
}

/**
 * 打开详情面板
 */
export async function openDetails(item) {
    const currentItem = getState('currentItem');
    const isSameImage = currentItem && currentItem.id === item.id;

    setState('currentItem', item);
    // 同步内联全局变量（双系统兼容）
    window.currentItem = item;

    const fileName = item.file_path.split('\\').pop();
    const isTif = item.url.toLowerCase().endsWith('.tif') || item.url.toLowerCase().endsWith('.tiff');
    const detailUrl = isTif
        ? `${API_BASE}/api/view/tif?path=${item.url.replace('/images/', '')}`
        : `${API_BASE}${item.url}`;

    // 更新 UI
    getEl('detail-img').src = detailUrl;
    getEl('d-name').innerText = fileName;
    getEl('d-source').innerText = item.source === 'XR' ? 'XR 梯度序列' : 'ZZY 参数序列';
    getEl('d-actual-temp').innerText = item.actual_temp ? formatTemp(item.actual_temp) : '--';
    getEl('d-pos-cm').innerText = item.membrane_pos_cm != null ? formatPosition(item.membrane_pos_cm) : '--';

    // 特征值
    getEl('f-dia').innerText = formatDiameterDisplay(item);
    getEl('f-ali').innerText = item.alignment != null ? formatNumber(item.alignment, 3) : '--';
    getEl('f-den').innerText = item.density != null ? formatNumber(item.density, 2) : '--';
    getEl('f-cur').innerText = (item.curvature != null && !isNaN(item.curvature)) ? formatNumber(item.curvature, 3) : (item.curvature ?? '--');
    getEl('f-tor').innerText = (item.tortuosity != null && !isNaN(item.tortuosity)) ? formatNumber(item.tortuosity, 3) : (item.tortuosity ?? '--');
    getEl('f-wav').innerText = (item.waviness_ratio != null && !isNaN(item.waviness_ratio)) ? formatNumber(item.waviness_ratio, 3) : (item.waviness_ratio ?? '--');

    // 状态
    const statusEl = getEl('d-status');
    statusEl.className = item.processed
        ? "text-[10px] bg-emerald-100 text-emerald-600 px-2 rounded"
        : "text-[10px] bg-slate-100 text-slate-400 px-2 rounded";
    statusEl.innerText = item.processed ? "解析完成" : "未解析";

    renderDetailDeleteActions(item);

    // 切换到不同图像时才重置对话和关闭解释面板
    if (!isSameImage) {
        const { closeInterpretPanel } = require('./details.js');
        closeInterpretPanel();
        setState('chatHistory', []);
        const { clearChat } = require('../ai-chat/index.js');
        clearChat();
    }

    getEl('details-panel').classList.add('open');
    emit(Events.DETAILS_OPENED, { itemId: item.id });
    updateDetailNavButtons();
}

/**
 * 打开详情面板（通过 ID）
 */
export async function openDetailsById(id) {
    const currentListItemsById = getState('data.currentListItemsById') || {};
    const fallbackItem = currentListItemsById[String(id)];
    try {
        const freshItem = await api.images.get(id);
        openDetails(freshItem);
        return;
    } catch (err) {
        console.warn('读取最新详情失败，回退使用列表缓存:', id, err);
    }
    if (!fallbackItem) {
        console.warn('找不到对应记录:', id);
        return;
    }
    openDetails(fallbackItem);
}

/**
 * 关闭详情面板
 */
export function closeDetails() {
    getEl('details-panel').classList.remove('open');
    emit(Events.DETAILS_CLOSED);
}

/**
 * 渲染详情面板删除操作按钮
 */
function renderDetailDeleteActions(item) {
    const deleteBtn = getEl('detail-delete-btn');
    const restoreBtn = getEl('detail-restore-btn');
    if (!deleteBtn || !restoreBtn) return;

    if (item.is_deleted) {
        deleteBtn.classList.add('hidden');
        restoreBtn.classList.remove('hidden');
    } else {
        deleteBtn.classList.remove('hidden');
        restoreBtn.classList.add('hidden');
    }
}

/**
 * 修复覆盖层挂载
 */
export function repairOverlayMounts() {
    ['details-panel', 'interpret-panel', 'config-modal', 'clean-lightbox'].forEach((id) => {
        const el = getEl(id);
        if (el && el.parentElement !== document.body) {
            document.body.appendChild(el);
        }
    });
}

/**
 * 重新分析图像
 */
export async function reanalyzeImage() {
    const currentItem = getState('currentItem');
    if (!currentItem) return;

    const btn = getEl('reanalyze-btn');
    const original = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1.5"></i>提取中...';
    btn.disabled = true;

    try {
        const data = await api.images.analyze(currentItem.id);
        if (data.status === 'success') {
            const r = data.results;
            getEl('f-dia').innerText = formatDiameterDisplay(r);
            getEl('f-ali').innerText = r.alignment != null ? formatNumber(r.alignment, 3) : '--';
            getEl('f-den').innerText = r.density != null ? formatNumber(r.density, 2) : '--';
            getEl('f-cur').innerText = r.curvature ?? '--';
            Object.assign(currentItem, r);

            const { loadData } = await import('../data-list/index.js');
            await loadData();

            const { openInterpretPanel } = require('./details.js');
            openInterpretPanel('algo');

            // 加载算法可视化
            await loadAlgorithmVisualization();
        }
    } catch (err) {
        alert('提取失败：' + err.message);
    } finally {
        btn.innerHTML = original;
        btn.disabled = false;
    }
}

/**
 * 加载算法可视化（委托给 Viz 模块）
 */
export async function loadAlgorithmVisualization() {
    const currentItem = getState('currentItem');
    if (!currentItem) return;

    try {
        const data = await Viz.loadVisualizationData(currentItem.id);
        if (data.steps && data.steps.length > 0) {
            Viz.setSteps(data.steps);
            showVisualizationPanel();
        }
    } catch (err) {
        console.error('加载算法可视化失败:', err);
    }
}

/**
 * 显示可视化面板（详情面板内嵌）
 */
function showVisualizationPanel() {
    const steps = Viz.getSteps();
    if (!steps || steps.length === 0) return;
    Viz.renderStepPanel('algo-content');
}

/**
 * 步骤导航 — 兼容旧事件接口
 */
export function prevStep() {
    Viz.prevStep();
}

export function nextStep() {
    Viz.nextStep();
}

export function goToStep(index) {
    Viz.goToStep(index);
}

/**
 * 打开解释面板
 */
export async function openInterpretPanel(tab) {
    getEl('interpret-panel').classList.add('open');

    const btn = getEl('toggle-interpret-btn');
    if (btn) {
        btn.classList.add('text-indigo-600', 'border-indigo-300', 'bg-indigo-50');
        btn.classList.remove('text-slate-400');
    }

    if (tab === 'algo') {
        showAlgoTab();
    } else {
        showAITab();
        const chatHistory = getState('chatHistory') || [];
        if (chatHistory.length === 0) {
            const { startAIInterpret } = await import('../ai-chat/index.js');
            startAIInterpret();
        }
    }
}

/**
 * 关闭解释面板
 */
export function closeInterpretPanel() {
    getEl('interpret-panel').classList.remove('open');

    const btn = getEl('toggle-interpret-btn');
    if (btn) {
        btn.classList.remove('text-indigo-600', 'border-indigo-300', 'bg-indigo-50');
        btn.classList.add('text-slate-400');
    }
}

/**
 * 切换解释面板
 */
export function toggleInterpretPanel() {
    const panel = getEl('interpret-panel');
    if (panel.classList.contains('open')) {
        closeInterpretPanel();
    } else {
        openInterpretPanel('ai');
    }
}

/**
 * 显示算法标签页
 */
async function showAlgoTab() {
    getEl('tab-algo').classList.add('active-tab');
    getEl('tab-ai').classList.remove('active-tab');
    getEl('algo-tab-wrapper').classList.remove('hidden');
    getEl('ai-tab-wrapper').classList.add('hidden');

    const currentItem = getState('currentItem');
    if (currentItem) {
        const { buildAlgoExplanation } = await import('../ai-chat/index.js');
        getEl('algo-content').innerHTML = marked.parse(buildAlgoExplanation(currentItem, currentItem));
    }

    const steps = Viz.getSteps();
    if (steps) {
        showVisualizationPanel();
    }
}

/**
 * 显示 AI 标签页
 */
function showAITab() {
    getEl('tab-ai').classList.add('active-tab');
    getEl('tab-algo').classList.remove('active-tab');
    getEl('ai-tab-wrapper').classList.remove('hidden');
    getEl('algo-tab-wrapper').classList.add('hidden');
}

/**
 * 关闭所有面板
 */
export function closeAll() {
    closeInterpretPanel();
    closeDetails();
}

/**
 * 获取当前列表 ID 数组（兼容 Store + 内联全局变量）
 */
function _getVisibleIds() {
    const storeMap = getState('data.currentListItemsById') || {};
    if (Object.keys(storeMap).length > 0) return Object.keys(storeMap).map(Number);
    // fallback: 内联全局变量
    if (window.currentListItemsById && Object.keys(window.currentListItemsById).length > 0) {
        return Object.keys(window.currentListItemsById).map(Number);
    }
    return [];
}

/**
 * 详情面板上下页导航
 */
export function navigateDetail(dir) {
    const currentItem = getState('currentItem') || window.currentItem;
    if (!currentItem) return;

    const ids = _getVisibleIds();
    const idx = ids.indexOf(Number(currentItem.id));
    if (idx === -1) return;

    const newIdx = idx + dir;
    if (newIdx < 0 || newIdx >= ids.length) return;

    openDetailsById(ids[newIdx]);
}

/**
 * 更新详情面板导航按钮状态
 */
export function updateDetailNavButtons() {
    const currentItem = getState('currentItem') || window.currentItem;
    if (!currentItem) return;

    const ids = _getVisibleIds();
    const idx = ids.indexOf(Number(currentItem.id));

    const prevBtn = getEl('detail-prev-btn');
    const nextBtn = getEl('detail-next-btn');
    if (prevBtn) prevBtn.disabled = idx <= 0;
    if (nextBtn) nextBtn.disabled = idx < 0 || idx >= ids.length - 1;
}

// 导出默认对象
export default {
    openDetails,
    openDetailsById,
    closeDetails,
    repairOverlayMounts,
    reanalyzeImage,
    openInterpretPanel,
    closeInterpretPanel,
    toggleInterpretPanel,
    prevStep,
    nextStep,
    goToStep,
    closeAll,
    navigateDetail,
    updateDetailNavButtons,
};
