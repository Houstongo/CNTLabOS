// Details 模块 - 详情面板与操作

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { emit, Events } from '../../core/events.js';
import { api } from '../../utils/api.js';
import { AI_BASE } from '../../core/constants.js';
import { formatNumber, formatTemp, formatPosition, formatMagnification, formatTime, formatMinutes, formatThickness } from '../../utils/format.js';
import { aiStorage } from '../../config/local-storage.js';

/**
 * 打开详情面板
 */
export async function openDetails(item) {
    const currentItem = getState('currentItem');
    const isSameImage = currentItem && currentItem.id === item.id;

    setState('currentItem', item);

    const fileName = item.file_path.split('\\').pop();
    const isTif = item.url.toLowerCase().endsWith('.tif') || item.url.toLowerCase().endsWith('.tiff');
    const detailUrl = isTif
        ? `${AI_BASE}/api/view/tif?path=${item.url.replace('/images/', '')}`
        : `${AI_BASE}${item.url}`;

    // 更新 UI
    getEl('detail-img').src = detailUrl;
    getEl('d-name').innerText = fileName;
    getEl('d-source').innerText = item.source === 'XR' ? 'XR 梯度序列' : 'ZZY 参数序列';
    getEl('d-actual-temp').innerText = item.actual_temp ? formatTemp(item.actual_temp) : '--';
    getEl('d-pos-cm').innerText = item.membrane_pos_cm != null ? formatPosition(item.membrane_pos_cm) : '--';

    // 特征值
    getEl('f-dia').innerText = item.diameter_mean != null
        ? `${formatNumber(item.diameter_mean, 1)}±${item.diameter_std != null ? formatNumber(item.diameter_std, 1) : '--'}`
        : '--';
    getEl('f-ali').innerText = item.alignment != null ? formatNumber(item.alignment, 3) : '--';
    getEl('f-den').innerText = item.density != null ? formatNumber(item.density, 2) : '--';
    getEl('f-cur').innerText = (item.curvature != null && !isNaN(item.curvature)) ? formatNumber(item.curvature, 3) : (item.curvature ?? '--');
    getEl('f-tor').innerText = (item.tortuosity != null && !isNaN(item.tortuosity)) ? formatNumber(item.tortuosity, 3) : (item.tortuosity ?? '--');

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
}

/**
 * 打开详情面板（通过 ID）
 */
export async function openDetailsById(id) {
    const currentListItemsById = getState('data.currentListItemsById') || {};
    const item = currentListItemsById[String(id)];
    if (!item) {
        console.warn('找不到对应记录:', id);
        return;
    }
    openDetails(item);
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
            getEl('f-dia').innerText = r.diameter_mean != null
                ? `${formatNumber(r.diameter_mean, 1)}±${r.diameter_std != null ? formatNumber(r.diameter_std, 1) : '--'}`
                : '--';
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
 * 加载算法可视化
 */
async function loadAlgorithmVisualization() {
    const currentItem = getState('currentItem');
    if (!currentItem) return;

    try {
        const data = await api.images.visualize(currentItem.id);
        if (data.steps && data.steps.length > 0) {
            setState('algorithm.steps', data.steps);
            setState('algorithm.currentStepIndex', 0);
            showVisualizationPanel();
        }
    } catch (err) {
        console.error('加载算法可视化失败:', err);
    }
}

/**
 * 显示可视化面板
 */
function showVisualizationPanel() {
    const steps = getState('algorithm.steps');
    const currentStepIndex = getState('algorithm.currentStepIndex') || 0;

    if (!steps || steps.length === 0) return;

    const algoContent = getEl('algo-content');

    let html = `
        <div class="space-y-4">
            <div class="flex justify-between items-center mb-4">
                <h4 class="font-bold text-sm text-slate-700">算法步骤可视化</h4>
                <div class="flex items-center gap-2">
                    <button onclick="window.dispatchEvent(new CustomEvent('algo-prev-step'))" class="px-3 py-1 bg-slate-100 rounded hover:bg-slate-200 text-xs font-bold">
                        <i class="fas fa-chevron-left mr-1"></i>上一步
                    </button>
                    <span id="step-indicator" class="text-xs font-bold text-indigo-600">${currentStepIndex + 1} / ${steps.length}</span>
                    <button onclick="window.dispatchEvent(new CustomEvent('algo-next-step'))" class="px-3 py-1 bg-slate-100 rounded hover:bg-slate-200 text-xs font-bold">
                        下一步<i class="fas fa-chevron-right ml-1"></i>
                    </button>
                </div>
            </div>
            <div class="flex gap-2 overflow-x-auto pb-2">
                ${steps.map((step, idx) => `
                    <button onclick="window.dispatchEvent(new CustomEvent('algo-goto-step', { detail: { index: ${idx} } }))"
                        class="step-nav-btn shrink-0 px-3 py-2 rounded-lg text-xs font-bold border transition-all
                        ${idx === currentStepIndex ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300'}">
                        ${idx + 1}. ${step.name || '步骤 ' + (idx + 1)}
                    </button>
                `).join('')}
            </div>
            <div class="bg-slate-50 rounded-lg p-4 border border-slate-200">
                <img id="step-image" src="data:image/jpeg;base64,${steps[currentStepIndex].image}"
                    class="w-full rounded-lg shadow-sm">
            </div>
            <div class="bg-white rounded-lg p-4 border border-slate-200">
                <div class="text-xs font-bold text-slate-400 uppercase mb-2">步骤说明</div>
                <div id="step-description" class="text-sm text-slate-700">
                    ${steps[currentStepIndex].description || '暂无说明'}
                </div>
            </div>
        </div>
    `;

    algoContent.innerHTML = html;
}

/**
 * 上一步
 */
export function prevStep() {
    const steps = getState('algorithm.steps');
    if (!steps) return;

    const currentStepIndex = getState('algorithm.currentStepIndex') || 0;
    if (currentStepIndex > 0) {
        setState('algorithm.currentStepIndex', currentStepIndex - 1);
        updateVisualization();
    }
}

/**
 * 下一步
 */
export function nextStep() {
    const steps = getState('algorithm.steps');
    if (!steps) return;

    const currentStepIndex = getState('algorithm.currentStepIndex') || 0;
    if (currentStepIndex < steps.length - 1) {
        setState('algorithm.currentStepIndex', currentStepIndex + 1);
        updateVisualization();
    }
}

/**
 * 跳转到指定步骤
 */
export function goToStep(index) {
    const steps = getState('algorithm.steps');
    if (!steps) return;

    if (index >= 0 && index < steps.length) {
        setState('algorithm.currentStepIndex', index);
        updateVisualization();
    }
}

/**
 * 更新可视化
 */
function updateVisualization() {
    const steps = getState('algorithm.steps');
    const currentStepIndex = getState('algorithm.currentStepIndex') || 0;

    if (!steps) return;

    const stepImage = getEl('step-image');
    const stepDescription = getEl('step-description');
    const stepIndicator = getEl('step-indicator');

    if (stepImage) {
        stepImage.src = `data:image/jpeg;base64,${steps[currentStepIndex].image}`;
    }

    if (stepDescription) {
        stepDescription.innerHTML = steps[currentStepIndex].description || '暂无说明';
    }

    if (stepIndicator) {
        stepIndicator.innerText = `${currentStepIndex + 1} / ${steps.length}`;
    }

    // 更新导航按钮状态
    document.querySelectorAll('.step-nav-btn').forEach((btn, idx) => {
        if (idx === currentStepIndex) {
            btn.classList.remove('bg-white', 'text-slate-600', 'border-slate-200', 'hover:border-indigo-300');
            btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-600');
        } else {
            btn.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-600');
            btn.classList.add('bg-white', 'text-slate-600', 'border-slate-200', 'hover:border-indigo-300');
        }
    });
}

/**
 * 打开解释面板
 */
export function openInterpretPanel(tab) {
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
function showAlgoTab() {
    getEl('tab-algo').classList.add('active-tab');
    getEl('tab-ai').classList.remove('active-tab');
    getEl('algo-tab-wrapper').classList.remove('hidden');
    getEl('ai-tab-wrapper').classList.add('hidden');

    const currentItem = getState('currentItem');
    if (currentItem) {
        const { buildAlgoExplanation } = await import('../ai-chat/index.js');
        getEl('algo-content').innerHTML = marked.parse(buildAlgoExplanation(currentItem, currentItem));
    }

    const steps = getState('algorithm.steps');
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
};
