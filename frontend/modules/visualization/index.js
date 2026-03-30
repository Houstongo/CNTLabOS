// Visualization 模块 — 共享算法步骤可视化
// 被 details 和 data-clean 模块复用

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { api } from '../../utils/api.js';

// ── 状态 ──────────────────────────────────────────────

const state = {
    steps: [],
    currentStepIndex: 0,
    backend: 'wcntsegnet',
};

// ── 阶段配置（匹配后端新步骤）────────────────────────

const PHASE_CONFIG_THRESHOLD = [
    { name: '图像预处理', steps: [0, 1, 2, 3, 4] },
    { name: '骨架与分支', steps: [5, 6] },
    { name: '特征提取', steps: null }, // 动态填充到末尾
];

const PHASE_CONFIG_CLDICE = [
    { name: '模型推理', steps: [0, 1, 2, 3, 4, 5] },
    { name: '骨架与分支', steps: [6, 7] },
    { name: '特征提取', steps: null }, // 动态填充到末尾
];

function getPhaseConfig(backend) {
    return backend === 'cldice' ? PHASE_CONFIG_CLDICE : PHASE_CONFIG_THRESHOLD;
}

function resolvePhaseSteps(phases, totalSteps) {
    return phases.map(p => {
        if (p.steps === null) {
            const prevEnd = phases.indexOf(p) > 0
                ? Math.max(...(phases[phases.indexOf(p) - 1].steps || [0])) + 1
                : 0;
            return { ...p, steps: Array.from({ length: totalSteps - prevEnd }, (_, i) => prevEnd + i) };
        }
        return p;
    });
}

// ── 状态操作 ──────────────────────────────────────────

export function setSteps(steps) {
    state.steps = steps;
    state.currentStepIndex = 0;
}

export function getSteps() {
    return state.steps;
}

export function setCurrentStep(idx) {
    state.currentStepIndex = Math.max(0, Math.min(idx, (state.steps.length || 1) - 1));
}

export function getCurrentStepIndex() {
    return state.currentStepIndex;
}

export function setBackend(backend) {
    state.backend = backend;
}

export function getBackend() {
    return state.backend;
}

// ── 导航 ──────────────────────────────────────────────

export function prevStep() {
    if (state.currentStepIndex > 0) {
        state.currentStepIndex--;
        _dispatchStepChange();
    }
}

export function nextStep() {
    if (state.currentStepIndex < state.steps.length - 1) {
        state.currentStepIndex++;
        _dispatchStepChange();
    }
}

export function goToStep(index) {
    if (index >= 0 && index < state.steps.length) {
        state.currentStepIndex = index;
        _dispatchStepChange();
    }
}

function _dispatchStepChange() {
    window.dispatchEvent(new CustomEvent('viz-step-changed', {
        detail: { index: state.currentStepIndex },
    }));
}

// ── 数据加载 ──────────────────────────────────────────

export async function loadVisualizationData(imageId, device = 'cpu') {
    const response = await api.images.visualize(imageId);
    return response;
}

// ── 渲染：步骤面板（单后端）───────────────────────────

/**
 * 渲染完整的步骤可视化面板 HTML。
 * @param {string} containerId - 放入 HTML 的容器 ID
 * @param {object} options - { backend, phases, caption }
 */
export function renderStepPanel(containerId, options = {}) {
    const container = getEl(containerId);
    if (!container || !state.steps.length) return;

    const backend = options.backend || state.backend || 'wcntsegnet';
    const rawPhases = options.phases || getPhaseConfig(backend);
    const phases = resolvePhaseSteps(rawPhases, state.steps.length);
    const idx = state.currentStepIndex;
    const step = state.steps[idx];

    container.innerHTML = `
        ${options.caption ? `<div class="text-xs text-slate-400 mb-2 font-bold">${options.caption}</div>` : ''}
        <!-- 步骤导航 -->
        <div class="flex gap-2 overflow-x-auto pb-2 mb-3">
            ${state.steps.map((s, i) => `
                <button data-viz-goto="${i}"
                    class="viz-step-btn shrink-0 px-3 py-2 rounded-lg text-xs font-bold border transition-all
                    ${i === idx ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm' : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300'}">
                    ${i + 1}. ${s.name || '步骤'}
                </button>
            `).join('')}
        </div>
        <!-- 步骤图像 -->
        <div class="bg-slate-50 rounded-xl p-3 border border-slate-200 mb-3">
            <img data-viz-image src="data:image/jpeg;base64,${step.image}"
                class="w-full rounded-lg shadow-sm">
        </div>
        <!-- 步骤说明 -->
        <div class="bg-white rounded-xl p-4 border border-slate-200">
            <div class="text-xs font-bold text-slate-400 uppercase mb-2">步骤说明</div>
            <div data-viz-desc class="text-sm text-slate-700 leading-relaxed">
                ${step.description || '暂无说明'}
            </div>
        </div>
        <!-- 阶段时间轴 -->
        <div class="mt-4 space-y-1">
            ${phases.map((phase, pi) => `
                <div class="flex items-start gap-2">
                    <div class="flex flex-col items-center">
                        <div class="w-3 h-3 rounded-full border-2 ${_isPhaseActive(phase, idx) ? 'bg-indigo-500 border-indigo-500' : 'bg-white border-slate-300'}"></div>
                        ${pi < phases.length - 1 ? '<div class="w-px h-6 bg-slate-200"></div>' : ''}
                    </div>
                    <div>
                        <div class="text-xs font-bold ${_isPhaseActive(phase, idx) ? 'text-indigo-600' : 'text-slate-400'}">${phase.name}</div>
                    </div>
                </div>
            `).join('')}
        </div>
        <!-- 前后步按钮 -->
        <div class="flex justify-between items-center mt-4">
            <button data-viz-prev class="px-3 py-1.5 bg-slate-100 rounded-lg hover:bg-slate-200 text-xs font-bold disabled:opacity-30" ${idx === 0 ? 'disabled' : ''}>
                <i class="fas fa-chevron-left mr-1"></i>上一步
            </button>
            <span data-viz-indicator class="text-xs font-bold text-indigo-600">${idx + 1} / ${state.steps.length}</span>
            <button data-viz-next class="px-3 py-1.5 bg-slate-100 rounded-lg hover:bg-slate-200 text-xs font-bold disabled:opacity-30" ${idx >= state.steps.length - 1 ? 'disabled' : ''}>
                下一步<i class="fas fa-chevron-right ml-1"></i>
            </button>
        </div>
    `;

    // 绑定事件
    container.querySelectorAll('[data-viz-goto]').forEach(btn => {
        btn.addEventListener('click', () => goToStep(Number(btn.dataset.vizGoto)));
    });
    container.querySelector('[data-viz-prev]')?.addEventListener('click', prevStep);
    container.querySelector('[data-viz-next]')?.addEventListener('click', nextStep);

    // 监听步骤变化
    const cleanup = () => container._vizCleanup?.();
    container._vizCleanup = () => window.removeEventListener('viz-step-changed', onUpdate);
    window.removeEventListener('viz-step-changed', cleanup);
    window.addEventListener('viz-step-changed', onUpdate);
}

function onUpdate(e) {
    const idx = e.detail.index;
    const step = state.steps[idx];
    if (!step) return;

    const container = getEl(document.querySelector('[data-viz-image]')?.closest('[id]')?.id);
    if (!container) return;

    const img = container.querySelector('[data-viz-image]');
    const desc = container.querySelector('[data-viz-desc]');
    const indicator = container.querySelector('[data-viz-indicator]');
    const prev = container.querySelector('[data-viz-prev]');
    const next = container.querySelector('[data-viz-next]');

    if (img) img.src = `data:image/jpeg;base64,${step.image}`;
    if (desc) desc.innerHTML = step.description || '暂无说明';
    if (indicator) indicator.innerText = `${idx + 1} / ${state.steps.length}`;
    if (prev) prev.disabled = idx === 0;
    if (next) next.disabled = idx >= state.steps.length - 1;

    container.querySelectorAll('.viz-step-btn').forEach((btn, i) => {
        if (i === idx) {
            btn.classList.remove('bg-white', 'text-slate-600', 'border-slate-200', 'hover:border-indigo-300');
            btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-600', 'shadow-sm');
        } else {
            btn.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-600', 'shadow-sm');
            btn.classList.add('bg-white', 'text-slate-600', 'border-slate-200', 'hover:border-indigo-300');
        }
    });
}

function _isPhaseActive(phase, stepIndex) {
    return phase.steps.includes(stepIndex);
}

// ── 渲染：步骤卡片列表（左右布局用）──────────────────

/**
 * 渲染左侧步骤卡片列表。
 * @param {string} containerId
 * @param {function} onStepClick - 点击回调
 */
export function renderStepCards(containerId, onStepClick) {
    const container = getEl(containerId);
    if (!container || !state.steps.length) return;

    const idx = state.currentStepIndex;
    container.innerHTML = state.steps.map((step, i) => `
        <button data-viz-card="${i}"
            class="viz-card w-full text-left px-3 py-2.5 rounded-lg border transition-all mb-1
            ${i === idx
                ? 'bg-indigo-50 border-indigo-200 text-indigo-700'
                : 'bg-white border-slate-100 text-slate-600 hover:bg-slate-50'}">
            <div class="flex items-center gap-2">
                <span class="w-5 h-5 rounded-full text-[10px] font-black flex items-center justify-center
                    ${i === idx ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-400'}">${i + 1}</span>
                <span class="text-xs font-bold truncate">${step.name || '步骤'}</span>
            </div>
        </button>
    `).join('');

    container.querySelectorAll('[data-viz-card]').forEach(btn => {
        btn.addEventListener('click', () => {
            goToStep(Number(btn.dataset.vizCard));
            if (onStepClick) onStepClick(Number(btn.dataset.vizCard));
        });
    });
}

/**
 * 渲染右侧步骤详情。
 * @param {string} containerId
 */
export function renderStepDetail(containerId) {
    const container = getEl(containerId);
    if (!container || !state.steps.length) return;

    const idx = state.currentStepIndex;
    const step = state.steps[idx];

    container.innerHTML = `
        <div class="flex justify-between items-center mb-3">
            <span class="text-xs font-bold text-indigo-600">${idx + 1} / ${state.steps.length}</span>
            <div class="flex gap-1">
                <button data-viz-prev class="px-2 py-1 bg-slate-100 rounded hover:bg-slate-200 text-xs" ${idx === 0 ? 'disabled' : ''}>◀</button>
                <button data-viz-next class="px-2 py-1 bg-slate-100 rounded hover:bg-slate-200 text-xs" ${idx >= state.steps.length - 1 ? 'disabled' : ''}>▶</button>
            </div>
        </div>
        <div class="bg-slate-50 rounded-xl p-2 border border-slate-200 mb-3">
            <img data-viz-image src="data:image/jpeg;base64,${step.image}" class="w-full rounded-lg">
        </div>
        <div class="text-xs font-bold text-slate-400 uppercase mb-1">说明</div>
        <div data-viz-desc class="text-sm text-slate-700 leading-relaxed">${step.description || '暂无说明'}</div>
    `;

    container.querySelector('[data-viz-prev]')?.addEventListener('click', prevStep);
    container.querySelector('[data-viz-next]')?.addEventListener('click', nextStep);
}

/**
 * 更新步骤卡片高亮（在 viz-step-changed 时调用）。
 */
export function updateCardHighlight(containerId) {
    const container = getEl(containerId);
    if (!container) return;
    const idx = state.currentStepIndex;
    container.querySelectorAll('.viz-card').forEach((card, i) => {
        const dot = card.querySelector('span:first-child');
        if (i === idx) {
            card.classList.remove('bg-white', 'border-slate-100', 'text-slate-600');
            card.classList.add('bg-indigo-50', 'border-indigo-200', 'text-indigo-700');
            dot.classList.remove('bg-slate-100', 'text-slate-400');
            dot.classList.add('bg-indigo-600', 'text-white');
        } else {
            card.classList.remove('bg-indigo-50', 'border-indigo-200', 'text-indigo-700');
            card.classList.add('bg-white', 'border-slate-100', 'text-slate-600');
            dot.classList.remove('bg-indigo-600', 'text-white');
            dot.classList.add('bg-slate-100', 'text-slate-400');
        }
    });
}

export default {
    setSteps, getSteps, setCurrentStep, getCurrentStepIndex,
    setBackend, getBackend,
    prevStep, nextStep, goToStep,
    loadVisualizationData,
    renderStepPanel, renderStepCards, renderStepDetail, updateCardHighlight,
};
