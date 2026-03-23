// 核心应用初始化
// 职责将所有模块整合到单页应用中

import { repairOverlayMounts } from '../utils/dom.js';
import { emit, Events } from '../../core/events.js';

// 等待 DOM 加载完成
function domReady(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback);
    } else {
        callback();
    }
}

/**
 * 绑定全局事件处理器
 */
function bindGlobalEvents() {
    // 数据行点击事件
    window.addEventListener('data-row-click', async (e) => {
        const { openDetailsById } = await import('../modules/details/index.js');
        openDetailsById(e.detail.id);
    });

    // 数据复选框变化事件
    window.addEventListener('data-checkbox-change', async (e) => {
        const { toggleDataRowSelection } = await import('../modules/data-list/index.js');
        toggleDataRowSelection(e.detail.id, e.detail.checked);
    });

    // 详情页点击事件
    window.addEventListener('data-details-click', async (e) => {
        const { openDetailsById } = await import('../modules/details/index.js');
        openDetailsById(e.detail.id);
    });

    // 清洗选择项事件
    window.addEventListener('clean-select-item', async (e) => {
        const { selectCleanItem } = await import('../modules/data-clean/index.js');
        selectCleanItem(e.detail.id);
    });

    // 殗法步骤导航事件
    window.addEventListener('algo-prev-step', async () => {
        const { prevStep } = await import('../modules/details/index.js');
        prevStep();
    });

    window.addEventListener('algo-next-step', async () => {
        const { nextStep } = await import('../modules/details/index.js');
        nextStep();
    });

    window.addEventListener('algo-goto-step', async (e) => {
        const { goToStep } = await import('../modules/details/index.js');
        goToStep(e.detail.index);
    });

    // RAG 文档删除事件
    window.addEventListener('rag-delete-doc', async (e) => {
        const { deleteRagDoc } = await import('../modules/rag/index.js');
        deleteRagDoc(e.detail.id, e.detail.filename);
    });

    // 窗口大小改变时调整图表
    window.addEventListener('resize', async () => {
        const { resizeMlCharts } = await import('../modules/charts/index.js');
        resizeMlCharts();
    });
}

/**
 * 初始化应用
 */
async function initApp() {
    console.log('CNTA Lab-OS 应用初始化中...');

    // 修复覆盖层挂载
    repairOverlayMounts();

    // 绑定全局事件
    bindGlobalEvents();

    // 初始化各模块 - 按顺序确保依赖
    const { initDataList } = await import('../modules/data-list/index.js');
    const { initCharts } = await import('../modules/charts/index.js');

    console.log('CNTA Lab-OS 应用初始化完成');
}

/**
 * 暴露全局函数给 HTML onclick 使用
 */
// 注意：这些函数需要被导出到 window 才能在 HTML onclick 中调用
export function loadData() {
    const { loadData } = require('../modules/data-list/index.js');
    return loadData();
}

export function toggleSort(field) {
    const { toggleSort } = require('../modules/data-list/index.js');
    return toggleSort(field);
}

export function changePage(dir) {
    const { changePage } = require('../modules/data-list/index.js');
    return changePage(dir);
}

export function jumpToPage() {
    const { jumpToPage } = require('../modules/data-list/index.js');
    return jumpToPage();
}

export function toggleBatchMode() {
    const { toggleBatchMode } = require('../modules/data-list/index.js');
    return toggleBatchMode();
}

export function toggleDataRowSelection(id, checked) {
    const { toggleDataRowSelection } = require('../modules/data-list/index.js');
    return toggleDataRowSelection(id, checked);
}

export function toggleSelectAllDataRows(checked) {
    const { toggleSelectAllDataRows } = require('../modules/data-list/index.js');
    return toggleSelectAllDataRows(checked);
}

export function runDataBatchAnalyze() {
    const { runDataBatchAnalyze } = require('../modules/data-list/index.js');
    runDataBatchAnalyze();
}

export function runDataBatchDelete() {
    const { runDataBatchDelete } = require('../modules/data-list/index.js');
    runDataBatchDelete();
}

export function toggleDataTrashView() {
    const { toggleDataTrashView } = require('../modules/data-list/index.js');
    toggleDataTrashView();
}

export function openDetailsById(id) {
    const { openDetailsById } = require('../modules/details/index.js');
    openDetailsById(id);
}

export function closeDetails() {
    const { closeDetails } = require('../modules/details/index.js');
    return closeDetails();
}

export function closeAll() {
    const { closeAll } = require('../modules/details/index.js');
    return closeAll();
}

export function reanalyzeImage() {
    const { reanalyzeImage } = require('../modules/details/index.js');
    reanalyzeImage();
}

export function openInterpretPanel(tab) {
    const { openInterpretPanel } = require('../modules/details/index.js');
    return openInterpretPanel(tab);
}

export function closeInterpretPanel() {
    const { closeInterpretPanel } = require('../modules/details/index.js');
    return closeInterpretPanel();
}

export function toggleInterpretPanel() {
    const { toggleInterpretPanel } = require('../modules/details/index.js');
    return toggleInterpretPanel();
}

export function startAIInterpret() {
    const { startAIInterpret } = require('../modules/ai-chat/index.js');
    startAIInterpret();
}

export function sendChat() {
    const { sendChat } = require('../modules/ai-chat/index.js');
    return sendChat();
}

export function toggleConfigModal() {
    const modal = getEl('config-modal');
    if (modal) {
        modal.classList.toggle('hidden');
        if (!modal.classList.contains('hidden')) {
            console.log('Config modal opened');
        }
    }
}

export function loadRagLinks() {
    const { loadRagLinks } = require('../modules/rag/index.js');
    return loadRagLinks();
}

export function uploadPDF(input) {
    const { uploadPDF } = require('../modules/rag/index.js');
    return uploadPDF(input);
}

export function deleteRagDoc(id, btn) {
    const docs = getState('rag.documents') || [];
    const doc = docs.find(d => d.id === id);
    const { deleteRagDoc } = require('../modules/rag/index.js');
    deleteRagDoc(id, doc?.filename || '');
}

export function switchRagSubPage(page) {
    const { switchRagSubPage } = require('../modules/rag/index.js');
    return switchRagSubPage(page);
}

export function setRagGraphFilter(filter) {
    const { setRagGraphFilter } = require('../modules/rag/index.js');
    return setRagGraphFilter(filter);
}

export function switchStepTab(btn) {
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    const img = getEl('clean-original-image');
    const b64 = btn.dataset.stepImage;
    const name = btn.dataset.stepName;
    if (img && b64) {
        img.src = b64;
        window.currentLightboxTitle = name;
    }
}

export function openCleanLightbox(src, title) {
    const modal = getEl('clean-lightbox');
    const img = getEl('clean-lightbox-image');
    const titleEl = getEl('clean-lightbox-title');
    if (!modal || !img || !src) return;
    img.src = src;
    if (titleEl) titleEl.innerText = title;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

export function closeCleanLightbox() {
    const modal = getEl('clean-lightbox');
    const img = getEl('clean-lightbox-image');
    if (img) img.removeAttribute('src');
    if (modal) modal.classList.add('hidden');
    document.body.style.overflow = '';
}

export function openCleanOriginal() {
    const img = getEl('clean-original-image');
    if (!img || img.classList.contains('hidden')) return;
    const src = img.src;
    const title = window.currentLightboxTitle || '图像放大查看';
    const { openCleanLightbox } = require('../modules/data-clean/index.js');
    openCleanLightbox(src, title);
}

export function switchMlSubPage(page) {
    const { switchMlSubPage } = require('../modules/charts/index.js');
    switchMlSubPage(page);
}

export function selectMlVizTarget(target) {
    const { selectMlVizTarget } = require('../modules/charts/index.js');
    selectMlVizTarget(target);
}

export function onMlVizOptionChange() {
    const { onMlVizOptionChange } = require('../modules/charts/index.js');
    onMlVizOptionChange();
}

export function switchMlInfoTab(tab) {
    const { switchMlInfoTab } = require('../modules/charts/index.js');
    switchMlInfoTab(tab);
}

// 导出供测试使用
export { initApp, bindGlobalEvents };
