// 核心应用初始化
// 职责将所有模块整合到单页应用中

import { repairOverlayMounts, getEl } from '../utils/dom.js';
import { getState } from './store.js';
import { API_BASE } from './constants.js';

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

    // 算法步骤导航事件（详情面板）
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

    // 清洗模块算法可视化事件
    window.addEventListener('clean-algo-toggle', async () => {
        const { toggleAlgoPanel } = await import('../modules/data-clean/index.js');
        toggleAlgoPanel();
    });

    // 模型异常点击查看图片
    window.addEventListener('model-anomaly-select', async (e) => {
        const { selectModelAnomaly } = await import('../modules/data-clean/index.js');
        selectModelAnomaly(e.detail.index);
    });

    // 模型异常软删除
    window.addEventListener('model-anomaly-delete', async (e) => {
        const { softDeleteAnomalyItem } = await import('../modules/data-clean/index.js');
        softDeleteAnomalyItem(e.detail.imageId, e.detail.sampleId);
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
 * 注意：export 不会挂到 window，必须显式 window.xxx 赋值
 */

// ── 数据列表 ──

window.loadData = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.loadData();
};

window.resetAndLoad = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.resetAndLoad();
};

window.toggleSort = async (field) => {
    const m = await import('../modules/data-list/index.js');
    return m.toggleSort(field);
};

window.changePage = async (dir) => {
    const m = await import('../modules/data-list/index.js');
    return m.changePage(dir);
};

window.jumpToPage = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.jumpToPage();
};

window.toggleBatchMode = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.toggleBatchMode();
};

window.toggleDataRowSelection = async (id, checked) => {
    const m = await import('../modules/data-list/index.js');
    return m.toggleDataRowSelection(id, checked);
};

window.toggleSelectAllDataRows = async (checked) => {
    const m = await import('../modules/data-list/index.js');
    return m.toggleSelectAllDataRows(checked);
};

window.runDataBatchAnalyze = async () => {
    const m = await import('../modules/data-list/index.js');
    m.runDataBatchAnalyze();
};

window.runDataBatchDelete = async () => {
    const m = await import('../modules/data-list/index.js');
    m.runDataBatchDelete();
};

window.toggleDataTrashView = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.toggleDataTrashView();
};

window.clearSearch = async () => {
    const m = await import('../modules/data-list/index.js');
    return m.clearSearch();
};

// ── 详情面板 ──

window.openDetailsById = async (id) => {
    const m = await import('../modules/details/index.js');
    m.openDetailsById(id);
};

window.closeDetails = async () => {
    const m = await import('../modules/details/index.js');
    return m.closeDetails();
};

window.closeAll = async () => {
    const m = await import('../modules/details/index.js');
    return m.closeAll();
};

window.navigateDetail = async (dir) => {
    const m = await import('../modules/details/index.js');
    m.navigateDetail(dir);
};

window.reanalyzeImage = async () => {
    const m = await import('../modules/details/index.js');
    m.reanalyzeImage();
};

window.openInterpretPanel = async (tab) => {
    const m = await import('../modules/details/index.js');
    return m.openInterpretPanel(tab);
};

window.closeInterpretPanel = async () => {
    const m = await import('../modules/details/index.js');
    return m.closeInterpretPanel();
};

window.toggleInterpretPanel = async () => {
    const m = await import('../modules/details/index.js');
    return m.toggleInterpretPanel();
};

window.openAlgoVisualization = async () => {
    const { openInterpretPanel, loadAlgorithmVisualization } = await import('../modules/details/index.js');
    const { getSteps } = await import('../modules/visualization/index.js');
    const currentItem = getState('currentItem');
    if (!currentItem) {
        alert('请先选择一张图像');
        return;
    }
    if (!getSteps() || getSteps().length === 0) {
        window.reanalyzeImage();
        return;
    }
    openInterpretPanel('algo');
};

// ── AI 对话 ──

window.startAIInterpret = async () => {
    const m = await import('../modules/ai-chat/index.js');
    m.startAIInterpret();
};

window.sendChat = async () => {
    const m = await import('../modules/ai-chat/index.js');
    return m.sendChat();
};

// ── 配置 ──

window.toggleConfigModal = () => {
    const modal = getEl('config-modal');
    if (modal) {
        modal.classList.toggle('hidden');
    }
};

// ── RAG ──

window.loadRagLinks = async () => {
    const m = await import('../modules/rag/index.js');
    return m.loadRagLinks();
};

window.uploadPDF = async (input) => {
    const m = await import('../modules/rag/index.js');
    return m.uploadPDF(input);
};

window.deleteRagDoc = async (id, btn) => {
    const docs = getState('rag.documents') || [];
    const doc = docs.find(d => d.id === id);
    const m = await import('../modules/rag/index.js');
    m.deleteRagDoc(id, doc?.filename || '');
};

window.switchRagSubPage = async (page) => {
    const m = await import('../modules/rag/index.js');
    return m.switchRagSubPage(page);
};

window.setRagGraphFilter = async (filter) => {
    const m = await import('../modules/rag/index.js');
    return m.setRagGraphFilter(filter);
};

// ── 清洗模块 ──

window.switchStepTab = (btn) => {
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');

    const img = getEl('clean-original-image');
    const b64 = btn.dataset.stepImage;
    const name = btn.dataset.stepName;
    if (img && b64) {
        img.src = b64;
        window.currentLightboxTitle = name;
    }
};

window.openCleanLightbox = (src, title) => {
    const modal = getEl('clean-lightbox');
    const img = getEl('clean-lightbox-image');
    const titleEl = getEl('clean-lightbox-title');
    if (!modal || !img || !src) return;
    img.src = src;
    if (titleEl) titleEl.innerText = title;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
};

window.closeCleanLightbox = () => {
    const modal = getEl('clean-lightbox');
    const img = getEl('clean-lightbox-image');
    if (img) img.removeAttribute('src');
    if (modal) modal.classList.add('hidden');
    document.body.style.overflow = '';
};

window.openCleanOriginal = () => {
    const img = getEl('clean-original-image');
    if (!img || img.classList.contains('hidden')) return;
    const src = img.src;
    const title = window.currentLightboxTitle || '图像放大查看';
    window.openCleanLightbox(src, title);
};

// 清洗模块算法可视化全局入口
window.toggleCleanAlgoPage = () => {
    window.dispatchEvent(new CustomEvent('clean-algo-toggle'));
};

// 模型异常数据加载（供 inline JS 调用）
window.loadModelAnomalyData = async () => {
    const { loadModelAnomalyData } = await import('../modules/data-clean/index.js');
    return loadModelAnomalyData();
};

window.switchCleanBackend = async () => {
    // 空壳兼容
};

window.expandCleanAlgoPhase = () => {};

window.showCleanAlgoStepDetail = () => {};

window.hideCleanStepDetailPanel = () => {};

// ── ML 图表 ──

window.switchMlSubPage = async (page) => {
    const m = await import('../modules/charts/index.js');
    m.switchMlSubPage(page);
};

window.selectMlVizTarget = async (target) => {
    const m = await import('../modules/charts/index.js');
    if (m.selectMlVizTarget) { m.selectMlVizTarget(target); return; }
    // fallback 到 index.html 中的原始实现
    if (typeof window._origSelectMlVizTarget === 'function') window._origSelectMlVizTarget(target);
};

window.onMlVizOptionChange = async () => {
    const m = await import('../modules/charts/index.js');
    if (m.onMlVizOptionChange) { m.onMlVizOptionChange(); return; }
    if (typeof window._origOnMlVizOptionChange === 'function') window._origOnMlVizOptionChange();
};

window.switchMlInfoTab = async (tab) => {
    const m = await import('../modules/charts/index.js');
    if (m.switchMlInfoTab) { m.switchMlInfoTab(tab); return; }
    if (typeof window._origSwitchMlInfoTab === 'function') window._origSwitchMlInfoTab(tab);
};

// ── 启动 ──

export { initApp, bindGlobalEvents };

domReady(initApp);
