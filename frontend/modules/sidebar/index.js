// Sidebar 模块 - 侧边栏控制

import { getEl } from '../../utils/dom.js';
import { emit, Events } from '../../core/events.js';

/**
 * 切换侧边栏展开/折叠状态
 */
export function toggleSidebar() {
    const side = getEl('sidebar');
    if (!side) return;

    const wasCollapsed = side.classList.contains('collapsed');
    side.classList.toggle('collapsed');

    const isCollapsed = side.classList.contains('collapsed');
    document.body.classList.toggle('sb-collapsed', isCollapsed);

    emit(Events.SIDEBAR_TOGGLED, { collapsed: isCollapsed });
}

/**
 * 设置 ML 聚焦模式
 * 当切换到 ML 页面时，隐藏常规头部，只保留 ML 控件
 */
export function setMlFocusMode(enabled) {
    const titleBlock = getEl('header-title-block');
    const actionBlock = getEl('header-actions-block');
    const sourceFilter = getEl('source-filter');
    const mlSwitchTop = getEl('ml-subpage-switch-top');

    if (!titleBlock || !actionBlock) return;

    titleBlock.classList.toggle('hidden', !!enabled);
    actionBlock.classList.toggle('w-full', !!enabled);
    actionBlock.classList.toggle('justify-end', !!enabled);

    if (sourceFilter) {
        sourceFilter.classList.toggle('hidden', !!enabled);
    }
    if (mlSwitchTop) {
        mlSwitchTop.classList.toggle('hidden', !enabled);
    }
}

/**
 * 设置 RAG 头部模式
 * RAG 页面使用自定义头部样式
 */
export function setRagHeaderMode(enabled) {
    const mainHeader = getEl('main-header');
    const titleBlock = getEl('header-title-block');
    const actionBlock = getEl('header-actions-block');
    const progressText = getEl('progress-text');

    if (!mainHeader || !titleBlock || !actionBlock) return;

    if (enabled) {
        mainHeader.classList.add('hidden');
        titleBlock.classList.remove('hidden');
        actionBlock.classList.add('hidden');
        if (progressText) {
            progressText.innerText = '知识库模式 · 展示当前可验证成果';
        }
    } else {
        mainHeader.classList.remove('hidden');
        actionBlock.classList.remove('hidden');
        if (progressText && progressText.innerText.includes('知识库模式')) {
            progressText.innerText = '正在检索本地数据库...';
        }
    }
}

/**
 * 设置数据源过滤
 * @param {string} val - 数据源值 ('', 'XR', 'ZZY')
 * @param {HTMLElement} triggerEl - 触发元素（用于设置 active 样式）
 */
export function setSource(val, triggerEl = null) {
    const sourceFilter = getEl('source-filter');
    if (sourceFilter) {
        sourceFilter.value = val;
    }

    // 更新 UI 激活状态
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const activeEl = triggerEl || window.event?.currentTarget || null;
    if (activeEl) {
        activeEl.classList.add('active');
    }

    // 确保显示数据页、隐藏其它页
    const dataPage = getEl('data-page');
    const mlPage = getEl('ml-page');
    const cleanPage = getEl('clean-page');
    const ragPage = getEl('rag-page');

    if (dataPage) dataPage.classList.remove('hidden');
    if (mlPage) mlPage.classList.add('hidden');
    if (cleanPage) cleanPage.classList.add('hidden');
    if (ragPage) ragPage.classList.add('hidden');

    setMlFocusMode(false);

    emit(Events.FILTER_CHANGED, { type: 'source', value: val });
}

/**
 * 页面导航
 * @param {string} name - 页面名称 ('data', 'ml', 'clean', 'rag')
 * @param {HTMLElement} el - 触发元素
 */
export function showPage(name, el) {
    const mainHeader = getEl('main-header');
    const dataPage = getEl('data-page');
    const mlPage = getEl('ml-page');
    const predictPage = getEl('predict-page');
    const knowledgePage = getEl('knowledge-page');
    const cleanPage = getEl('clean-page');
    const ragPage = getEl('rag-page');
    const tccerPage = getEl('tccer-page');
    const qaPage = getEl('qa-page');
    const mlSubSwitchTop = getEl('ml-subpage-switch-top');
    const sourceFilter = getEl('source-filter');

    // 更新导航激活状态
    document.querySelectorAll('.nav-item').forEach(e => e.classList.remove('active'));
    if (el) el.classList.add('active');

    // 根据页面名称显示对应内容
    switch (name) {
        case 'predict':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (predictPage) predictPage.classList.remove('hidden');
            if (knowledgePage) knowledgePage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (tccerPage) tccerPage.classList.add('hidden');
            if (qaPage) qaPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;

        case 'knowledge':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (predictPage) predictPage.classList.add('hidden');
            if (knowledgePage) knowledgePage.classList.remove('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (tccerPage) tccerPage.classList.add('hidden');
            if (qaPage) qaPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;

        case 'rag':
            // 已弃用，重定向到 knowledge 页面
            showPage('knowledge', el);
            return;

        case 'tccer':
            // 已弃用，重定向到 knowledge 页面
            showPage('knowledge', el);
            return;

        case 'qa':
            // 已弃用，重定向到 knowledge 页面
            showPage('knowledge', el);
            return;

        case 'ml':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.remove('hidden');
            if (predictPage) predictPage.classList.add('hidden');
            if (knowledgePage) knowledgePage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (tccerPage) tccerPage.classList.add('hidden');
            if (qaPage) qaPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.remove('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(true);
            setRagHeaderMode(false);
            break;

        case 'clean':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (predictPage) predictPage.classList.add('hidden');
            if (knowledgePage) knowledgePage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.remove('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (tccerPage) tccerPage.classList.add('hidden');
            if (qaPage) qaPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;

        case 'data':
        default:
            if (dataPage) dataPage.classList.remove('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (predictPage) predictPage.classList.add('hidden');
            if (knowledgePage) knowledgePage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (tccerPage) tccerPage.classList.add('hidden');
            if (qaPage) qaPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            if (sourceFilter) sourceFilter.classList.remove('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;
    }

    emit(Events.PAGE_NAVIGATED, { page: name });
}

/**
 * 初始化侧边栏
 */
export function initSidebar() {
    // 绑定导航项点击事件
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const pageName = item.getAttribute('data-page') || 'data';
        item.addEventListener('click', () => {
            showPage(pageName, item);
        });
    });
}

// 导出默认对象
export default {
    toggleSidebar,
    setMlFocusMode,
    setRagHeaderMode,
    setSource,
    showPage,
    initSidebar,
};
