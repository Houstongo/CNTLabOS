// Sidebar 模块 - 命名空间封装
// 模块化引用原 index.html 中的 Sidebar 函数
// 逐步将实现迁移到真正的模块中

import { getEl } from '../../utils/dom.js';
import { emit, Events } from '../../core/events.js';

/**
 * 初始化侧边栏（占位函数，等待真实迁移）
 */
export function initSidebar() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const pageName = item.getAttribute('data-page') || 'data';
        item.addEventListener('click', () => {
            showPage(pageName, item);
        });
    });
}

/**
 * 页面导航 - 占位函数
 */
export function showPage(name, el) {
    const mainHeader = getEl('main-header');
    const dataPage = getEl('data-page');
    const mlPage = getEl('ml-page');
    const cleanPage = getEl('clean-page');
    const ragPage = getEl('rag-page');
    const mlSubSwitchTop = getEl('ml-subpage-switch-top');
    const sourceFilter = getEl('source-filter');

    if (!mainHeader || !dataPage || !mlPage || !cleanPage || !ragPage || !mlSubSwitchTop || !sourceFilter) return;

    document.querySelectorAll('.nav-item').forEach(e => e.classList.remove('active'));
    if (el) el.classList.contains('nav-item')) el.classList.add('active');

    // 根据页面名称显示/隐藏元素
    switch (name) {
        case 'rag':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.remove('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(true);
            break;

        case 'ml':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.remove('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.remove('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(true);
            setRagHeaderMode(false);
            break;

        case 'clean':
            if (dataPage) dataPage.classList.add('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.remove('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.add('hidden');
            if (sourceFilter) sourceFilter.classList.add('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;

        case 'data':
        default:
            if (dataPage) dataPage.classList.remove('hidden');
            if (mlPage) mlPage.classList.add('hidden');
            if (cleanPage) cleanPage.classList.add('hidden');
            if (ragPage) ragPage.classList.add('hidden');
            if (mlSubSwitchTop) mlSubSwitchTop.classList.remove('hidden');
            if (sourceFilter) sourceFilter.classList.remove('hidden');
            setMlFocusMode(false);
            setRagHeaderMode(false);
            break;
    }

    emit(Events.PAGE_NAVIGATED, { page: name });
}

/**
 * 切换侧边栏 - 占位函数
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
 * 设置 ML 聚焦模式 - 占位函数
 */
export function setMlFocusMode(enabled) {
    const titleBlock = getEl('header-title-block');
    const actionBlock = getEl('header-actions-block');
    const sourceFilter = getEl('source-filter');
    const mlSwitchTop = getEl('ml-subpage-switch-top');

    if (!titleBlock || !actionBlock || !sourceFilter || !mlSwitchTop) return;

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
 * 设置 RAG 头部模式 - 占位函数
 */
export function setRagHeaderMode(enabled) {
    const mainHeader = getEl('main-header');
    const titleBlock = getEl('header-title-block');
    const actionBlock = getEl('header-actions-block');
    const progressText = getEl('progress-text');

    if (!mainHeader || !titleBlock || !actionBlock || !progressText) return;

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
 * 设置数据源 - 占位函数
 */
export function setSource(val, triggerEl = null) {
    const sourceFilter = getEl('source-filter');
    if (!sourceFilter) return;

    sourceFilter.value = val;

    // 更新 UI 激活状态
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const activeEl = triggerEl || window.event?.currentTarget || null;
    if (activeEl && activeEl.classList.contains('nav-item')) {
        activeEl.classList.add('active');
    }

    // 确保显示数据页、隐藏其它页
    const dataPage = getEl('data-page');
    const mlPage = getEl('ml-page');
    const cleanPage = getEl('clean-page');
    const ragPage = getEl('rag-page');
    const mlSubSwitchTop = getEl('ml-subpage-switch-top');
    const sourceFilterEl = getEl('source-filter');

    if (dataPage) dataPage.classList.remove('hidden');
    if (mlPage) mlPage.classList.add('hidden');
    if (cleanPage) cleanPage.classList.add('hidden');
    if (ragPage) ragPage.classList.add('hidden');
    if (mlSubSwitchTop) mlSwitchTop.classList.add('hidden');
    setMlFocusMode(false);
    setRagHeaderMode(false);

    emit(Events.FILTER_CHANGED, { type: 'source', value: val });
}

/**
 * 修复覆盖层挂载
 */
export function repairOverlayMounts() {
    ['details-panel', 'interpret-panel', 'config-modal', 'clean-lightbox'].forEach(id => {
        const el = getEl(id);
        if (el && el.parentElement !== document.body) {
            document.body.appendChild(el);
        }
    });
}

// 默认导出对象（保持向后兼容）
const Sidebar = {
    toggleSidebar,
    setMlFocusMode,
    setRagHeaderMode,
    setSource,
    showPage,
    initSidebar,
};

// 兼容原 global函数暴露
window.CNTA = {
    ...Sidebar,

    // 占位函数，待模块完成后替换
    toggleSidebar: () => {
        const { toggleSidebar } = require('./index.js');
        return toggleSidebar();
    },
    setMlFocusMode: (enabled) => {
        const { setMlFocusMode } = require('./index.js');
        return setMlFocusMode(enabled);
    },
    setRagHeaderMode: (enabled) => {
        const { setRagHeaderMode } = require('./index.js');
        return setRagHeaderMode(enabled);
    },
    setSource: (val, triggerEl) => {
        const { setSource } = require('./index.js');
        return setSource(val, triggerEl);
    },
    showPage: (name, el) => {
        const { showPage } = require('./index.js');
        return showPage(name, el);
    },
    initSidebar: () => {
        const { initSidebar } = require('./index.js');
        return initSidebar();
    },
};

export default Sidebar;
