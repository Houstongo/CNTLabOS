// DOM 工具函数

/**
 * 转义 HTML 防止 XSS
 */
export function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * 查询单个元素 (querySelector 简写)
 */
export const qs = (selector, parent = document) => parent.querySelector(selector);

/**
 * 查询多个元素 (querySelectorAll 简写)
 */
export const qsa = (selector, parent = document) => Array.from(parent.querySelectorAll(selector));

/**
 * 获取元素
 */
export const getEl = (id) => document.getElementById(id);

/**
 * 创建元素
 */
export function createEl(tag, attributes = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
            el.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(el.style, value);
        } else if (key.startsWith('data-')) {
            el.dataset[key.slice(5)] = value;
        } else if (value != null) {
            el.setAttribute(key, value);
        }
    });
    children.forEach(child => {
        if (typeof child === 'string') {
            el.appendChild(document.createTextNode(child));
        } else if (child instanceof Node) {
            el.appendChild(child);
        }
    });
    return el;
}

/**
 * 移除元素
 */
export function removeEl(selectorOrEl) {
    const el = typeof selectorOrEl === 'string' ? qs(selectorOrEl) : selectorOrEl;
    if (el && el.parentElement) {
        el.parentElement.removeChild(el);
    }
}

/**
 * 显示/隐藏元素
 */
export function toggleEl(elOrId, show) {
    const el = typeof elOrId === 'string' ? getEl(elOrId) : elOrId;
    if (!el) return;
    if (show == null) {
        el.classList.toggle('hidden');
    } else if (show) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

/**
 * 添加/移除类
 */
export function toggleClass(elOrId, className, force) {
    const el = typeof elOrId === 'string' ? getEl(elOrId) : elOrId;
    if (!el) return;
    el.classList.toggle(className, force);
}

/**
 * 检查元素是否在视口内
 */
export function isInViewport(el) {
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * 平滑滚动到元素
 */
export function scrollToEl(elOrId, offset = 0) {
    const el = typeof elOrId === 'string' ? qs(elOrId) : elOrId;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    window.scrollTo({
        top: rect.top + scrollTop - offset,
        behavior: 'smooth',
    });
}

/**
 * 防抖函数
 */
export function debounce(fn, delay = 300) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/**
 * 节流函数
 */
export function throttle(fn, delay = 100) {
    let lastTime = 0;
    return function (...args) {
        const now = Date.now();
        if (now - lastTime >= delay) {
            lastTime = now;
            fn.apply(this, args);
        }
    };
}

/**
 * 等待 DOM 加载完成
 */
export function domReady(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback);
    } else {
        callback();
    }
}

/**
 * 从 URL 查询参数获取值
 */
export function getQueryParam(name) {
    const params = new URLSearchParams(window.location.search);
    return params.get(name);
}

/**
 * 设置 URL 查询参数
 */
export function setQueryParam(name, value) {
    const url = new URL(window.location);
    if (value === null || value === undefined) {
        url.searchParams.delete(name);
    } else {
        url.searchParams.set(name, String(value));
    }
    window.history.replaceState({}, '', url);
}

/**
 * 等待元素出现
 */
export function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const el = qs(selector);
        if (el) {
            resolve(el);
            return;
        }
        const observer = new MutationObserver((mutations) => {
            const found = qs(selector);
            if (found) {
                observer.disconnect();
                resolve(found);
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        const timer = setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Element "${selector}" not found within ${timeout}ms`));
        }, timeout);
    });
}

/**
 * 将 DOM 元素修复到 body
 */
export function repairOverlayMounts(elementIds = ['details-panel', 'interpret-panel', 'config-modal', 'clean-lightbox']) {
    elementIds.forEach(id => {
        const el = getEl(id);
        if (el && el.parentElement !== document.body) {
            document.body.appendChild(el);
        }
    });
}
