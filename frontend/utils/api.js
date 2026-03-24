// API 请求工具

import { API_BASE } from '../core/constants.js';
import { emit, Events } from '../core/events.js';

/**
 * 基础请求函数
 */
async function request(url, options = {}) {
    const fullUrl = url.startsWith('http') ? url : `${API_BASE}${url}`;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    try {
        const response = await fetch(fullUrl, { ...defaultOptions, ...options });

        // 自动解析 JSON
        let data;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            data = await response.text();
        }

        // 触发事件
        if (!response.ok) {
            emit(Events.API_ERROR, { url: fullUrl, status: response.status, data });
            throw new Error(data?.detail || data?.message || `HTTP ${response.status}`);
        }

        emit(Events.API_SUCCESS, { url: fullUrl, data });
        return data;
    } catch (err) {
        emit(Events.API_ERROR, { url: fullUrl, error: err });
        throw err;
    }
}

/**
 * GET 请求
 */
export async function get(url, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const fullUrl = queryString ? `${url}?${queryString}` : url;
    return request(fullUrl);
}

/**
 * POST 请求
 */
export async function post(url, data = {}) {
    return request(url, {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

/**
 * PUT 请求
 */
export async function put(url, data = {}) {
    return request(url, {
        method: 'PUT',
        body: JSON.stringify(data),
    });
}

/**
 * DELETE 请求
 */
export async function del(url) {
    return request(url, { method: 'DELETE' });
}

/**
 * POST 表单数据（用于文件上传）
 */
export async function postForm(url, formData) {
    const fullUrl = url.startsWith('http') ? url : `${API_BASE}${url}`;

    try {
        const response = await fetch(fullUrl, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data?.detail || `HTTP ${response.status}`);
        }
        return data;
    } catch (err) {
        console.error('Form upload error:', err);
        throw err;
    }
}

/**
 * 导出 API 对象（按模块分组）
 */
export const api = {
    // 图像相关
    images: {
        list: (params) => get('/api/images', params),
        get: (id) => get(`/api/images/${id}`),
        update: (id, data) => put(`/api/images/${id}`, data),
        delete: (id) => del(`/api/images/${id}`),
        analyze: (id, backend = 'wcntsegnet') => post(`/api/images/${id}/analyze?backend=${encodeURIComponent(backend)}`),
        features: (id, data) => put(`/api/images/${id}/features`, data),
        visualize: (id, backend = 'wcntsegnet') => get(`/api/images/${id}/visualize`, { backend }),
        interpret: (id, data) => post(`/api/images/${id}/interpret`, data),
        batch: {
            analyze: (imageIds, backend = 'wcntsegnet') => post('/api/images/batch/analyze', { image_ids: imageIds, backend }),
            delete: (imageIds) => put('/api/images/batch/delete', { image_ids: imageIds }),
        },
    },

    // 摘要
    summary: () => get('/api/summary'),

    // ML 相关
    ml: {
        simpleModel: (params) => get('/api/ml/xr/simple-model', params),
    },

    // 聊天
    chat: (data) => post('/api/chat', data),

    // RAG 相关
    rag: {
        documents: () => get('/api/rag/documents'),
        upload: (file) => {
            const formData = new FormData();
            formData.append('file', file);
            return postForm('/api/rag/upload', formData);
        },
        deleteDoc: (id) => del(`/api/rag/documents/${id}`),
        search: (query) => post('/api/rag/search', query),
        links: (query) => post('/api/rag/links', query),
        stats: () => get('/api/rag/stats'),
    },
};

export default request;
