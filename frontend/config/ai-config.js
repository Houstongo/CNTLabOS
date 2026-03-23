// AI 配置管理

/**
 * 模型配置
 */
export const MODEL_CONFIGS = {
    glm: {
        name: 'GLM',
        color: 'purple',
        models: [
            { id: 'glm-4.7-plus', name: 'GLM-4.7 Plus (最新增强)' },
            { id: 'glm-4-plus', name: 'GLM-4 Plus (旗舰版)' },
            { id: 'glm-4-flash', name: 'GLM-4 Flash (极速版)' },
            { id: 'glm-4-air', name: 'GLM-4 Air (平衡版)' },
            { id: 'glm-4.5-air', name: 'GLM-4.5 Air (经典平衡)' },
            { id: 'glm-4.6v', name: 'GLM-4.6v (多模态增强)' },
            { id: 'glm-4', name: 'GLM-4 (标准版)' },
            { id: 'glm-5', name: 'GLM-5.0 (前瞻支持)' },
        ],
        testUrl: 'https://open.bigmodel.cn/api/paas/v4/models',
    },
    deepseek: {
        name: 'DeepSeek',
        color: 'blue',
        models: [
            { id: 'deepseek-chat', name: 'DeepSeek-V3' },
            { id: 'deepseek-reasoner', name: 'DeepSeek-R1 (推理版)' },
        ],
        testUrl: 'https://api.deepseek.com/models',
    },
};

/**
 * 获取当前激活的提供商
 */
export function getActiveProvider() {
    return localStorage.getItem('CNTA_PROVIDER') || 'glm';
}

/**
 * 获取提供商配置
 */
export function getProviderConfig(provider) {
    return MODEL_CONFIGS[provider] || MODEL_CONFIGS.glm;
}

/**
 * 获取 API Key
 */
export function getApiKey(provider = null) {
    const p = provider || getActiveProvider();
    return localStorage.getItem(`CNTA_KEY_${p}`) || '';
}

/**
 * 获取模型 ID
 */
export function getModelId(provider = null) {
    const p = provider || getActiveProvider();
    const config = getProviderConfig(p);
    return localStorage.getItem(`CNTA_MODEL_${p}`) || config.models[0].id;
}

/**
 * 获取 API 请求头
 */
export function getApiHeaders() {
    const provider = getActiveProvider();
    const key = getApiKey(provider);
    const model = getModelId(provider);
    return {
        'X-Provider': provider,
        'X-Api-Key': key,
        'X-Model': model,
    };
}

/**
 * 设置提供商
 */
export function setActiveProvider(provider) {
    if (!MODEL_CONFIGS[provider]) {
        throw new Error(`Unknown provider: ${provider}`);
    }
    localStorage.setItem('CNTA_PROVIDER', provider);
}

/**
 * 设置 API Key
 */
export function setApiKey(provider, key) {
    localStorage.setItem(`CNTA_KEY_${provider}`, key);
}

/**
 * 设置模型
 */
export function setModelId(provider, modelId) {
    localStorage.setItem(`CNTA_MODEL_${provider}`, modelId);
}

/**
 * 测试连接
 */
export async function testConnection(provider, apiKey) {
    const config = getProviderConfig(provider);

    if (!apiKey || apiKey.length < 10) {
        throw new Error('请先输入 API Key');
    }

    try {
        const response = await fetch(config.testUrl, {
            method: 'GET',
            headers: { 'Authorization': `Bearer ${apiKey}` },
        });

        if (!response.ok) {
            const errorBody = await response.text();
            let errorMessage = `HTTP ${response.status}`;
            try {
                const errJson = JSON.parse(errorBody);
                errorMessage = errJson.error?.message || errJson.message || errorMessage;
            } catch {}
            throw new Error(errorMessage);
        }

        return true;
    } catch (err) {
        throw err;
    }
}

/**
 * 获取完整的模型选择选项
 */
export function getModelOptions() {
    const options = [];
    for (const [provider, config] of Object.entries(MODEL_CONFIGS)) {
        const group = {
            label: config.name,
            provider,
            color: config.color,
            models: config.models.map(m => ({
                value: `${provider}:${m.id}`,
                label: m.name,
            })),
        };
        options.push(group);
    }
    return options;
}

/**
 * 临时配置缓存（用于配置模态框）
 */
class TempConfigCache {
    constructor() {
        this.cache = {};
    }

    get(provider) {
        if (!this.cache[provider]) {
            this.cache[provider] = {
                key: getApiKey(provider),
                model: getModelId(provider),
            };
        }
        return this.cache[provider];
    }

    set(provider, key, model) {
        this.cache[provider] = { key, model };
    }

    clear() {
        this.cache = {};
    }

    save() {
        for (const [provider, config] of Object.entries(this.cache)) {
            if (config.key !== undefined) setApiKey(provider, config.key);
            if (config.model !== undefined) setModelId(provider, config.model);
        }
    }
}

export const tempConfigCache = new TempConfigCache();

/**
 * 检查 API 状态
 */
export function checkApiStatus() {
    const provider = getActiveProvider();
    const key = getApiKey(provider);
    const model = getModelId(provider);
    const config = getProviderConfig(provider);

    const isReady = key && key.trim().length > 10;

    return {
        ready: isReady,
        provider: config.name,
        model: model.split('-').pop(),
        color: config.color,
    };
}
