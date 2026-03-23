// localStorage 封装，带类型安全和默认值

const PREFIX = 'CNTA_';

/**
 * 设置值（自动序列化）
 */
export function set(key, value) {
    try {
        if (value == null) {
            remove(key);
            return;
        }
        const fullKey = PREFIX + key;
        localStorage.setItem(fullKey, JSON.stringify(value));
    } catch (err) {
        console.error(`localStorage set error for "${key}":`, err);
    }
}

/**
 * 获取值（自动反序列化）
 */
export function get(key, defaultValue = null) {
    try {
        const fullKey = PREFIX + key;
        const value = localStorage.getItem(fullKey);
        if (value == null) return defaultValue;
        return JSON.parse(value);
    } catch (err) {
        console.error(`localStorage get error for "${key}":`, err);
        return defaultValue;
    }
}

/**
 * 删除值
 */
export function remove(key) {
    try {
        const fullKey = PREFIX + key;
        localStorage.removeItem(fullKey);
    } catch (err) {
        console.error(`localStorage remove error for "${key}":`, err);
    }
}

/**
 * 检查键是否存在
 */
export function has(key) {
    const fullKey = PREFIX + key;
    return localStorage.getItem(fullKey) != null;
}

/**
 * 获取所有 CNTA 前缀的键
 */
export function keys() {
    return Object.keys(localStorage)
        .filter(k => k.startsWith(PREFIX))
        .map(k => k.slice(PREFIX.length));
}

/**
 * 清空所有 CNTA 前缀的数据
 */
export function clear() {
    keys().forEach(key => remove(key));
}

/**
 * 获取并删除值（pop 操作）
 */
export function pop(key, defaultValue = null) {
    const value = get(key, defaultValue);
    remove(key);
    return value;
}

/**
 * 批量设置
 */
export function setMany(items) {
    Object.entries(items).forEach(([key, value]) => set(key, value));
}

/**
 * 批量获取
 */
export function getMany(keyArray, defaultValueMap = {}) {
    const result = {};
    keyArray.forEach(key => {
        result[key] = get(key, defaultValueMap[key]);
    });
    return result;
}

/**
 * 存储空间检查
 */
export function getStorageInfo() {
    let total = 0;
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith(PREFIX)) {
            total += localStorage.getItem(key).length * 2; // UTF-16 编码，每个字符 2 字节
        }
    }
    return {
        used: total,
        usedHuman: formatBytes(total),
        // localStorage 通常限制 5-10MB
        limit: 5 * 1024 * 1024,
        limitHuman: '5MB',
    };
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

/**
 * 专用键名常量
 */
export const StorageKeys = {
    PROVIDER: 'PROVIDER',
    // GLM
    GLM_KEY: 'KEY_glm',
    GLM_MODEL: 'MODEL_glm',
    // DeepSeek
    DEEPSEEK_KEY: 'KEY_deepseek',
    DEEPSEEK_MODEL: 'MODEL_deepseek',
};

/**
 * AI 相关的便捷方法
 */
export const aiStorage = {
    getProvider: () => get(StorageKeys.PROVIDER, 'glm'),
    setProvider: (provider) => set(StorageKeys.PROVIDER, provider),
    getKey: (provider) => get(`KEY_${provider}`, ''),
    setKey: (provider, key) => set(`KEY_${provider}`, key),
    getModel: (provider) => get(`MODEL_${provider}`, ''),
    setModel: (provider, model) => set(`MODEL_${provider}`, model),
};
