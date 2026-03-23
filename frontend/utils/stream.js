// SSE (Server-Sent Events) 流式响应处理工具

/**
 * 解析 SSE 数据行
 * SSE 格式示例：
 * data: {"message": "Hello"}
 * data: {"message": "World"}
 */
export function parseSSELine(line) {
    if (!line || line.startsWith(':')) return null; // 注释行

    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) return null;

    const field = line.slice(0, colonIndex).trim();
    let value = line.slice(colonIndex + 1).trim();

    return { field, value };
}

/**
 * 消费 SSE JSON 流
 * @param {Response} response - fetch 返回的 Response 对象
 * @param {Function} onChunk - 每接收到一个数据块时的回调 (data) => void
 * @param {Function} onComplete - 流结束时的回调 () => void
 * @param {Function} onError - 错误时的回调 (error) => void
 * @returns {Promise<void>}
 */
export async function consumeSseJsonStream(response, onChunk, onComplete, onError) {
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                // 处理缓冲区剩余数据
                if (buffer.trim()) {
                    tryParseAndEmit(buffer, onChunk);
                }
                if (onComplete) onComplete();
                break;
            }

            // 解码并添加到缓冲区
            buffer += decoder.decode(value, { stream: true });

            // 按行分割处理
            const lines = buffer.split('\n');
            // 保留最后一个可能不完整的行
            buffer = lines.pop() || '';

            for (const line of lines) {
                tryParseAndEmit(line, onChunk);
            }
        }
    } catch (err) {
        if (onError) {
            onError(err);
        } else {
            throw err;
        }
    } finally {
        reader.releaseLock();
    }
}

/**
 * 尝试解析并发射 SSE 数据
 */
function tryParseAndEmit(line, onChunk) {
    const parsed = parseSSELine(line);
    if (!parsed) return;

    if (parsed.field === 'data') {
        // SSE data 字段可能包含 JSON
        try {
            const json = JSON.parse(parsed.value);
            if (onChunk) onChunk(json);
        } catch {
            // 不是 JSON，直接发送原始字符串
            if (onChunk) onChunk(parsed.value);
        }
    }
}

/**
 * 创建 SSE 数据源（用于手动触发）
 */
export function createSSEEmitter() {
    let listeners = [];

    return {
        /**
         * 添加监听器
         */
        on(listener) {
            listeners.push(listener);
            return () => {
                listeners = listeners.filter(l => l !== listener);
            };
        },
        /**
         * 发射数据
         */
        emit(data) {
            listeners.forEach(listener => {
                try {
                    listener(data);
                } catch (err) {
                    console.error('SSE listener error:', err);
                }
            });
        },
        /**
         * 清空所有监听器
         */
        clear() {
            listeners = [];
        },
    };
}

/**
 * 模拟流式响应（用于测试或本地生成）
 * @param {Array} chunks - 数据块数组
 * @param {Function} onChunk - 每个数据块的回调
 * @param {number} delay - 每块之间的延迟（毫秒）
 */
export async function simulateStream(chunks, onChunk, delay = 100) {
    for (const chunk of chunks) {
        await new Promise(resolve => setTimeout(resolve, delay));
        if (onChunk) onChunk(chunk);
    }
}

/**
 * 流式文本累加器
 * 用于将多个文本块合并成一个完整文本
 */
export class StreamAccumulator {
    constructor() {
        this.text = '';
    }

    /**
     * 添加文本块
     */
    append(chunk) {
        if (chunk == null) return this;
        this.text += String(chunk);
        return this;
    }

    /**
     * 获取累积的文本
     */
    getText() {
        return this.text;
    }

    /**
     * 清空
     */
    clear() {
        this.text = '';
        return this;
    }

    /**
     * 获取当前长度
     */
    getLength() {
        return this.text.length;
    }
}

/**
 * 创建 SSE 请求
 * @param {string} url - 请求 URL
 * @param {Object} options - fetch 选项
 * @param {Function} onMessage - 消息回调
 * @returns {Promise<{ done: Promise, abort: Function }>}
 */
export function fetchSSE(url, options = {}, onMessage) {
    const controller = new AbortController();
    const { signal } = controller;

    const donePromise = fetch(url, { ...options, signal })
        .then(response => consumeSseJsonStream(
            response,
            onMessage,
            () => {},
            err => {
                if (err.name !== 'AbortError') {
                    console.error('SSE fetch error:', err);
                }
            }
        ))
        .catch(err => {
            if (err.name !== 'AbortError') {
                throw err;
            }
        });

    return {
        done: donePromise,
        abort: () => controller.abort(),
    };
}
