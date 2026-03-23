// 简单事件总线，用于模块间通信
class EventBus {
    constructor() {
        this.events = {};
    }

    // 注册事件监听
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }

    // 移除事件监听
    off(event, callback) {
        if (!this.events[event]) return;
        this.events[event] = this.events[event].filter(cb => cb !== callback);
    }

    // 触发事件
    emit(event, data) {
        if (!this.events[event]) return;
        this.events[event].forEach(callback => {
            try {
                callback(data);
            } catch (err) {
                console.error(`Event handler error for "${event}":`, err);
            }
        });
    }

    // 清空所有监听
    clear() {
        this.events = {};
    }

    // 清空指定事件的所有监听
    clearEvent(event) {
        delete this.events[event];
    }
}

// 单例实例
export const EventBusInstance = new EventBus();

// 事件名称常量
export const Events = {
    // 数据相关
    DATA_LOADED: 'data:loaded',
    DATA_REFRESH: 'data:refresh',
    ITEM_SELECTED: 'item:selected',
    ITEM_DESELECTED: 'item:deselected',

    // 分页相关
    PAGE_CHANGED: 'page:changed',
    SORT_CHANGED: 'sort:changed',
    FILTER_CHANGED: 'filter:changed',

    // 批量操作
    BATCH_MODE_TOGGLED: 'batch:mode:toggled',
    BATCH_SELECTION_CHANGED: 'batch:selection:changed',
    BATCH_ACTION_COMPLETED: 'batch:action:completed',

    // 详情面板
    DETAILS_OPENED: 'details:opened',
    DETAILS_CLOSED: 'details:closed',

    // AI 相关
    AI_INTERPRET_START: 'ai:interpret:start',
    AI_INTERPRET_COMPLETE: 'ai:interpret:complete',
    AI_MESSAGE_RECEIVED: 'ai:message:received',

    // 页面导航
    PAGE_NAVIGATED: 'page:navigated',
    SIDEBAR_TOGGLED: 'sidebar:toggled',

    // API 错误
    API_ERROR: 'api:error',
    API_SUCCESS: 'api:success',

    // RAG 相关
    RAG_DOCS_LOADED: 'rag:docs:loaded',
    RAG_SEARCH_COMPLETED: 'rag:search:completed',
    RAG_LINKS_LOADED: 'rag:links:loaded',

    // 图表
    CHART_RESIZE: 'chart:resize',
    CHART_DATA_UPDATED: 'chart:data:updated',

    // 数据清洗
    CLEAN_ITEM_SELECTED: 'clean:item:selected',
    CLEAN_DATA_LOADED: 'clean:data:loaded',
};

// 便捷方法
export const on = (event, callback) => EventBusInstance.on(event, callback);
export const off = (event, callback) => EventBusInstance.off(event, callback);
export const emit = (event, data) => EventBusInstance.emit(event, data);
export const clear = () => EventBusInstance.clear();
export const clearEvent = (event) => EventBusInstance.clearEvent(event);
