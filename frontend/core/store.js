// 全局状态管理 Store
// 使用路径点号语法访问：'data.offset', 'currentItem', 'ml.vizState.target'

class Store {
    constructor() {
        // 初始状态
        this.state = {
            // 数据列表相关
            data: {
                offset: 0,
                limit: 10,
                totalItems: 0,
                currentSort: 'id',
                currentOrder: 'desc',
                currentListItemsById: {},
                selectedDataIds: new Set(),
                batchMode: false,
                viewMode: 'active', // 'active' | 'deleted'
            },
            // 当前选中的项目
            currentItem: null,
            // 聊天历史
            chatHistory: [],
            // ML 相关
            ml: {
                payload: null,
                subPage: 'visual', // 'visual' | 'data'
                vizState: {
                    target: 'diameter',
                    mode: '2d',
                    xKey: 'actual_temp',
                    yMode: 'pred',
                    xKey3d: 'actual_temp',
                    yKey3d: 'flow_rate',
                    infoTab: 'coef',
                    mainPoints: [],
                    selectedPoint: null,
                },
                chartInstances: {},
            },
            // 数据清洗
            clean: {
                items: [],
                filteredItems: [],
                selectedId: null,
                view: 'active',
            },
            // RAG
            rag: {
                currentChain: null,
                currentQuery: '',
                graphFilter: 'all',
                allStats: null,
                currentSubgraph: null,
                currentConstrainedChain: null,
                currentThemeAggregation: null,
                documents: [],
            },
            // 算法可视化
            algorithm: {
                steps: null,
                currentStepIndex: 0,
            },
            // AI 配置缓存
            aiConfig: {
                tempConfigCache: {},
            },
        };

        // 订阅者
        this.subscribers = new Map();
    }

    /**
     * 获取状态值
     * @param {string} path - 点号分隔的路径，如 'data.offset'
     * @returns {any}
     */
    get(path) {
        if (!path) return this.state;
        const keys = path.split('.');
        let value = this.state;
        for (const key of keys) {
            if (value == null || typeof value !== 'object') return undefined;
            value = value[key];
        }
        return value;
    }

    /**
     * 设置状态值
     * @param {string} path - 点号分隔的路径
     * @param {any} value - 新值
     */
    set(path, value) {
        const keys = path.split('.');
        let target = this.state;
        // 遍历到倒数第二层
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            if (!(key in target) || typeof target[key] !== 'object') {
                target[key] = {};
            }
            target = target[key];
        }
        const lastKey = keys[keys.length - 1];
        const oldValue = target[lastKey];
        target[lastKey] = value;

        // 触发订阅者
        this._notify(path, value, oldValue);
    }

    /**
     * 批量设置状态
     * @param {Object} updates - { 'data.offset': 0, 'currentItem': item }
     */
    setMultiple(updates) {
        Object.entries(updates).forEach(([path, value]) => {
            this.set(path, value);
        });
    }

    /**
     * 更新状态中的某个属性（用于对象或数组）
     * @param {string} path - 如 'data.selectedDataIds'
     * @param {Function} updater - 函数接收当前值返回新值
     */
    update(path, updater) {
        const currentValue = this.get(path);
        const newValue = updater(currentValue);
        this.set(path, newValue);
    }

    /**
     * 订阅状态变化
     * @param {string} path - 路径，支持通配符 'data.*'
     * @param {Function} callback - 回调函数 (newValue, oldValue, path) => void
     * @returns {Function} 取消订阅函数
     */
    subscribe(path, callback) {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path).add(callback);

        // 返回取消订阅函数
        return () => {
            const subs = this.subscribers.get(path);
            if (subs) {
                subs.delete(callback);
                if (subs.size === 0) {
                    this.subscribers.delete(path);
                }
            }
        };
    }

    /**
     * 通知订阅者
     */
    _notify(changedPath, newValue, oldValue) {
        // 精确匹配
        if (this.subscribers.has(changedPath)) {
            this.subscribers.get(changedPath).forEach(cb => {
                try {
                    cb(newValue, oldValue, changedPath);
                } catch (err) {
                    console.error(`Store subscriber error for "${changedPath}":`, err);
                }
            });
        }

        // 通配符匹配 (data.* 会匹配 data.offset)
        const wildcardPath = changedPath.split('.').slice(0, -1).join('') + '.*';
        const directWildcard = changedPath.substring(0, changedPath.lastIndexOf('.')) + '.*';
        this.subscribers.forEach((subs, pattern) => {
            if (pattern.endsWith('*') && changedPath.startsWith(pattern.slice(0, -1))) {
                subs.forEach(cb => {
                    try {
                        cb(newValue, oldValue, changedPath);
                    } catch (err) {
                        console.error(`Store wildcard subscriber error for "${pattern}":`, err);
                    }
                });
            }
        });
    }

    /**
     * 重置整个状态
     */
    reset() {
        this.state = {
            data: {
                offset: 0,
                limit: 10,
                totalItems: 0,
                currentSort: 'id',
                currentOrder: 'desc',
                currentListItemsById: {},
                selectedDataIds: new Set(),
                batchMode: false,
                viewMode: 'active',
            },
            currentItem: null,
            chatHistory: [],
            ml: {
                payload: null,
                subPage: 'visual',
                vizState: {
                    target: 'diameter',
                    mode: '2d',
                    xKey: 'actual_temp',
                    yMode: 'pred',
                    xKey3d: 'actual_temp',
                    yKey3d: 'flow_rate',
                    infoTab: 'coef',
                    mainPoints: [],
                    selectedPoint: null,
                },
                chartInstances: {},
            },
            clean: {
                items: [],
                filteredItems: [],
                selectedId: null,
                view: 'active',
            },
            rag: {
                currentChain: null,
                currentQuery: '',
                graphFilter: 'all',
                allStats: null,
                currentSubgraph: null,
                currentConstrainedChain: null,
                currentThemeAggregation: null,
                documents: [],
            },
            algorithm: {
                steps: null,
                currentStepIndex: 0,
            },
            aiConfig: {
                tempConfigCache: {},
            },
        };
        this._notify('*', this.state, null);
    }

    /**
     * 获取整个状态的副本
     */
    getState() {
        return JSON.parse(JSON.stringify(this.state));
    }
}

// 单例实例
export const store = new Store();

// 便捷函数
export const getState = (path) => store.get(path);
export const setState = (path, value) => store.set(path, value);
export const updateState = (path, updater) => store.update(path, updater);
export const subscribe = (path, callback) => store.subscribe(path, callback);
export const resetStore = () => store.reset();

// 导出 Store 类供测试使用
export default Store;
