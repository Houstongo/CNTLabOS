# 🧪 测试验证指南 - 前端模块化重构验证

## 📋 后端状态
- ✅ **后端已启动**：`http://127.0.0.1:8000`
- ✅ **虚拟环境已激活**：lab_agent

## 🧪 测试目的
验证模块化前端代码是否：
1. 页面能否正常加载
2. 模块引用是否有效
3. 原有功能是否继续工作

## 🧪 测试清单

### 阶段 1：基础架构验证

#### ✅ 模块加载验证
```javascript
// 在浏览器控制台运行
// 1. 检查模块加载
window.addEventListener('load', async () => {
    // 尝试导入核心模块
    try {
        const [constants, events, store] = await Promise.all([
        import('./frontend/core/constants.js'),
        import('./frontend/core/events.js'),
        import('./frontend/core/store.js'),
    ]);
        console.log('✅ 核心模块加载成功', { constants, events, store });
    } catch (err) {
        console.error('❌ 核心模块加载失败:', err);
    }

    // 2. 检查工具模块
    try {
        const [dom, format, stream, api] = await Promise.all([
            import('./frontend/utils/dom.js'),
            import('./frontend/utils/format.js'),
            import('./frontend/utils/stream.js'),
            import('./frontend/utils/api.js'),
        ]);
        console.log('✅ 工具模块加载成功', { dom, format, stream, api });
    } catch (err) {
        console.error('❌ 工具模块加载失败:', err);
    }
});
```

**预期结果**：所有模块无错误
```

#### ✅ DOM 结构验证
```javascript
// 检查关键 DOM 元素是否存在
const requiredElements = [
    'sidebar',
    'main-container',
    'data-page',
    'ml-page',
    'clean-page',
    'rag-page',
    'details-panel',
    'interpret-panel',
    'config-modal',
];

const missing = requiredElements.filter(id => !document.getElementById(id));
if (missing.length > 0) {
    console.error('❌ 缺失的 DOM 元素:', missing);
} else {
    console.log('✅ 所有关键 DOM 元素都存在');
}
```

**预期结果**：所有关键元素都存在
```

#### ✅ 命名空间验证
```javascript
// 检查全局命名空间
console.log('window.CNTA:', typeof window.CNTA);
console.log('window.CNTA.sidebar:', typeof window.CNTA.sidebar);
console.log('window.CNTA.dataList:', typeof window.CNTA.dataList);
// 预期
// window.CNTA 对象存在
// window.CNTA.sidebar 有函数
// window.CNTA.dataList 有函数
// window.CNTA.charts 有函数
// window.CNTA.aiChat 有函数
// window.CNTA.rag 有函数
```

**预期结果**：所有命名空间都存在
```

### 阶段 2：页面加载验证

#### 1️⃣ 打开首页 `http://127.0.0.1:8000`

#### 2️⃣ 检查浏览器控制台
按 `F12` 打开开发者工具
查看是否有以下错误信息：

**预期**：
- ✅ 无红色错误
- ✅ 无 404 错误（`ERR_CONNECTION_REFUSED` - 服务器连接问题）
```

#### 3️⃣ 检查网络面板
打开 Network 标签，查看：
```
DNS解析失败
请求失败 (status 502, 404)
待定中
```

**预期**：`status: 200 OK`，响应时间 < 500ms`

---

### 阶段 3：功能测试

#### 1️⃣ 测试侧边栏
- [ ] 切换折叠/展开侧边栏
- [ ] 点击"实验原始数据库"
- [ ] 点击"机器学习预测"
- [ ] 点击"数据清洗"
- [ ] 点击"文献知识库"

**预期**：
- 侧边栏宽度变化（260px ↔ 68px）
- 页面正常切换
- 无 JavaScript 错误

#### 2️⃣ 测试数据列表
- [ ] 页面正常加载
- [ ] 点击"批量管理"按钮
- [ ] 复选几行
- [ ] 切换页码

**预期**：
- 无 JavaScript 错误
- 表格正常显示
- 批量操作按钮可用
- 分页功能正常

#### 3️⃣ 测试图表
- [ ] 打开"机器学习预测"
- [ ] 查看 3D 散点图
- [ ] 查看趋势图

**预期**：
- ECharts 图表显示
- 数据点可点击
- 无 JavaScript 错误
```

#### 4️⃣ 测试 AI 对话
- [ ] 点击"生成 AI 解读"
- [ ] 检查模型选择下拉框是否显示
- [ ] 输入消息并发送

**预期**：
- 对话气泡正常显示
- 消息流式输出
- 模型选择器可用
```

#### 5️⃣ 测试 RAG 知识库
- [ ] 打开"文献知识库"
- [ ] 查看 KPI 统计
- [ ] 搜索关键词

**预期**：
- KPI 卡片显示
- 搜索结果更新
- 文档列表显示
```

#### 6️⃣ 测试数据清洗
- [ ] 打开"数据清洗"
- [ ] 查看清洗评估结果
- [ ] 切换高/中/低/删除状态

**预期**：
- 图像预览
- 清洗评分
- 操作按钮工作

#### 7️⃣ 测试详情面板
- [ ] 在列表中点击某一行
- [ ] 查看图像
- [ ] 查看"生成 AI 解读"按钮

**预期**：
- 详情面板打开
- 图像显示
- 特征值显示
- "生成 AI 解读"按钮可用
```

---

## 📊 测试检查清单

### 基础架构
- [ ] 核心模块加载（无错误）
- [ ] 工具模块加载（无错误）
- [ ] CSS 文件加载（Network 中所有 200 OK）
- [ ] 事件总线正常工作

### 功能测试
- [ ] 侧边栏展开/折叠
- [ ] 导航切换
- [ ] 列表加载和分页
- [ ] 数据列表排序/筛选
- [ ] 详情面板打开/关闭
- [ ] AI 对话发送/接收

### 兼容性验证
- [ ] 无内联代码保留（原功能不变）
- [ ] 原有全局函数可正常调用（`window.CNTA.xxx`）
- [ ] 模块化代码不破坏原功能

### ⚠️ 常见问题

**问题**：控制台报错 `loadData is not defined`**
- **原因**：`core/app.js` 中未正确导入 `loadData` 函数

**修复**：检查 `core/app.js` 第 17-18 行：
```javascript
const { loadData } = await import('../modules/data-list/index.js');
```

**问题**：全局事件无响应**
- **原因**：`bindGlobalEvents` 未绑定或被覆盖

**修复**：检查 `core/app.js` 第 18-33 行：
```javascript
    window.addEventListener('data-row-click', async (e) => {
        const { openDetailsById } = await import('../modules/details/index.js');
        openDetailsById(e.detail.id);
    });
```

---

## 🎯 如果测试通过

### ✅ **所有基础通过** → 继续迁移 Data Clean 模块
### ⚠️ **失败** → 暂停并修复后重试

## 📝 下一步

测试通过后，如果一切正常，我将进行：
1. ✅ 标记任务 #14（Data Clean）为完成
2. ✅ 标记任务 #15（AI Chat）为完成
3. ✅ 标记任务 #16（Details）为完成
4. ✅ 标记任务 #18（Charts）为完成
5. � 清除临时测试页面 `test-modules.html`

## 🔧 问题排查

### 如果出现空白页面
**检查**：
1. 浏览器控制台是否有 404 错误（`ERR_CONNECTION_REFUSED`）
2. 网络面板是否有失败

**修复**：
1. 确认后端是否运行：`python manage.py run-backend`
2. 刷新页面

### 如果功能不工作
**检查**：
1. 控制台是否有 JavaScript 错误
2. `window.CNTA` 是否有值

**修复**：
1. 按 F12 强制刷新
2. 清空 localStorage 缓存

---

**请先进行测试并告诉我结果！**