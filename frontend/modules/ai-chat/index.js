// AI Chat 模块 - AI 对话与解释

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { consumeSseJsonStream } from '../../utils/stream.js';
import { getApiHeaders } from '../../config/ai-config.js';
import { aiStorage } from '../../config/local-storage.js';
import { emit, Events } from '../../core/events.js';
import { API_BASE } from '../../core/constants.js';

/**
 * 清空对话
 */
export function clearChat() {
    setState('chatHistory', []);

    const chatHistoryEl = getEl('chat-history');
    if (chatHistoryEl) {
        chatHistoryEl.innerHTML = `
            <div id="chat-placeholder" class="text-center text-slate-300 mt-16 select-none">
                <i class="fas fa-brain text-4xl mb-3 block"></i>
                <span class="text-sm leading-relaxed">点击"生成 AI 解读"开始分析<br>或直接输入问题与 AI 实验员对话</span>
            </div>
        `;
    }
}

/**
 * 添加对话气泡
 */
export function appendChatBubble(role, text) {
    const history = getEl('chat-history');
    if (!history) return;

    const div = document.createElement('div');
    div.className = role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai interp-md';
    div.innerHTML = text ? (role === 'user' ? escapeHtml(text) : marked.parse(text)) : '';
    history.appendChild(div);
    scrollChatToBottom();
    return div;
}

/**
 * 滚动对话到底部
 */
export function scrollChatToBottom() {
    const el = getEl('chat-history');
    if (el) {
        el.scrollTop = el.scrollHeight;
    }
}

/**
 * HTML 转义
 */
function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * 同步解释面板的模型选择器
 */
function syncInterpretModelSelect() {
    const sel = getEl('interpret-model-select');
    if (!sel) return;

    // 首次调用时动态填充
    if (sel.options.length === 0) {
        const MODEL_CONFIGS = {
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
            },
            deepseek: {
                name: 'DeepSeek',
                color: 'blue',
                models: [
                    { id: 'deepseek-chat', name: 'DeepSeek-V3' },
                    { id: 'deepseek-reasoner', name: 'DeepSeek-R1 (推理版)' },
                ],
            },
        };

        for (const [pid, conf] of Object.entries(MODEL_CONFIGS)) {
            const grp = document.createElement('optgroup');
            grp.label = conf.name;
            for (const m of conf.models) {
                const opt = document.createElement('option');
                opt.value = pid + ':' + m.id;
                opt.textContent = m.name;
                grp.appendChild(opt);
            }
            sel.appendChild(grp);
        }
    }

    // 同步到当前默认 provider + 已保存的模型
    const provider = aiStorage.getProvider();
    const model = aiStorage.getModel(provider) || '';
    const target = provider + ':' + model;

    const exists = [...sel.options].some(o => o.value === target);
    if (exists) {
        sel.value = target;
    } else {
        const fallback = [...sel.options].find(o => o.value.startsWith(provider + ':'));
        if (fallback) sel.value = fallback.value;
    }
}

/**
 * 开始 AI 解释（图像解读）
 */
export async function startAIInterpret() {
    const currentItem = getState('currentItem');
    if (!currentItem) return;

    syncInterpretModelSelect();

    const sel = getEl('interpret-model-select');
    const [provider, model] = (sel?.value || 'glm:glm-4-flash').split(':');
    const key = aiStorage.getKey(provider);

    if (!key || key.length < 11) {
        alert('请先在配置中填写 ' + (provider === 'glm' ? 'GLM' : 'DeepSeek') + ' API Key');
        const { toggleConfigModal } = await import('../../config/ai-config.js');
        toggleConfigModal();
        return;
    }

    const temperature = getEl('ai-temperature')?.value || '0.5';
    const btn = getEl('start-ai-btn');
    if (!btn) return;

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1.5"></i>生成中...';

    // 移除 placeholder，新增 AI 消息气泡
    getEl('chat-placeholder')?.remove();
    const aiEl = appendChatBubble('assistant', '');
    aiEl.classList.add('typing-cursor');
    scrollChatToBottom();

    let fullText = '';
    let chatHistory = getState('chatHistory') || [];

    try {
        emit(Events.AI_INTERPRET_START, { itemId: currentItem.id });

        const res = await fetch(`${API_BASE}/api/images/${currentItem.id}/interpret`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Provider': provider,
                'X-Api-Key': key,
                'X-Model': model,
                'X-Temperature': temperature,
            }
        });

        if (!res.ok) {
            const e = await res.json().catch(() => ({ detail: res.statusText }));
            aiEl.classList.remove('typing-cursor');
            aiEl.innerHTML = `<span class="text-red-500 text-sm">错误: ${e.detail}</span>`;
            return;
        }

        await consumeSseJsonStream(res, async (m) => {
            if (m.type === 'content') {
                fullText += m.text || '';
                aiEl.classList.remove('typing-cursor');
                aiEl.innerHTML = marked.parse(fullText);
                scrollChatToBottom();
            } else if (m.type === 'error') {
                aiEl.classList.remove('typing-cursor');
                aiEl.innerHTML = `<span class="text-red-500 text-sm">请求失败: ${escapeHtml(m.detail || '上游模型流异常中断')}</span>`;
            } else if (m.type === 'done') {
                aiEl.classList.remove('typing-cursor');
                aiEl.innerHTML = marked.parse(fullText);
                if (fullText.trim()) {
                    chatHistory.push({ role: 'assistant', content: fullText });
                    setState('chatHistory', chatHistory);
                    getEl('similar-exps-section')?.classList.remove('hidden');
                }
                emit(Events.AI_INTERPRET_COMPLETE, { itemId: currentItem.id, content: fullText });
            }
        });
    } catch (err) {
        aiEl.classList.remove('typing-cursor');
        aiEl.innerHTML = `<span class="text-red-500 text-sm">请求失败: ${err.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sparkles mr-1.5"></i>重新生成';
    }
}

/**
 * 发送聊天消息
 */
export async function sendChat() {
    const input = getEl('chat-input');
    const msg = input?.value?.trim();
    if (!msg) return;

    // 读取面板内的模型设置
    syncInterpretModelSelect();

    const sel = getEl('interpret-model-select');
    const [provider, model] = (sel?.value || 'glm:glm-4-flash').split(':');
    const key = aiStorage.getKey(provider);
    const temperature = getEl('ai-temperature')?.value || '0.5';
    const ragEnabled = getEl('rag-toggle')?.checked ?? true;
    const currentItem = getState('currentItem');

    if (!key || key.length < 11) {
        alert('请先在配置中填写 ' + (provider === 'glm' ? 'GLM' : 'DeepSeek') + ' API Key');
        const { toggleConfigModal } = await import('../../config/ai-config.js');
        toggleConfigModal();
        return;
    }

    input.value = '';
    getEl('chat-placeholder')?.remove();

    // 用户消息气泡
    appendChatBubble('user', msg);

    let chatHistory = getState('chatHistory') || [];
    chatHistory.push({ role: 'user', content: msg });
    setState('chatHistory', chatHistory);

    // AI 占位气泡
    const aiEl = appendChatBubble('assistant', '');
    aiEl.classList.add('typing-cursor');
    scrollChatToBottom();

    const sendBtn = getEl('chat-send-btn');
    if (sendBtn) sendBtn.disabled = true;

    let fullReply = '';

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Provider': provider,
                'X-Api-Key': key,
                'X-Model': model,
                'X-Temperature': temperature,
            },
            body: JSON.stringify({
                message: msg,
                // RAG 关闭时不传 image_id，不触发相似实验检索
                image_id: ragEnabled ? (currentItem?.id ?? null) : null,
                history: chatHistory.slice(-10),
            })
        });

        if (!res.ok) {
            const err = await res.json();
            aiEl.innerHTML = `<span class="text-red-500">错误: ${err.detail}</span>`;
            aiEl.classList.remove('typing-cursor');
            return;
        }

        await consumeSseJsonStream(res, async (m) => {
            if (m.type === 'content') {
                fullReply += m.text || '';
                aiEl.classList.remove('typing-cursor');
                aiEl.innerHTML = marked.parse(fullReply);
                scrollChatToBottom();
            } else if (m.type === 'error') {
                aiEl.classList.remove('typing-cursor');
                aiEl.innerHTML = `<span class="text-red-500">请求失败: ${escapeHtml(m.detail || '上游模型流异常中断')}</span>`;
            } else if (m.type === 'done') {
                aiEl.classList.remove('typing-cursor');
                if (fullReply.trim()) {
                    chatHistory.push({ role: 'assistant', content: fullReply });
                    setState('chatHistory', chatHistory);
                }
                emit(Events.AI_MESSAGE_RECEIVED, { content: fullReply });
            }
        });
    } catch (err) {
        aiEl.innerHTML = `<span class="text-red-500">请求失败: ${err.message}</span>`;
        aiEl.classList.remove('typing-cursor');
    } finally {
        if (sendBtn) sendBtn.disabled = false;
    }
}

/**
 * 构建算法解释（本地计算，无需 AI）
 */
export function buildAlgoExplanation(features, params) {
    const den = features.density != null ? Number(features.density).toFixed(2) : '--';
    const ali = features.alignment != null ? Number(features.alignment).toFixed(3) : '--';
    const diaMean = features.diameter_mean != null ? Number(features.diameter_mean).toFixed(1) : '--';
    const diaStd = features.diameter_std != null ? Number(features.diameter_std).toFixed(1) : '--';
    const diaMin = features.diameter_min != null ? Number(features.diameter_min).toFixed(1) : '--';
    const diaMax = features.diameter_max != null ? Number(features.diameter_max).toFixed(1) : '--';
    const tort = features.tortuosity || (features.curvature != null && !isNaN(Number(features.curvature)) ? 1.0 / Number(features.curvature) : null);
    const curRaw = tort ? (1.0 / tort) : null;
    const cur = curRaw != null ? curRaw.toFixed(3) : '--';
    const mag = params?.magnification ?? '未知';
    const sid = params?.sample_id ?? '';
    const actualTemp = params?.actual_temp ? Number(params.actual_temp).toFixed(1) : (params?.growth_temp || '--');
    const posCm = params?.membrane_pos_cm != null ? Number(params.membrane_pos_cm).toFixed(1) : '--';

    const dq = features.density == null ? '' : (features.density > 25 && features.density < 65) ? '✅ 正常' : features.density <= 25 ? '⚠️ 偏低' : '⚠️ 偏高';
    const aq = features.alignment == null ? '' : features.alignment > 0.7 ? '✅ 高取向' : features.alignment > 0.4 ? '🔶 中等' : '❌ 取向差';
    const diq = features.diameter_mean == null ? '' : (features.diameter_mean > 8 && features.diameter_mean < 30) ? '✅ 正常' : '⚠️ 偏离';

    let cq = '', curDesc = '';
    if (curRaw == null) {
        cq = '未提取'; curDesc = '特征未提取，请先运行特征提取。';
    } else if (curRaw > 0.85) {
        cq = '✅ 最优 (Straight)'; curDesc = 'CNT生长应力均匀，直线度极高，工艺条件理想。';
    } else if (curRaw > 0.60) {
        cq = '🔶 中等 (Wavy)'; curDesc = 'CNT呈现轻微波浪状，可能存在热场不均或局部气流扰动。建议检查基底平整度。';
    } else {
        cq = '❌ 较差 (Coiled)'; curDesc = 'CNT由于内应力或催化剂活性分布不均，呈现明显的弯曲或卷曲，会显著降低取向性和导电性。';
    }

    const aliDesc = features.alignment == null ? '未提取' : features.alignment > 0.7 ? '高取向性，CNT 生长平行度极佳。' : features.alignment > 0.4 ? '取向性中等，存在部分杂乱区域。' : '取向性偏低，阵列生长较为紊乱。';

    return `## 📊 特征提取总览${sid ? ' · ' + sid : ''}

| 特征 | 数值 | 评价 |
|------|------|------|
| 面密度 Density | **${den}%** | ${dq} |
| 取向性 Alignment | **${ali}** | ${aq} |
| 平均管径 Diameter | **${diaMean}±${diaStd} nm** (范围: ${diaMin}-${diaMax}) | ${diq} |
| 弯曲度 Curvature | **${cur}** | ${cq} |

> **关键工艺溯源**：样点物理位置 **${posCm} cm**，炉温校准后实际生长温度 **${actualTemp} ℃**。
> 放大倍率：${mag}× — 特征由算法引擎本地计算。
---

## 🔬 算法一：面密度提取（Density）

**提取结果：${den}%**

**处理流程：**

1. **CLAHE 自适应直方图均衡** — clipLimit=2.0，瓦片 8×8，增强局部CNT与背景对比度
2. **高斯降噪** — 5×5 卷积核，抑制椒盐噪声
3. **自适应二值化** — 以 11×11 邻域高斯加权均值为阈值（偏移 C=2），将明亮CNT像素与背景分离
4. **面积比统计** — D = N_white / N_total × 100%

**物理意义：** 反映SEM视野中CNT阵列填充率，典型优质样品 25–60%，与催化剂活性位点密度正相关。

---

## 📐 算法二：取向性（Alignment 值）

**提取结果：S = ${ali}**

**处理流程：**

1. **Sobel 梯度场** — 计算 X/Y 方向一阶梯度（ksize=3）得 Gx、Gy
2. **强边缘筛选** — 幅值 M=√(Gx²+Gy²) > 第70百分位，过滤弱梯度区域噪声
3. **轴向角转换** — 梯度法向 +90° → CNT轴向角 θ，计算加权主方向 θ̄
4. **序参数** — S = ⟨2·cos²(θ − θ̄) − 1⟩，S→1 完美垂直，S=0 随机，S→−0.5 水平

**当前：** S=${ali}，${aliDesc}

---

## 📏 算法三：平均管径估计（Diameter Distribution）

**提取结果：均值 ${diaMean} nm ± ${diaStd} nm，范围 ${diaMin}-${diaMax} nm**

**处理流程：**

1. **Otsu 二值化** — 自动最优灰度阈值，分离CNT前景与背景
2. **形态学骨架化** — scikit-image.skeletonize，将CNT区域细化为 1px 宽中心线
3. **欧氏距离变换** — 每个前景像素到最近背景像素的距离 = 局部半径
4. **中位数管径** — 骨架点处 距离×2 = 局部直径，取中位数排除端部误差
5. **单位换算** — 基于倍率（${mag}×）和 HFW 将像素换算为纳米

**物理意义：** 管径由Fe纳米催化剂颗粒尺寸决定，MWCNT典型范围 8–30 nm。

---

## 🌀 算法四：弯曲度分类（Curvature）

**提取结果：${cur}**

**处理流程：**

1. **连通域标记** — scipy.ndimage.label 识别骨架各独立CNT线段
2. **直线度计算** — 直线度 = 两端点欧氏距离 ÷ 骨架路径像素数
3. **分类规则：**
   - > 0.85 → **Straight**（笔直，最优）
   - 0.60–0.85 → **Wavy**（波浪形，轻度应力不均）
   - < 0.60 → **Coiled**（螺旋/团簇，存在明显内应力）
4. **加权投票** — 以各连通域面积为权重多数投票

**当前：** ${cur} — ${curDesc}`;
}

// 导出默认对象
export default {
    startAIInterpret,
    sendChat,
    appendChatBubble,
    clearChat,
    scrollChatToBottom,
    buildAlgoExplanation,
    syncInterpretModelSelect,
};
