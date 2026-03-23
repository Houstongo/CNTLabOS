// Data Clean 模块 - 数据清洗与审查

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { API_BASE } from '../../core/constants.js';
import { fmtMl } from '../../utils/format.js';

/**
 * 计算清洗评估
 */
export function computeCleanAssessment(item) {
    const reasons = [];
    let score = 100;
    const density = Number(item.density);
    const alignment = Number(item.alignment);
    const diameter = Number(item.diameter);
    const curvature = Number(item.curvature);
    const tortuosity = Number(item.tortuosity);
    const mag = Number(item.magnification);

    if (!Number.isFinite(density)) {
        score -= 35;
        reasons.push({ level: 'bad', text: '缺少 density，无法判断前景面积比例。' });
    } else {
        if (density <= 0 || density > 95) {
            score -= 40;
            reasons.push({ level: 'bad', text: `density=${fmtMl(density, 1)}%，明显异常，疑似二值化失败。` });
        } else if (density < 5 || density > 85) {
            score -= 18;
            reasons.push({ level: 'warn', text: `density=${fmtMl(density, 1)}%，处于警戒区。` });
        }
    }

    if (!Number.isFinite(alignment)) {
        score -= 20;
        reasons.push({ level: 'bad', text: '缺少 alignment，无法判断取向。' });
    } else if (alignment < -0.5 || alignment > 1.0) {
        score -= 30;
        reasons.push({ level: 'bad', text: `alignment=${fmtMl(alignment, 3)} 超出理论范围。` });
    } else if (alignment < 0) {
        score -= 12;
        reasons.push({ level: 'warn', text: `alignment=${fmtMl(alignment, 3)} 偏低，可能为杂乱网络或算法偏差。` });
    }

    if (!Number.isFinite(diameter)) {
        score -= 15;
        reasons.push({ level: 'warn', text: '缺少 diameter，可能倍率不足或直径提取失败。' });
    } else if (diameter < 5 || diameter > 200) {
        score -= 30;
        reasons.push({ level: 'bad', text: `diameter=${fmtMl(diameter, 1)} nm 不合理，可能测到噪声或束宽。` });
    } else if (diameter > 120) {
        score -= 12;
        reasons.push({ level: 'warn', text: `diameter=${fmtMl(diameter, 1)} nm 偏大，更像束宽而非单管直径。` });
    }

    if (!Number.isFinite(curvature)) {
        score -= 18;
        reasons.push({ level: 'warn', text: '缺少 curvature，局部弯曲强度不可用。' });
    } else if (curvature < 0 || curvature > 2.0) {
        score -= 28;
        reasons.push({ level: 'bad', text: `curvature=${fmtMl(curvature, 3)} 超出推荐范围。` });
    } else if (curvature > 0.8) {
        score -= 10;
        reasons.push({ level: 'warn', text: `curvature=${fmtMl(curvature, 3)} 偏高，需检查骨架是否抖动。` });
    }

    if (!Number.isFinite(tortuosity)) {
        score -= 10;
        reasons.push({ level: 'warn', text: '缺少 tortuosity，整体绕曲度无法参考。' });
    } else if (tortuosity < 1.0 || tortuosity > 3.0) {
        score -= 25;
        reasons.push({ level: 'bad', text: `tortuosity=${fmtMl(tortuosity, 3)} 不合理，可能路径追踪异常。` });
    } else if (tortuosity > 2.2) {
        score -= 10;
        reasons.push({ level: 'warn', text: `tortuosity=${fmtMl(tortuosity, 3)} 偏高，需确认是否真实卷曲。` });
    }

    if (Number.isFinite(mag) && mag < 20000) {
        score -= 14;
        reasons.push({ level: 'warn', text: `倍率 ${mag}x 偏低，直径与曲率结果天然低可信。` });
    }

    let confidence = 'high';
    let label = '高可信';
    if (score < 55) {
        confidence = 'low';
        label = '低可信';
    } else if (score < 80) {
        confidence = 'medium';
        label = '可参考';
    }

    if (!reasons.length) {
        reasons.push({ level: 'ok', text: '当前样品各字段都在推荐范围内，可先作为高可信样品查看。' });
    }

    return { score: Math.max(0, Math.round(score)), confidence, label, reasons };
}

/**
 * 标准化清洗项
 */
export function normalizeCleanItem(item) {
    const assessment = computeCleanAssessment(item);
    return { ...item, assessment, is_deleted: Number(item.is_deleted || 0) };
}

/**
 * 获取过滤后的清洗项
 */
function getFilteredCleanItems() {
    const cleanState = getState('clean') || {};
    let rows = [...(cleanState.items || [])];
    const confidence = getEl('clean-confidence-filter')?.value || '';
    const processed = getEl('clean-processed-filter')?.value || '';
    const keyword = (getEl('clean-keyword-filter')?.value || '').trim().toLowerCase();

    if (confidence) rows = rows.filter(x => x.assessment.confidence === confidence);
    if (processed !== '') rows = rows.filter(x => String(Number(x.processed || 0)) === processed);
    if (keyword) {
        rows = rows.filter(x => {
            const text = [
                x.sample_id, x.source, x.file_path,
                x.position_label, x.horizontal_pos,
            ].filter(Boolean).join(' ').toLowerCase();
            return text.includes(keyword);
        });
    }

    return rows;
}

/**
 * 渲染清洗列表
 */
export function renderCleanList() {
    const list = getEl('clean-sample-list');
    const meta = getEl('clean-list-meta');
    if (!list) return;

    const cleanState = getState('clean') || {};
    const rows = getFilteredCleanItems();
    cleanState.filteredItems = rows;

    if (meta) {
        const scopeText = cleanState.view === 'deleted' ? '回收站显示' : '显示';
        meta.innerText = `${scopeText} ${rows.length} / ${cleanState.items.length}`;
    }

    if (!rows.length) {
        list.innerHTML = `<div class="text-sm text-slate-400 font-bold py-8 text-center">${cleanState.view === 'deleted' ? '回收站中没有符合筛选条件的样品' : '没有符合筛选条件的样品'}</div>`;
        return;
    }

    list.innerHTML = rows.map(item => `
        <button type="button" onclick="window.dispatchEvent(new CustomEvent('clean-select-item', { detail: { id: ${item.id} } }))" class="clean-list-row ${cleanState.selectedId === item.id ? 'active' : ''} w-full text-left mb-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 transition p-3">
            <div class="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-2">
                <div class="min-w-0">
                    <div class="font-black text-slate-800 text-sm truncate">${escapeHtml(String(item.sample_id || item.file_path || `ID-${item.id}`))}</div>
                    <div class="text-[11px] text-slate-500 mt-1 truncate">${escapeHtml(String(item.file_path || ''))}</div>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                    ${item.is_deleted ? '<span class="bg-red-100 text-red-700 text-xs px-2 py-1 rounded">已删除</span>' : ''}
                    <span class="clean-chip ${item.assessment.confidence}">${item.assessment.label}</span>
                    <span class="text-[11px] font-black text-slate-500">分数 ${item.assessment.score}</span>
                </div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-2 mt-3 text-[11px] text-slate-500">
                <div>density <span class="font-black text-slate-700">${fmtMl(item.density, 1)}</span></div>
                <div>alignment <span class="font-black text-slate-700">${fmtMl(item.alignment, 3)}</span></div>
                <div>diameter <span class="font-black text-slate-700">${fmtMl(item.diameter, 1)}</span></div>
                <div>curvature <span class="font-black text-slate-700">${fmtMl(item.curvature, 3)}</span></div>
                <div>tortuosity <span class="font-black text-slate-700">${fmtMl(item.tortuosity, 3)}</span></div>
            </div>
        </button>
    `).join('');
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

// 导出默认对象
export default {
    computeCleanAssessment,
    normalizeCleanItem,
    getFilteredCleanItems,
    renderCleanList,
};
