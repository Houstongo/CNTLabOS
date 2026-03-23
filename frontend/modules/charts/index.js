// Charts 模块 - 图表渲染与可视化

import { getEl } from '../../utils/dom.js';
import { getState, setState } from '../../core/store.js';
import { ML_TARGET_META, ML_X_META, ML_Y_MODE_META } from '../../core/constants.js';
import { fmtMl } from '../../utils/format.js';
import { escapeHtml } from '../../utils/dom.js';

/**
 * 获取 ML Y 轴值
 */
function getMlYValue(row, target, yMode) {
    const meta = ML_TARGET_META[target] || ML_TARGET_META.diameter;
    const pred = Number(row[meta.predKey]);
    const actual = Number(row[meta.actualKey]);
    if (yMode === 'actual') return Number.isFinite(actual) ? actual : null;
    if (yMode === 'residual') return (Number.isFinite(actual) && Number.isFinite(pred)) ? (actual - pred) : null;
    return Number.isFinite(pred) ? pred : null;
}

/**
 * 确保图表实例存在
 */
function ensureMlChart(key, domId) {
    if (!window.echarts) return null;
    const dom = getEl(domId);
    if (!dom) return null;

    const chartInstances = getState('ml.chartInstances') || {};
    const cached = chartInstances[key];

    if (cached && cached.getDom() === dom) return cached;
    if (cached) cached.dispose();

    const found = echarts.getInstanceByDom(dom);
    const chart = found || echarts.init(dom, null, { renderer: 'canvas' });

    setState('ml.chartInstances', { ...chartInstances, [key]: chart });
    return chart;
}

/**
 * 构建 ML 趋势数据
 */
function buildMlTrend(points) {
    if (points.length < 8) return [];
    const sorted = [...points].sort((a, b) => a.x - b.x);
    const binCount = Math.max(5, Math.min(12, Math.round(Math.sqrt(sorted.length))));
    const xMin = sorted[0].x;
    const xMax = sorted[sorted.length - 1].x;
    const step = ((xMax - xMin) || 1) / binCount;

    const bins = Array.from({ length: binCount }, (_, i) => ({ x: xMin + step * (i + 0.5), values: [] }));

    sorted.forEach(p => {
        const idx = Math.min(binCount - 1, Math.max(0, Math.floor((p.x - xMin) / (step || 1))));
        bins[idx].values.push(p.y);
    });

    return bins
        .map(b => ({ x: b.x, y: b.values.length ? (b.values.reduce((s, v) => s + v, 0) / b.values.length) : null }))
        .filter(p => Number.isFinite(p.y));
}

/**
 * 绘制 ML 散点图
 */
export function drawMlScatter(chartId, points, xLabel, yLabel, xUnit) {
    const chart = ensureMlChart('main', chartId);
    if (!chart) return;

    const unitLabel = xUnit ? ` (${xUnit})` : '';
    const data = points.map(p => ({
        value: [Number.isFinite(p.xPlot) ? p.xPlot : p.x, p.y],
        rawX: p.x,
        row: p.row,
        hasActual: p.hasActual,
    }));

    const option = {
        animation: false,
        grid: { left: 58, right: 18, top: 16, bottom: 48 },
        tooltip: {
            trigger: 'item',
            formatter: params => {
                const d = params.data || {};
                const r = d.row || {};
                return [
                    `${escapeHtml(String(r.run_name || '--'))}`,
                    `样品: C${r.sample_no ?? '--'} / ${r.position ?? '--'} / ${r.shot_no ?? '--'}`,
                    `${xLabel}: ${fmtMl(d.rawX, 3)}${xUnit ? ` ${xUnit}` : ''}`,
                    `${yLabel}: ${fmtMl(params.value?.[1], 3)}`,
                ].join('<br/>');
            },
        },
        xAxis: {
            type: 'value',
            name: `${xLabel}${unitLabel}`,
            nameTextStyle: { fontWeight: 700, color: '#64748b' },
            axisLabel: { color: '#64748b' },
            axisLine: { lineStyle: { color: '#cbd5e1' } },
            splitLine: { show: true, lineStyle: { color: '#f1f5f9' } },
        },
        yAxis: {
            type: 'value',
            name: yLabel,
            nameTextStyle: { fontWeight: 700, color: '#64748b' },
            axisLabel: { color: '#64748b' },
            axisLine: { lineStyle: { color: '#cbd5e1' } },
            splitLine: { show: true, lineStyle: { color: '#f1f5f9' } },
        },
        series: [{
            type: 'scatter',
            data,
            symbolSize: (value, params) => (params?.data?.hasActual ? 8 : 6),
            itemStyle: {
                color: params => (params.data?.hasActual ? 'rgba(22,163,74,0.82)' : 'rgba(37,99,235,0.52)'),
            },
        }],
    };

    chart.setOption(option, true);
    chart.off('click');
    chart.on('click', params => {
        if (params?.data?.row) renderMlPointDetail({ row: params.data.row });
    });
}

/**
 * 绘制 ML 趋势图
 */
export function drawMlTrend(chartId, points, xLabel, yLabel) {
    const chart = ensureMlChart('trend', chartId);
    if (!chart) return;

    const trend = buildMlTrend(points);

    if (trend.length < 2) {
        chart.setOption({
            animation: false,
            title: {
                text: '趋势分箱后有效点不足',
                left: 10,
                top: 8,
                textStyle: { fontSize: 12, color: '#94a3b8', fontWeight: 600 },
            },
            xAxis: { show: false, type: 'value' },
            yAxis: { show: false, type: 'value' },
            series: [],
        }, true);
        return;
    }

    const option = {
        animation: false,
        grid: { left: 52, right: 18, top: 16, bottom: 34 },
        tooltip: {
            trigger: 'axis',
            formatter: params => {
                const p = params?.[0];
                if (!p) return '';
                return `${xLabel}: ${fmtMl(p.value?.[0], 3)}<br/>${yLabel}: ${fmtMl(p.value?.[1], 3)}`;
            },
        },
        xAxis: {
            type: 'value',
            name: xLabel,
            axisLabel: { color: '#64748b' },
            axisLine: { lineStyle: { color: '#cbd5e1' } },
            splitLine: { show: true, lineStyle: { color: '#f1f5f9' } },
        },
        yAxis: {
            type: 'value',
            name: yLabel,
            axisLabel: { color: '#64748b' },
            axisLine: { lineStyle: { color: '#cbd5e1' } },
            splitLine: { show: true, lineStyle: { color: '#f1f5f9' } },
        },
        series: [{
            type: 'line',
            data: trend.map(p => [p.x, p.y]),
            showSymbol: true,
            symbolSize: 7,
            smooth: false,
            lineStyle: { color: '#0284c7', width: 2.4 },
            itemStyle: { color: '#0284c7' },
        }],
    };

    chart.setOption(option, true);
}

/**
 * 计算相关系数
 */
function calculateCorrelation(xData, yData) {
    const n = xData.length;
    const meanX = xData.reduce((a, b) => a + b, 0) / n;
    const meanY = yData.reduce((a, b) => a + b, 0) / n;

    let num = 0;
    let denX = 0;
    let denY = 0;

    for (let i = 0; i < n; i++) {
        const dx = xData[i] - meanX;
        const dy = yData[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }

    return denX === 0 || denY === 0 ? 0 : num / Math.sqrt(denX * denY);
}

/**
 * 渲染 ML 点详情
 */
export function renderMlPointDetail(point) {
    const panel = getEl('ml-point-detail');
    if (!panel) return;

    if (!point || !point.row) {
        panel.innerHTML = '点击左侧散点查看具体实验数据';
        return;
    }

    const r = point.row;
    panel.innerHTML = `
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
            <div class="text-slate-500">批次</div><div class="font-bold text-slate-700 text-right">${escapeHtml(String(r.run_name || '--'))}</div>
            <div class="text-slate-500">样品/部位/拍摄</div><div class="font-bold text-slate-700 text-right">C${r.sample_no ?? '--'} / ${r.position ?? '--'} / ${r.shot_no ?? '--'}</div>
            <div class="text-slate-500">实际温度</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.actual_temp, 2)}</div>
            <div class="text-slate-500">流速</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.flow_rate, 2)}</div>
            <div class="text-slate-500">催化剂浓度</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.catalyst_concentration, 3)}</div>
            <div class="text-slate-500">管径分布(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.diameter_actual, 2)} / ${fmtMl(r.diameter_pred, 2)}</div>
            <div class="text-slate-500">覆盖密度(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.density_actual, 2)} / ${fmtMl(r.density_pred, 2)}</div>
            <div class="text-slate-500">取向度(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.alignment_actual, 3)} / ${fmtMl(r.alignment_pred, 3)}</div>
            <div class="text-slate-500">平均曲率(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.curvature_actual, 3)} / ${fmtMl(r.curvature_pred, 3)}</div>
            <div class="text-slate-500">波浪度(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.tortuosity_actual, 3)} / ${fmtMl(r.tortuosity_pred, 3)}</div>
        </div>
    `;
}

/**
 * 渲染 ML 3D 图
 */
export function renderMl3D(rows, target, yMode) {
    const chartDom = getEl('ml-chart-3d');
    const titleDom = getEl('ml-3d-title');
    if (!chartDom) return;

    const vizState = getState('ml.vizState') || {};
    const mode = vizState.mode || '2d';
    const xKey = vizState.xKey3d || 'actual_temp';
    const yKey = vizState.yKey3d || 'flow_rate';

    const xLabel = ML_X_META[xKey]?.label || xKey;
    const yLabel = ML_X_META[yKey]?.label || yKey;
    const zLabel = ML_TARGET_META[target]?.label || target;

    if (titleDom) {
        titleDom.innerText = `三维空间模型（X=${xLabel}, Y=${yLabel}, Z=${zLabel} · ${ML_Y_MODE_META[yMode] || yMode}）`;
    }

    const chart3d = ensureMlChart('chart3d', 'ml-chart-3d');
    if (!chart3d) return;

    const pts = [];
    rows.forEach(r => {
        const x = Number(r[xKey]);
        const y = Number(r[yKey]);
        const z = getMlYValue(r, target, yMode);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            pts.push({ x, y, z, row: r });
        }
    });

    if (!pts.length) {
        chart3d.clear();
        return;
    }

    const zValues = pts.map(p => p.z);
    let zMin = Math.min(...zValues), zMax = Math.max(...zValues);
    if (zMin === zMax) { zMin -= 0.1; zMax += 0.1; }

    const common3d = {
        animation: false,
        tooltip: {
            formatter: p => {
                const v = p.value || [];
                return `${xLabel}: ${fmtMl(v[0], 2)}<br>${yLabel}: ${fmtMl(v[1], 2)}<br>${zLabel}: ${fmtMl(v[2], 3)}`;
            }
        },
        visualMap: {
            min: zMin,
            max: zMax,
            dimension: 2,
            calculable: true,
            orient: 'vertical',
            right: 10,
            top: 40,
            text: ['高', '低'],
            textStyle: { color: '#475569', fontSize: 10 },
            inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
        },
        xAxis3D: { type: 'value', name: xLabel },
        yAxis3D: { type: 'value', name: yLabel },
        zAxis3D: { type: 'value', name: zLabel },
        grid3D: {
            boxWidth: 100,
            boxHeight: 60,
            boxDepth: 100,
            viewControl: { projection: 'perspective', autoRotate: false, beta: 30, alpha: 20 },
            light: { main: { intensity: 1.2 }, ambient: { intensity: 0.4 } }
        }
    };

    if (mode === '3d-heatmap') {
        const gridN = 25;
        const xMin = Math.min(...pts.map(p => p.x)), xMax = Math.max(...pts.map(p => p.x));
        const yMin = Math.min(...pts.map(p => p.y)), yMax = Math.max(...pts.map(p => p.y));
        const xS = (xMax - xMin) / gridN || 1, yS = (yMax - yMin) / gridN || 1;
        const heatData = [];

        for (let i = 0; i <= gridN; i++) {
            for (let j = 0; j <= gridN; j++) {
                const gx = xMin + i * xS, gy = yMin + j * yS;
                let sZ = 0, sW = 0;
                pts.forEach(p => {
                    const dSq = Math.pow((p.x - gx) / xS, 2) + Math.pow((p.y - gy) / yS, 2);
                    if (dSq < 0.0001) { sZ = p.z; sW = 1; return; }
                    const w = 1.0 / (dSq + 0.1); sZ += p.z * w; sW += w;
                });
                if (sW > 0) heatData.push([gx, gy, sZ / sW]);
            }
        }
        chart3d.setOption({ ...common3d, series: [{ type: 'bar3D', data: heatData, shading: 'lambert' }] }, true);
    } else {
        chart3d.setOption({ ...common3d, series: [{ type: 'scatter3D', data: pts.map(p => ({ value: [p.x, p.y, p.z], row: p.row })), symbolSize: 8 }] }, true);
        chart3d.off('click');
        chart3d.on('click', p => { if (p.data?.row) renderMlPointDetail({ row: p.data.row }); });
    }
}

/**
 * 渲染 ML 相关系数矩阵
 */
export function renderMlCorrMatrix(rows) {
    const chartDom = getEl('ml-chart-corr-matrix');
    if (!chartDom || !rows.length) return;

    const vars = [
        { key: 'actual_temp', label: '实际温度' },
        { key: 'flow_rate', label: '混合流速' },
        { key: 'catalyst_concentration', label: '催化剂' },
        { key: 'diameter_actual', label: '管径分布' },
        { key: 'density_actual', label: '密度' },
        { key: 'alignment_actual', label: '取向度' },
        { key: 'curvature_actual', label: '曲率' },
        { key: 'tortuosity_actual', label: '波浪度' }
    ];

    const chart = ensureMlChart('corrMatrix', 'ml-chart-corr-matrix');
    if (!chart) return;

    const matrix = [], labels = vars.map(v => v.label);

    vars.forEach((v1, i) => {
        vars.forEach((v2, j) => {
            if (i === j) { matrix.push([i, j, 1]); return; }
            const data1 = [], data2 = [];
            rows.forEach(r => {
                const val1 = Number(r[v1.key]), val2 = Number(r[v2.key]);
                if (Number.isFinite(val1) && Number.isFinite(val2)) { data1.push(val1); data2.push(val2); }
            });
            if (data1.length < 3) matrix.push([i, j, 0]);
            else matrix.push([i, j, calculateCorrelation(data1, data2)]);
        });
    });

    chart.setOption({
        animation: false,
        tooltip: {
            position: 'top',
            formatter: p => `${vars[p.value[0]].label} ↔ ${vars[p.value[1]].label}<br/>相关系数: <b>${p.value[2].toFixed(3)}</b>`
        },
        grid: { top: 30, bottom: 60, left: 80, right: 30 },
        xAxis: { type: 'category', data: labels, axisLabel: { interval: 0, rotate: 30, fontSize: 10 } },
        yAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10 } },
        visualMap: {
            min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
            textStyle: { fontSize: 10 },
            inRange: { color: ['#3b82f6', '#f1f5f9', '#ef4444'] }
        },
        series: [{
            type: 'heatmap',
            data: matrix,
            label: { show: false },
            emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
        }]
    }, true);
}

/**
 * 渲染 ML 系数重要性图
 */
export function renderMlCoefImportance(data) {
    const chartDom = getEl('ml-chart-coef-importance');
    if (!chartDom) return;

    const chart = ensureMlChart('coefImportance', 'ml-chart-coef-importance');
    if (!chart) return;

    if (!data || data.length === 0) {
        chart.clear();
        return;
    }

    const sortedData = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    chart.setOption({
        animation: false,
        grid: { top: 10, right: 20, bottom: 30, left: 100 },
        tooltip: {
            trigger: 'axis',
            formatter: p => `${p[0].name}: <b>${fmtMl(p[0].value, 4)}</b>`
        },
        xAxis: {
            type: 'value',
            axisLine: { show: false },
            splitLine: { show: true, lineStyle: { color: '#f1f5f9' } },
            axisLabel: { color: '#64748b', fontSize: 10 }
        },
        yAxis: {
            type: 'category',
            data: sortedData.map(d => d.name),
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: '#475569', fontSize: 10 }
        },
        series: [{
            type: 'bar',
            data: sortedData.map(d => ({
                value: d.value,
                itemStyle: { color: d.value > 0 ? '#10b981' : '#ef4444' }
            })),
            barWidth: '60%'
        }]
    }, true);
}

/**
 * 调整所有图表大小
 */
export function resizeMlCharts() {
    const chartInstances = getState('ml.chartInstances') || {};
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
}

/**
 * 切换 ML 子页面
 */
export function switchMlSubPage(page) {
    const target = page === 'data' ? 'data' : 'visual';
    const visual = getEl('ml-subpage-visual');
    const data = getEl('ml-subpage-data');
    const selTop = getEl('ml-subpage-switch-top');
    const selInline = getEl('ml-subpage-switch');

    if (visual) visual.classList.toggle('hidden', target !== 'visual');
    if (data) data.classList.toggle('hidden', target !== 'data');
    if (selTop && selTop.value !== target) selTop.value = target;
    if (selInline && selInline.value !== target) selInline.value = target;

    setState('ml.subPage', target);

    setTimeout(() => resizeMlCharts(), 0);
}

/**
 * 初始化图表模块
 */
export function initCharts() {
    window.addEventListener('resize', () => {
        resizeMlCharts();
    });
}

// 导出默认对象
export default {
    drawMlScatter,
    drawMlTrend,
    renderMl3D,
    renderMlCorrMatrix,
    renderMlCoefImportance,
    renderMlPointDetail,
    resizeMlCharts,
    switchMlSubPage,
    initCharts,
};
