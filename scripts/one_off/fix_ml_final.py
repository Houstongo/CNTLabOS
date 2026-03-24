
import os
import re
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
file_path = PROJECT_ROOT / 'index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the CLEAN block for ML Logic
ml_logic_block = """
        function onMlVizOptionChange() {
            const modeVal = document.getElementById('ml-viz-mode')?.value || '2d';
            const isMulti = modeVal.startsWith('3d');
            
            mlVizState.mode = modeVal;
            mlVizState.xKey = document.getElementById('ml-viz-xkey')?.value || 'actual_temp';
            mlVizState.yMode = document.getElementById('ml-viz-y-mode')?.value || 'pred';
            mlVizState.xKey3d = document.getElementById('ml-viz-xkey-3d')?.value || 'actual_temp';
            mlVizState.yKey3d = document.getElementById('ml-viz-ykey-3d')?.value || 'flow_rate';

            const axis2d = document.getElementById('ml-viz-axis-2d');
            const axis3d = document.getElementById('ml-viz-axis-3d');
            if (axis2d) axis2d.classList.toggle('hidden', isMulti);
            if (axis3d) axis3d.classList.toggle('hidden', !isMulti);

            if (mlUiPerfState.vizTimer) clearTimeout(mlUiPerfState.vizTimer);
            mlUiPerfState.vizTimer = setTimeout(() => applyXrMlView({ skipTable: true }), 100);
        }

        function switchMlInfoTab(tab) {
            const target = tab === 'corr' ? 'corr' : 'coef';
            const coef = document.getElementById('ml-info-coef');
            const corr = document.getElementById('ml-info-corr');
            if (coef) coef.classList.toggle('hidden', target !== 'coef');
            if (corr) corr.classList.toggle('hidden', target !== 'corr');
            document.querySelectorAll('.ml-info-tab').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.info === target);
            });
            mlVizState.infoTab = target;
        }

        function renderMlPointDetail(point) {
            const panel = document.getElementById('ml-point-detail');
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
                    <div class="text-slate-500">平均管径(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.diameter_actual, 2)} / ${fmtMl(r.diameter_pred, 2)}</div>
                    <div class="text-slate-500">覆盖密度(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.density_actual, 2)} / ${fmtMl(r.density_pred, 2)}</div>
                    <div class="text-slate-500">取向性(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.alignment_actual, 3)} / ${fmtMl(r.alignment_pred, 3)}</div>
                    <div class="text-slate-500">平均曲率(实测/预测)</div><div class="font-bold text-slate-700 text-right">${fmtMl(r.curvature_actual, 3)} / ${fmtMl(r.curvature_pred, 3)}</div>
                </div>
            `;
        }

        function renderMl3D(rows, target, yMode) {
            const chartDom = document.getElementById('ml-chart-3d');
            const titleDom = document.getElementById('ml-3d-title');
            if (!chartDom) return;

            const mode = mlVizState.mode;
            const xKey = mlVizState.xKey3d || 'actual_temp';
            const yKey = mlVizState.yKey3d || 'flow_rate';
            const xLabel = ML_X_META[xKey]?.label || xKey;
            const yLabel = ML_X_META[yKey]?.label || yKey;
            const zLabel = ML_TARGET_META[target]?.label || target;

            if (titleDom) titleDom.innerText = `三维空间模型（X=${xLabel}, Y=${yLabel}, Z=${zLabel} · ${ML_Y_MODE_META[yMode] || yMode}）`;

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
                tooltip: { formatter: p => {
                    const v = p.value || [];
                    return `${xLabel}: ${fmtMl(v[0],2)}<br>${yLabel}: ${fmtMl(v[1],2)}<br>${zLabel}: ${fmtMl(v[2],3)}`;
                }},
                visualMap: {
                    min: zMin, max: zMax, dimension: 2, calculable: true, orient: 'vertical', right: 10, top: 40,
                    text: ['高', '低'], textStyle: { color: '#475569', fontSize: 10 },
                    inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
                },
                xAxis3D: { type: 'value', name: xLabel },
                yAxis3D: { type: 'value', name: yLabel },
                zAxis3D: { type: 'value', name: zLabel },
                grid3D: {
                    boxWidth: 100, boxHeight: 60, boxDepth: 100,
                    viewControl: { projection: 'perspective', beta: 30, alpha: 20 },
                    light: { main: { intensity: 1.2 }, ambient: { intensity: 0.4 } }
                }
            };

            if (mode === '3d-heatmap') {
                const gridN = 25;
                const xMin = Math.min(...pts.map(p => p.x)), xMax = Math.max(...pts.map(p => p.x));
                const yMin = Math.min(...pts.map(p => p.y)), yMax = Math.max(...pts.map(p => p.y));
                const xS = (xMax-xMin)/gridN || 1, yS = (yMax-yMin)/gridN || 1;
                const heatData = [];
                for(let i=0; i<=gridN; i++){
                    for(let j=0; j<=gridN; j++){
                        const gx = xMin+i*xS, gy = yMin+j*yS;
                        let sZ=0, sW=0;
                        pts.forEach(p => {
                            const dSq = Math.pow((p.x-gx)/xS,2)+Math.pow((p.y-gy)/yS,2);
                            if(dSq<0.0001){ sZ=p.z; sW=1; return; }
                            const w = 1.0/(dSq+0.1); sZ+=p.z*w; sW+=w;
                        });
                        if(sW>0) heatData.push([gx, gy, sZ/sW]);
                    }
                }
                chart3d.setOption({...common3d, series:[{type:'bar3D', data:heatData, shading:'lambert'}]}, true);
            } else {
                chart3d.setOption({...common3d, series:[{type:'scatter3D', data:pts.map(p=>({value:[p.x,p.y,p.z],row:p.row})), symbolSize:8}]}, true);
                chart3d.off('click');
                chart3d.on('click', p => { if(p.data?.row) renderMlPointDetail({row: p.data.row}); });
            }
        }

        function renderMlCorrMatrix(rows) {
            const chartDom = document.getElementById('ml-chart-corr-matrix');
            if (!chartDom || !rows.length) return;

            const vars = [
                { key: 'actual_temp', label: '实际温度' },
                { key: 'flow_rate', label: '混合流速' },
                { key: 'catalyst_concentration', label: '催化剂' },
                { key: 'diameter_actual', label: '管径' },
                { key: 'density_actual', label: '密度' },
                { key: 'alignment_actual', label: '取向性' },
                { key: 'curvature_actual', label: '曲率' }
            ];

            const chart = ensureMlChart('corrMatrix', 'ml-chart-corr-matrix');
            if (!chart) return;

            const matrix = [];
            const labels = vars.map(v => v.label);
            
            vars.forEach((v1, i) => {
                vars.forEach((v2, j) => {
                    if (i === j) { matrix.push([i, j, 1]); return; }
                    const data1 = [], data2 = [];
                    rows.forEach(r => {
                        const val1 = Number(r[v1.key]);
                        const val2 = Number(r[v2.key]);
                        if (Number.isFinite(val1) && Number.isFinite(val2)) {
                            data1.push(val1);
                            data2.push(val2);
                        }
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
                    inRange: { color: ['#3b82f6', '#f8fafc', '#ef4444'] }
                },
                series: [{
                    name: 'Correlation',
                    type: 'heatmap',
                    data: matrix,
                    label: { show: true, fontSize: 9, formatter: p => p.value[2].toFixed(2) },
                    emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } }
                }]
            }, true);
        }

        function calculateCorrelation(x, y) {
            const n = x.length;
            const avgX = x.reduce((a, b) => a + b) / n;
            const avgY = y.reduce((a, b) => a + b) / n;
            let num = 0, den1 = 0, den2 = 0;
            for (let i = 0; i < n; i++) {
                const dx = x[i] - avgX;
                const dy = y[i] - avgY;
                num += dx * dy;
                den1 += dx * dx;
                den2 += dy * dy;
            }
            if (den1 === 0 || den2 === 0) return 0;
            return num / Math.sqrt(den1 * den2);
        }

        function renderXrMlCharts(rows) {
            updateMlVizControls();
            const target = ML_TARGET_META[mlVizState.target] ? mlVizState.target : 'diameter';
            const xKey = ML_X_META[mlVizState.xKey] ? mlVizState.xKey : 'actual_temp';
            const yMode = mlVizState.yMode || 'pred';
            const mode = mlVizState.mode || '2d';
            const targetMeta = ML_TARGET_META[target];
            const xMeta = ML_X_META[xKey];
            const panel2d = document.getElementById('ml-2d-panels');
            const panel3d = document.getElementById('ml-3d-panel');
            const points = [];
            let actualCount = 0;
            rows.forEach(r => {
                const x = Number(r[xKey]);
                const y = getMlYValue(r, target, yMode);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return;
                const hasActual = Number.isFinite(Number(r[targetMeta.actualKey]));
                if (hasActual) actualCount += 1;
                points.push({ x, y, hasActual, row: r });
            });

            const isMultiVar = mode.startsWith('3d');
            if (panel2d) panel2d.classList.toggle('hidden', isMultiVar);
            if (panel3d) panel3d.classList.toggle('hidden', !isMultiVar);
            if (isMultiVar) {
                renderMl3D(rows, target, yMode);
            }
            renderMlCorrMatrix(rows);

            const uniqueX = [...new Set(points.map(p => p.x))].sort((a, b) => a - b);
            const shouldJitter = uniqueX.length > 0 && uniqueX.length <= 12;
            let jitter = 0;
            if (shouldJitter) {
                if (uniqueX.length > 1) {
                    const gaps = [];
                    for (let i = 1; i < uniqueX.length; i += 1) gaps.push(uniqueX[i] - uniqueX[i - 1]);
                    const minGap = Math.min(...gaps.filter(g => g > 0));
                    jitter = Number.isFinite(minGap) ? minGap * 0.08 : 0.2;
                } else { jitter = 0.2; }
                points.forEach((p, i) => {
                    const offset = ((i % 7) - 3) * jitter;
                    p.xPlot = p.x + offset;
                });
            }

            const yTitle = `${targetMeta.label} · ${ML_Y_MODE_META[yMode] || yMode}`;
            const mainTitle = document.getElementById('ml-main-chart-title');
            const trendTitle = document.getElementById('ml-trend-chart-title');
            if (!isMultiVar) {
                if (mainTitle) mainTitle.innerText = `${targetMeta.label} vs ${xMeta.label}（${ML_Y_MODE_META[yMode] || yMode}）`;
                if (trendTitle) trendTitle.innerText = `${targetMeta.label} 分箱趋势（按 ${xMeta.label}）`;
                drawMlScatter('ml-chart-main', points, xMeta.label, yTitle, xMeta.unit);
                drawMlTrend('ml-chart-trend', points, xMeta.label, targetMeta.label);
            }

            const panel = document.getElementById('ml-viz-summary');
            if (panel) {
                const ys = points.map(p => p.y);
                const yAvg = ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : null;
                const ys_sorted = [...ys].sort((a,b)=>a-b);
                const yMin = ys_sorted[0], yMax = ys_sorted[ys_sorted.length-1];
                const uniquePreview = uniqueX.slice(0, 8).map(v => fmtMl(v, 2)).join(', ') + (uniqueX.length > 8 ? ' ...' : '');
                panel.innerHTML = `
                    <div class="flex items-center justify-between"><span class="text-slate-500">指标</span><span class="font-bold text-slate-700">${targetMeta.label}</span></div>
                    <div class="flex items-center justify-between"><span class="text-slate-500">横轴</span><span class="font-bold text-slate-700">${xMeta.label}</span></div>
                    <div class="flex items-center justify-between"><span class="text-slate-500">Y 值来源</span><span class="font-bold text-slate-700">${ML_Y_MODE_META[yMode] || yMode}</span></div>
                    <div class="flex items-center justify-between"><span class="text-slate-500">绘图点数</span><span class="font-bold text-slate-700">${points.length}</span></div>
                    <div class="flex items-center justify-between"><span class="text-slate-500">含实测标签</span><span class="font-bold text-slate-700">${actualCount}</span></div>
                    <div class="flex items-center justify-between"><span class="text-slate-500">Y 均值</span><span class="font-bold text-slate-700">${fmtMl(yAvg, 3)}</span></div>
                    <div class="text-[11px] text-slate-500 leading-5 mt-2">横轴取值：${uniquePreview || '--'}</div>
                `;
            }
            mlVizState.selectedPoint = null;
        }
"""

# Match the entire block from the old onMlVizOptionChange to the end of renderXrMlCharts
pattern = r"function onMlVizOptionChange\(\) \{.*?function renderXrMlCharts\(rows\) \{.*?mlVizState\.selectedPoint = null;\s*\}"
content = re.sub(pattern, ml_logic_block, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleanup and Corr Matrix addition DONE.")
