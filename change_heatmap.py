
import re

file_path = r'd:\CNTDATA\CNTA_ML_Project\index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Update dropdown
html = html.replace(
    '<option value="3d-surface">三维响应面 (曲面)</option>',
    '<option value="heatmap">二维特征热力图</option>'
)

# 2. Update toggle condition
html = html.replace(
    "const is3d = mlVizState.mode.startsWith('3d');",
    "const is3d = mlVizState.mode.startsWith('3d') || mlVizState.mode === 'heatmap';"
)

# 3. Update the inner rendering logic in renderMl3D
render_pattern = r"const commonOption = \{.*?\}\s*;\s*if \(mode === '3d-surface'\) \{.*?(?=else \{)"
render_replacement = """
            if (mode === 'heatmap') {
                // 响应面逻辑转换为热力图
                const gridN = 40; // 提高分辨率
                const xMin = Math.min(...pts.map(p => p.x));
                const xMax = Math.max(...pts.map(p => p.x));
                const yMin = Math.min(...pts.map(p => p.y));
                const yMax = Math.max(...pts.map(p => p.y));
                const xStep = (xMax - xMin) / gridN || 1;
                const yStep = (yMax - yMin) / gridN || 1;
                
                const xAxisData = [];
                const yAxisData = [];
                for (let i = 0; i <= gridN; i++) xAxisData.push(fmtMl(xMin + i * xStep, 2));
                for (let j = 0; j <= gridN; j++) yAxisData.push(fmtMl(yMin + j * yStep, 2));

                const heatData = [];
                for (let i = 0; i <= gridN; i++) {
                    for (let j = 0; j <= gridN; j++) {
                        const gx = xMin + i * xStep;
                        const gy = yMin + j * yStep;
                        
                        let sumZ = 0, sumW = 0;
                        pts.forEach(p => {
                            const dx = (p.x - gx) / xStep;
                            const dy = (p.y - gy) / yStep;
                            const distSq = dx*dx + dy*dy;
                            if (distSq < 0.0001) { sumZ = p.z; sumW = 1; return; }
                            const w = 1.0 / (distSq + 0.1);
                            sumZ += p.z * w;
                            sumW += w;
                        });
                        if (sumW > 0) heatData.push([i, j, sumZ / sumW]);
                    }
                }

                chart3d.setOption({
                    animation: false,
                    tooltip: {
                        position: 'top',
                        formatter: p => `${xLabel}: ${xAxisData[p.value[0]]}<br/>${yLabel}: ${yAxisData[p.value[1]]}<br/>${zLabel}: ${fmtMl(p.value[2], 3)}`
                    },
                    grid: { right: 80, top: 40, left: 60, bottom: 50 },
                    xAxis: { type: 'category', data: xAxisData, name: xLabel, nameLocation: 'middle', nameGap: 30, axisLabel: { interval: Math.floor(gridN/5) } },
                    yAxis: { type: 'category', data: yAxisData, name: yLabel, nameLocation: 'middle', nameGap: 45, axisLabel: { interval: Math.floor(gridN/5) } },
                    visualMap: {
                        min: zMin, max: zMax,
                        calculable: true,
                        orient: 'vertical',
                        right: 15, top: 40,
                        textStyle: { color: '#475569', fontSize: 10 },
                        inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
                    },
                    series: [{
                        type: 'heatmap',
                        data: heatData,
                        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } }
                    }]
                }, true);
            } """

# It's safer to just replace from "const commonOption" downwards to the scatter3D options
# Let's write a precise regex or manual find.

old_logic = '''
            const commonOption = {
                animation: false,
                tooltip: {
                    formatter: params => {
                        const d = params?.data || {};
                        const row = d.row || {};
                        const val = Array.isArray(d) ? d : (d.value || []);
                        return [
                            `X (${xLabel}): ${fmtMl(val[0], 2)}`,
                            `Y (${yLabel}): ${fmtMl(val[1], 2)}`,
                            `Target (${zLabel}): ${fmtMl(val[2], 3)}`,
                        ].join('<br/>');
                    }
                },
                visualMap: {
                    min: zMin, max: zMax, dimension: 2,
                    calculable: true, orient: 'vertical', right: 8, top: 40,
                    textStyle: { color: '#475569', fontSize: 10 },
                    inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
                },
                xAxis3D: { type: 'value', name: xLabel, nameTextStyle: { color: '#94a3b8' } },
                yAxis3D: { type: 'value', name: yLabel, nameTextStyle: { color: '#94a3b8' } },
                zAxis3D: { type: 'value', name: zLabel, nameTextStyle: { color: '#94a3b8' } },
                grid3D: {
                    boxWidth: 100, boxHeight: 60, boxDepth: 100,
                    axisLine: { lineStyle: { color: '#cbd5e1' } },
                    axisPointer: { lineStyle: { color: '#38bdf8', width: 2 } },
                    viewControl: { projection: 'perspective', autoRotate: false, beta: 30, alpha: 20 },
                    light: { main: { intensity: 1.2, shadow: true }, ambient: { intensity: 0.35 } }
                }
            };

            if (mode === '3d-surface') {
                // 响应面逻辑：简单网格分箱插值
                const gridN = 20;
                const xMin = Math.min(...pts.map(p => p.x));
                const xMax = Math.max(...pts.map(p => p.x));
                const yMin = Math.min(...pts.map(p => p.y));
                const yMax = Math.max(...pts.map(p => p.y));
                const xStep = (xMax - xMin) / gridN || 1;
                const yStep = (yMax - yMin) / gridN || 1;
                
                const surfaceData = [];
                for (let i = 0; i <= gridN; i++) {
                    for (let j = 0; j <= gridN; j++) {
                        const gx = xMin + i * xStep;
                        const gy = yMin + j * yStep;
                        
                        // 寻找邻近点的加权平均（反距离衰减）
                        let sumZ = 0, sumW = 0;
                        pts.forEach(p => {
                            const dx = (p.x - gx) / xStep;
                            const dy = (p.y - gy) / yStep;
                            const distSq = dx*dx + dy*dy;
                            if (distSq < 0.0001) { sumZ = p.z; sumW = 1; return; }
                            const w = 1.0 / (distSq + 0.1); // 平滑核
                            sumZ += p.z * w;
                            sumW += w;
                        });
                        if (sumW > 0) surfaceData.push([gx, gy, sumZ / sumW]);
                    }
                }

                chart3d.setOption({
                    ...commonOption,
                    series: [{
                        type: 'surface',
                        data: surfaceData,
                        wireframe: { show: true, lineStyle: { color: 'rgba(255,255,255,0.2)', width: 0.5 } },
                        shading: 'lambert',
                        itemStyle: { opacity: 0.9 }
                    }]
                }, true);
            } else {'''

new_logic = '''
            const commonOption = {
                animation: false,
                tooltip: {
                    formatter: params => {
                        const d = params?.data || {};
                        const row = d.row || {};
                        const val = Array.isArray(d) ? d : (d.value || []);
                        return [
                            `X (${xLabel}): ${fmtMl(val[0], 2)}`,
                            `Y (${yLabel}): ${fmtMl(val[1], 2)}`,
                            `Target (${zLabel}): ${fmtMl(val[2], 3)}`,
                        ].join('<br/>');
                    }
                },
                visualMap: {
                    min: zMin, max: zMax, dimension: 2,
                    calculable: true, orient: 'vertical', right: 8, top: 40,
                    textStyle: { color: '#475569', fontSize: 10 },
                    inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
                },
                xAxis3D: { type: 'value', name: xLabel, nameTextStyle: { color: '#94a3b8' } },
                yAxis3D: { type: 'value', name: yLabel, nameTextStyle: { color: '#94a3b8' } },
                zAxis3D: { type: 'value', name: zLabel, nameTextStyle: { color: '#94a3b8' } },
                grid3D: {
                    boxWidth: 100, boxHeight: 60, boxDepth: 100,
                    axisLine: { lineStyle: { color: '#cbd5e1' } },
                    axisPointer: { lineStyle: { color: '#38bdf8', width: 2 } },
                    viewControl: { projection: 'perspective', autoRotate: false, beta: 30, alpha: 20 },
                    light: { main: { intensity: 1.2, shadow: true }, ambient: { intensity: 0.35 } }
                }
            };

            if (mode === 'heatmap') {
                // 响应面逻辑转换为热力图
                const gridN = 40; // 提高分辨率
                const xMin = Math.min(...pts.map(p => p.x));
                const xMax = Math.max(...pts.map(p => p.x));
                const yMin = Math.min(...pts.map(p => p.y));
                const yMax = Math.max(...pts.map(p => p.y));
                const xStep = (xMax - xMin) / gridN || 1;
                const yStep = (yMax - yMin) / gridN || 1;
                
                const xAxisData = [];
                const yAxisData = [];
                for (let i = 0; i <= gridN; i++) xAxisData.push(fmtMl(xMin + i * xStep, 2));
                for (let j = 0; j <= gridN; j++) yAxisData.push(fmtMl(yMin + j * yStep, 2));

                const heatData = [];
                for (let i = 0; i <= gridN; i++) {
                    for (let j = 0; j <= gridN; j++) {
                        const gx = xMin + i * xStep;
                        const gy = yMin + j * yStep;
                        
                        let sumZ = 0, sumW = 0;
                        pts.forEach(p => {
                            const dx = (p.x - gx) / xStep;
                            const dy = (p.y - gy) / yStep;
                            const distSq = dx*dx + dy*dy;
                            if (distSq < 0.0001) { sumZ = p.z; sumW = 1; return; }
                            const w = 1.0 / (distSq + 0.1);
                            sumZ += p.z * w;
                            sumW += w;
                        });
                        if (sumW > 0) heatData.push([i, j, sumZ / sumW]);
                    }
                }

                chart3d.setOption({
                    animation: false,
                    tooltip: {
                        position: 'top',
                        formatter: p => `${xLabel}: ${xAxisData[p.value[0]]}<br/>${yLabel}: ${yAxisData[p.value[1]]}<br/>${zLabel}: ${fmtMl(p.value[2], 3)}`
                    },
                    grid: { right: 80, top: 40, left: 60, bottom: 50 },
                    xAxis: { type: 'category', data: xAxisData, name: xLabel, nameLocation: 'middle', nameGap: 30, axisLabel: { interval: Math.floor(gridN/5) } },
                    yAxis: { type: 'category', data: yAxisData, name: yLabel, nameLocation: 'middle', nameGap: 45, axisLabel: { interval: Math.floor(gridN/5) } },
                    visualMap: {
                        min: zMin, max: zMax,
                        calculable: true,
                        orient: 'vertical',
                        right: 15, top: 40,
                        textStyle: { color: '#475569', fontSize: 10 },
                        inRange: { color: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] }
                    },
                    series: [{
                        type: 'heatmap',
                        data: heatData,
                        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } }
                    }]
                }, true);
            } else {'''

html = html.replace(old_logic, new_logic)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(html)
