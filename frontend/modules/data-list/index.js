// Data List 模块 - 数据列表控制

import { api } from '../../utils/api.js';
import { getEl, qs } from '../../utils/dom.js';
import { getState, setState, updateState } from '../../core/store.js';
import { emit, Events } from '../../core/events.js';
import { formatNumber } from '../../utils/format.js';
import { API_BASE } from '../../core/constants.js';

/**
 * 切换排序
 */
export function toggleSort(field) {
    const currentSort = getState('data.currentSort') || 'id';
    const currentOrder = getState('data.currentOrder') || 'desc';

    let newOrder = 'desc';
    if (currentSort === field) {
        newOrder = currentOrder === 'desc' ? 'asc' : 'desc';
    }

    // 更新状态
    setState('data.currentSort', field);
    setState('data.currentOrder', newOrder);

    // 更新 UI 图标
    document.querySelectorAll('.table-header i').forEach(el => {
        el.className = 'fas fa-sort text-[10px] ml-1 opacity-20';
    });
    const activeIcon = getEl('sort-' + field);
    if (activeIcon) {
        activeIcon.className = `fas fa-sort-${newOrder === 'desc' ? 'down' : 'up'} text-[10px] ml-1 text-blue-500 opacity-100`;
    }

    emit(Events.SORT_CHANGED, { field, order: newOrder });
    return resetAndLoad();
}

/**
 * 重置偏移量并加载数据
 */
export async function resetAndLoad() {
    setState('data.offset', 0);
    return loadData();
}

/**
 * 切换页面
 */
export async function changePage(dir) {
    const offset = getState('data.offset') || 0;
    const limit = getState('data.limit') || 10;
    const totalItems = getState('data.totalItems') || 0;

    const newOffset = offset + (dir * limit);
    if (newOffset < 0 || newOffset >= totalItems) return;

    setState('data.offset', newOffset);
    emit(Events.PAGE_CHANGED, { offset: newOffset, page: Math.floor(newOffset / limit) + 1 });
    return loadData();
}

/**
 * 跳转到指定页面
 */
export async function jumpToPage() {
    const input = getEl('jump-input');
    if (!input) return;

    const page = parseInt(input.value);
    if (!page || page < 1) return;

    const limit = getState('data.limit') || 10;
    const totalItems = getState('data.totalItems') || 0;
    const newOffset = (page - 1) * limit;

    if (newOffset >= totalItems) {
        alert("超出最大范围");
        return;
    }

    setState('data.offset', newOffset);
    emit(Events.PAGE_CHANGED, { offset: newOffset, page });
    return loadData();
}

/**
 * 获取当前可见的数据 ID 列表
 */
export function getVisibleDataIds() {
    const currentListItemsById = getState('data.currentListItemsById') || {};
    return Object.keys(currentListItemsById).map(id => Number(id)).filter(id => Number.isFinite(id));
}

/**
 * 检查数据行是否被选中
 */
export function isDataRowSelected(id) {
    const selectedDataIds = getState('data.selectedDataIds') || new Set();
    return selectedDataIds.has(Number(id));
}

/**
 * 协调数据选择（过滤掉不可见项）
 */
export function reconcileDataSelection(items) {
    const visibleIds = new Set((items || []).map(item => Number(item.id)));
    const selectedDataIds = getState('data.selectedDataIds') || new Set();

    const newSelection = new Set([...selectedDataIds].filter(id => visibleIds.has(Number(id))));
    setState('data.selectedDataIds', newSelection);
}

/**
 * 更新数据行选中状态样式
 */
export function updateDataRowSelectionState(id) {
    const row = getEl(`data-row-${id}`);
    if (!row) return;

    const selected = isDataRowSelected(id);
    row.classList.toggle('bg-blue-50/70', selected);
    row.classList.toggle('ring-1', selected);
    row.classList.toggle('ring-blue-100', selected);
}

/**
 * 更新全选复选框状态
 */
export function updateDataSelectAllState() {
    const checkbox = getEl('data-select-all');
    if (!checkbox) return;

    const visibleIds = getVisibleDataIds();
    const selectedDataIds = getState('data.selectedDataIds') || new Set();
    const selectedVisibleCount = visibleIds.filter(id => selectedDataIds.has(id)).length;

    checkbox.checked = visibleIds.length > 0 && selectedVisibleCount === visibleIds.length;
    checkbox.indeterminate = selectedVisibleCount > 0 && selectedVisibleCount < visibleIds.length;
}

/**
 * 渲染数据批量工具栏
 */
export function renderDataBatchToolbar() {
    const countEl = getEl('header-batch-count');
    const analyzeBtn = getEl('data-batch-analyze-btn');
    const deleteBtn = getEl('data-batch-delete-btn');

    if (!countEl || !analyzeBtn || !deleteBtn) return;

    const selectedDataIds = getState('data.selectedDataIds') || new Set();
    const count = selectedDataIds.size;
    const viewMode = getState('data.viewMode') || 'active';

    countEl.innerText = String(count);

    analyzeBtn.disabled = count === 0;
    deleteBtn.disabled = count === 0 || viewMode === 'deleted';
    deleteBtn.classList.toggle('hidden', viewMode === 'deleted');
}

/**
 * 切换数据行选择
 */
export function toggleDataRowSelection(id, checked) {
    const value = Number(id);
    updateState('data.selectedDataIds', ids => {
        const newIds = new Set(ids);
        if (checked) {
            newIds.add(value);
        } else {
            newIds.delete(value);
        }
        return newIds;
    });

    updateDataRowSelectionState(value);
    updateDataSelectAllState();
    renderDataBatchToolbar();

    emit(Events.BATCH_SELECTION_CHANGED, { id: value, selected: checked });
}

/**
 * 切换全选
 */
export function toggleSelectAllDataRows(checked) {
    const visibleIds = getVisibleDataIds();

    updateState('data.selectedDataIds', ids => {
        const newIds = new Set(ids);
        visibleIds.forEach(id => {
            if (checked) {
                newIds.add(id);
            } else {
                newIds.delete(id);
            }
            const checkbox = qs(`.data-row-checkbox[data-id="${id}"]`);
            if (checkbox) checkbox.checked = checked;
            updateDataRowSelectionState(id);
        });
        return newIds;
    });

    updateDataSelectAllState();
    renderDataBatchToolbar();
}

/**
 * 清空数据选择
 */
export function clearDataSelection() {
    setState('data.selectedDataIds', new Set());

    qs('.data-row-checkbox', document)?.forEach(el => {
        el.checked = false;
    });

    getVisibleDataIds().forEach(updateDataRowSelectionState);
    updateDataSelectAllState();
    renderDataBatchToolbar();
}

/**
 * 切换批量模式
 */
export function toggleBatchMode() {
    const batchMode = getState('data.batchMode') || false;
    const newBatchMode = !batchMode;

    setState('data.batchMode', newBatchMode);

    const dataPage = getEl('data-page');
    const initBtn = getEl('batch-init-btn');
    const actionGroup = getEl('batch-action-group');

    if (dataPage) dataPage.classList.toggle('batch-active', newBatchMode);
    if (initBtn) initBtn.classList.toggle('hidden', newBatchMode);
    if (actionGroup) actionGroup.classList.toggle('hidden', !newBatchMode);

    if (!newBatchMode) {
        clearDataSelection();
    }

    emit(Events.BATCH_MODE_TOGGLED, { batchMode: newBatchMode });
}

/**
 * 描述批量操作结果
 */
function describeBatchActionResult(actionLabel, payload) {
    const summary = payload?.summary || {};
    const success = Number(summary.success_count || 0);
    const skipped = Number(summary.skipped_count || 0);
    const failed = Number(summary.failed_count || 0);
    const effectiveFailed = Math.max(0, failed - skipped);
    const skippedText = skipped > 0 ? `, skipped ${skipped}` : '';
    const failedText = effectiveFailed > 0 ? `, failed ${effectiveFailed}` : '';
    return `${actionLabel} complete: success ${success}${skippedText}${failedText}`;
}

/**
 * 执行批量分析
 */
export async function runDataBatchAnalyze() {
    const selectedDataIds = getState('data.selectedDataIds') || new Set();
    const imageIds = [...selectedDataIds];

    if (!imageIds.length) return;
    if (!confirm(`确认对选中的 ${imageIds.length} 项进行批量分析？`)) return;

    const button = getEl('data-batch-analyze-btn');
    if (!button) return;

    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Running';

    try {
        const payload = await api.images.batch.analyze(imageIds);
        alert(describeBatchActionResult('Batch analyze', payload));

        await loadData();

        const currentItem = getState('currentItem');
        const currentListItemsById = getState('data.currentListItemsById') || {};
        if (currentItem && currentListItemsById[String(currentItem.id)]) {
            const { openDetailsById } = await import('../details/index.js');
            openDetailsById(currentItem.id);
        }

        emit(Events.BATCH_ACTION_COMPLETED, { action: 'analyze', count: imageIds.length });
    } catch (err) {
        console.error('Batch analyze failed:', err);
        alert('Batch analyze failed: ' + err.message);
    } finally {
        button.innerHTML = original;
        renderDataBatchToolbar();
    }
}

/**
 * 执行批量删除
 */
export async function runDataBatchDelete() {
    const selectedDataIds = getState('data.selectedDataIds') || new Set();
    const imageIds = [...selectedDataIds];

    if (!imageIds.length) return;
    if (!confirm(`确认将选中的 ${imageIds.length} 项移入回收站？`)) return;

    const button = getEl('data-batch-delete-btn');
    if (!button) return;

    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Deleting';

    try {
        const payload = await api.images.batch.delete(imageIds);
        alert(describeBatchActionResult('Batch delete', payload));

        const currentItem = getState('currentItem');
        if (currentItem && imageIds.includes(Number(currentItem.id))) {
            const { closeDetails } = await import('../details/index.js');
            closeDetails();
        }

        await loadData();
        emit(Events.BATCH_ACTION_COMPLETED, { action: 'delete', count: imageIds.length });
    } catch (err) {
        console.error('Batch delete failed:', err);
        alert('Batch delete failed: ' + err.message);
    } finally {
        button.innerHTML = original;
        renderDataBatchToolbar();
    }
}

/**
 * 切换删除视图模式
 */
export function toggleDataTrashView() {
    const viewMode = getState('data.viewMode') || 'active';
    const newMode = viewMode === 'active' ? 'deleted' : 'active';
    setState('data.viewMode', newMode);
    return resetAndLoad();
}

/**
 * 渲染删除视图切换按钮
 */
export function renderDataTrashToggle() {
    const viewMode = getState('data.viewMode') || 'active';
    const toggleBtn = getEl('trash-toggle-btn');

    if (!toggleBtn) return;

    if (viewMode === 'deleted') {
        toggleBtn.innerHTML = '<i class="fas fa-arrow-left mr-1"></i> 返回数据列表';
        toggleBtn.className = 'px-3 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-700 rounded-lg text-[11px] font-black transition-all flex items-center gap-1';
    } else {
        toggleBtn.innerHTML = '<i class="fas fa-trash mr-1"></i> 回收站';
        toggleBtn.className = 'px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-500 rounded-lg text-[11px] font-black transition-all flex items-center gap-1';
    }
}

/**
 * 加载数据列表
 */
export async function loadData() {
    const sourceFilter = getEl('source-filter');
    const source = sourceFilter?.value || '';

    const searchInput = getEl('search-input');
    const search = searchInput?.value?.trim() || '';

    const offset = getState('data.offset') || 0;
    const limit = getState('data.limit') || 10;
    const currentSort = getState('data.currentSort') || 'id';
    const currentOrder = getState('data.currentOrder') || 'desc';
    const viewMode = getState('data.viewMode') || 'active';

    // 搜索框清除按钮显隐
    const clearBtn = getEl('search-clear-btn');
    if (clearBtn) clearBtn.classList.toggle('hidden', !search);

    try {
        const data = await api.images.list({
            limit,
            offset,
            sort_by: currentSort,
            order: currentOrder,
            source,
            deletion_view: viewMode,
            ...(search ? { search } : {}),
        });

        const totalItems = data.total || 0;

        setState('data.totalItems', totalItems);
        setState('data.currentListItemsById', Object.fromEntries((data.items || []).map(item => [String(item.id), item])));
        // 同步内联全局变量（双系统兼容）
        window.currentListItemsById = Object.fromEntries((data.items || []).map(item => [String(item.id), item]));

        renderDataTrashToggle();

        const totalCountEl = getEl('total-count');
        const pageDisplayEl = getEl('page-display');
        const prevBtn = getEl('prev-btn');
        const nextBtn = getEl('next-btn');

        if (totalCountEl) {
            totalCountEl.innerText = viewMode === 'deleted'
                ? `Trash view: ${totalItems} records`
                : `Total ${totalItems} records`;
        }
        if (pageDisplayEl) {
            const currentPage = Math.floor(offset / limit) + 1;
            const totalPages = Math.max(1, Math.ceil(totalItems / limit));
            pageDisplayEl.innerText = `${currentPage} / ${totalPages}`;
        }
        if (prevBtn) prevBtn.disabled = offset === 0;
        if (nextBtn) nextBtn.disabled = offset + limit >= totalItems;

        reconcileDataSelection(data.items || []);
        renderDataTable(data.items || []);
        renderDataBatchToolbar();
        updateDataSelectAllState();

        emit(Events.DATA_LOADED, {
            items: data.items,
            total: totalItems,
            offset,
            source,
            viewMode,
        });
    } catch (err) {
        console.error('Load data failed:', err);
        const dataBody = getEl('data-body');
        if (dataBody) {
            dataBody.innerHTML = '<div class="p-4 text-red-500">加载数据失败: ' + err.message + '</div>';
        }
    }
}

/**
 * 渲染数据表格
 */
function renderDataTable(items) {
    const body = getEl('data-body');
    if (!body) return;

    if (!items || items.length === 0) {
        body.innerHTML = '<div class="p-8 text-center text-slate-400">暂无数据</div>';
        return;
    }

    body.innerHTML = items.map(item => {
        const isTif = item.url.toLowerCase().endsWith('.tif') || item.url.toLowerCase().endsWith('.tiff');
        const thumbUrl = isTif
            ? `${API_BASE}/api/view/tif?path=${item.url.replace('/images/', '')}`
            : `${API_BASE}${item.url}`;

        const checked = isDataRowSelected(item.id) ? 'checked' : '';
        const selectedClass = isDataRowSelected(item.id) ? ' bg-blue-50/70 ring-1 ring-blue-100' : '';

        return `
            <div id="data-row-${item.id}" class="grid items-center px-6 table-row text-[14px] hover:bg-blue-50/40 transition-all gap-3 h-[calc(100%/10)]${selectedClass}" onclick="window.dispatchEvent(new CustomEvent('data-row-click', { detail: { id: ${item.id} } }))">
                <div class="flex items-center justify-center">
                    <input type="checkbox" class="data-row-checkbox h-4 w-4 accent-blue-600 cursor-pointer" data-id="${item.id}" ${checked} onchange="window.dispatchEvent(new CustomEvent('data-checkbox-change', { detail: { id: ${item.id}, checked: this.checked } }))">
                </div>
                <div class="h-10 w-14 border border-slate-100 rounded overflow-hidden shadow-sm">
                    <img src="${thumbUrl}" class="w-full h-full object-cover">
                </div>
                <div class="truncate">
                    <div class="font-extrabold text-[#181c32] truncate pr-1 text-[13px]">${item.sample_id || 'ID:' + item.id}</div>
                    <div class="text-[10px] text-slate-300 font-bold">#${item.id}</div>
                </div>
                <div class="text-center">
                    <span class="font-black text-slate-800">${item.al2o3_thickness || '--'} <small class="text-slate-400 font-medium">nm</small></span>
                    <span class="mx-1 text-slate-200">/</span>
                    <span class="font-bold text-slate-500 text-[12px]">${item.al2o3_power || '--'} <small class="text-slate-400 font-medium">W</small></span>
                </div>
                <div class="text-center">
                    <span class="font-black text-orange-600">${item.fe_thickness || '--'} <small class="text-orange-300 font-medium">nm</small></span>
                    <span class="mx-1 text-slate-200">/</span>
                    <span class="font-bold text-slate-500 text-[12px]">${item.fe_power || '--'} <small class="text-slate-400 font-medium">W</small></span>
                </div>
                <div class="text-center text-slate-700 font-bold text-[13px]">${item.ar_flow ?? '--'}<span class="text-[8px] text-slate-300 ml-0.5">s</span></div>
                <div class="text-center text-slate-700 font-bold text-[13px]">${item.h2_flow ?? '--'}<span class="text-[8px] text-slate-300 ml-0.5">s</span></div>
                <div class="text-center text-slate-700 font-bold text-[13px]">${item.c2h4_flow ?? '--'}<span class="text-[8px] text-slate-300 ml-0.5">s</span></div>
                <div class="text-center">
                    <div class="font-black text-slate-700 text-[13px]">${item.anneal_temp || '--'}<sub class="text-[9px]">C</sub></div>
                    <div class="text-[9px] text-slate-400 font-bold">${item.anneal_time || '--'}m</div>
                </div>
                <div class="text-center font-black text-blue-700 bg-blue-50/50 py-1 rounded mx-1 text-[13px]">${item.growth_temp || '--'}<sub class="text-[9px]">C</sub></div>
                <div class="text-center font-black text-blue-500 text-[13px]">${item.growth_time ? formatNumber(item.growth_time, 1) : '--'}<span class="text-[9px] ml-0.5">h</span></div>
                <div class="text-center">
                    <span class="px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 text-[10px] font-black uppercase border border-slate-200">${item.position_label || '--'}</span>
                </div>
                <div class="text-center text-slate-600 font-black text-[13px]">${item.magnification || '--'}<span class="text-[9px] ml-0.5 text-slate-300">x</span></div>
                <div class="text-center">
                    <span class="${item.processed ? 'text-emerald-500' : 'text-slate-300'} inline-flex items-center justify-center w-8 h-8 rounded-full hover:bg-slate-100 transition cursor-pointer" title="Open details" onclick="window.dispatchEvent(new CustomEvent('data-details-click', { detail: { id: ${item.id} } }))">
                        <i class="fas ${item.processed ? 'fa-check-circle' : 'fa-clock'} text-lg"></i>
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * 初始化数据列表模块
 */
export function initDataList() {
    // 事件监听已在 app.js bindGlobalEvents() 中绑定，此处无需重复
}

/**
 * 清除搜索框并重新加载
 */
export function clearSearch() {
    const searchInput = getEl('search-input');
    if (searchInput) searchInput.value = '';
    return resetAndLoad();
}

// 导出默认对象
export default {
    loadData,
    resetAndLoad,
    toggleSort,
    changePage,
    jumpToPage,
    toggleBatchMode,
    runDataBatchAnalyze,
    runDataBatchDelete,
    clearDataSelection,
    toggleSelectAllDataRows,
    toggleDataTrashView,
    renderDataTrashToggle,
    initDataList,
    clearSearch,
};
