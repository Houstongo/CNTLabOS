// 格式化工具函数

/**
 * 格式化数值，保留小数位
 */
export function formatNumber(value, digits = 2) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : '--';
}

/**
 * 格式化 ML 数值（兼容原 fmtMl 函数）
 */
export function fmtMl(value, digits = 2) {
    return formatNumber(value, digits);
}

/**
 * 格式化带单位的数值
 */
export function formatWithUnit(value, unit = '', digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n.toFixed(digits)}${unit}`;
}

/**
 * 格式化百分比
 */
export function formatPercent(value, digits = 1) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n.toFixed(digits)}%`;
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes) {
    if (bytes == null || isNaN(bytes)) return '--';
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * 格式化日期时间
 */
export function formatDate(timestamp, format = 'datetime') {
    if (timestamp == null) return '--';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return '--';

    const pad = (n) => String(n).padStart(2, '0');

    if (format === 'date') {
        return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
    }
    if (format === 'time') {
        return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
    }
    if (format === 'datetime') {
        return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    }
    return date.toLocaleString('zh-CN');
}

/**
 * 格式化相对时间
 */
export function formatRelativeTime(timestamp) {
    if (timestamp == null) return '--';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return '--';

    const now = new Date();
    const diff = now - date;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) return '刚刚';
    if (minutes < 60) return `${minutes}分钟前`;
    if (hours < 24) return `${hours}小时前`;
    if (days < 30) return `${days}天前`;
    return formatDate(timestamp, 'date');
}

/**
 * 格式化持续时间
 */
export function formatDuration(seconds) {
    if (seconds == null || isNaN(seconds)) return '--';
    if (seconds < 60) return `${Math.round(seconds)}秒`;
    if (seconds < 3600) {
        const m = Math.floor(seconds / 60);
        const s = Math.round(seconds % 60);
        return `${m}分${s}秒`;
    }
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}小时${m}分`;
}

/**
 * 格式化数量（添加千位分隔符）
 */
export function formatCount(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return n.toLocaleString('zh-CN');
}

/**
 * 截断文本
 */
export function truncate(text, maxLength = 50, suffix = '...') {
    if (text == null) return '';
    const str = String(text);
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength) + suffix;
}

/**
 * 格式化分数 (如 "3/10")
 */
export function formatFraction(numerator, denominator) {
    if (denominator == null || denominator === 0) return '--';
    return `${numerator}/${denominator}`;
}

/**
 * 格式化范围值 (如 "10-20")
 */
export function formatRange(min, max, unit = '') {
    if (min == null || max == null) return '--';
    if (min === max) return `${min}${unit}`;
    return `${min}-${max}${unit}`;
}

/**
 * 格式化置信度
 */
export function formatConfidence(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${(n * 100).toFixed(digits)}%`;
}

/**
 * 格式化温度
 */
export function formatTemp(value, unit = '℃') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n.toFixed(1)}${unit}`;
}

/**
 * 格式化功率
 */
export function formatPower(value, unit = 'W') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n}${unit}`;
}

/**
 * 格式化厚度
 */
export function formatThickness(value, unit = 'nm') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n}${unit}`;
}

/**
 * 格式化流速
 */
export function formatFlow(value, unit = 'sccm') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n}${unit}`;
}

/**
 * 格式化倍率
 */
export function formatMagnification(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${Math.round(n)}x`;
}

/**
 * 格式化时间 (小时)
 */
export function formatTime(value, unit = 'h') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n.toFixed(1)}${unit}`;
}

/**
 * 格式化分钟
 */
export function formatMinutes(value, unit = 'min') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n}${unit}`;
}

/**
 * 格式化位置 (cm)
 */
export function formatPosition(value, unit = 'cm') {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    return `${n.toFixed(1)}${unit}`;
}

/**
 * 检查值是否为空或未定义
 */
export function isEmpty(value) {
    return value == null || value === '' || (Array.isArray(value) && value.length === 0);
}

/**
 * 安全格式化：确保返回字符串
 */
export function safeFormat(value, defaultValue = '--') {
    if (value == null || value === '') return defaultValue;
    return String(value);
}
