// 系统常量定义
export const API_BASE = window.location.origin || 'http://127.0.0.1:8000';

// 分页配置
export const DEFAULT_LIMIT = 10;

// 默认排序配置
export const DEFAULT_SORT = { field: 'id', order: 'desc' };

// RAG 相关常量
export const RAG_TOP_K = 20;

// 算法可视化步骤
export const ALGO_STEPS = [
    { name: '灰度化', key: 'grayscale' },
    { name: '高斯模糊', key: 'blur' },
    { name: 'Canny 边缘检测', key: 'canny' },
    { name: 'Hough 变换', key: 'hough' },
    { name: '特征提取', key: 'feature' },
];

// 数据源配置
export const SOURCE_CONFIG = {
    '': { label: '全部数据', icon: 'fa-layer-group' },
    'XR': { label: 'XR 梯度序列', icon: 'fa-chart-line' },
    'ZZY': { label: 'ZZY 参数序列', icon: 'fa-database' },
};

// 删除视图模式
export const DELETION_VIEWS = {
    active: 'active',
    deleted: 'deleted',
};

// AI 提供商配置
export const AI_PROVIDERS = {
    glm: { name: 'GLM', color: 'purple' },
    deepseek: { name: 'DeepSeek', color: 'blue' },
};

// ML 目标元数据
export const ML_TARGET_META = {
    diameter: { label: '管径分布', predKey: 'diameter_pred', actualKey: 'diameter_actual' },
    density: { label: '覆盖密度', predKey: 'density_pred', actualKey: 'density_actual' },
    alignment: { label: '取向度', predKey: 'alignment_pred', actualKey: 'alignment_actual' },
    curvature: { label: '平均曲率 (um^-1)', predKey: 'curvature_pred', actualKey: 'curvature_actual' },
    tortuosity: { label: '波浪度', predKey: 'tortuosity_pred', actualKey: 'tortuosity_actual' },
    waviness_ratio: { label: '波曲度', predKey: 'waviness_ratio_pred', actualKey: 'waviness_ratio_actual' },
};

// ML X 轴元数据
export const ML_X_META = {
    actual_temp: { label: '实际温度', unit: 'C' },
    flow_rate: { label: '流速', unit: '' },
    catalyst_concentration: { label: '催化剂浓度', unit: '' },
    sample_no: { label: '样品编号', unit: '' },
    shot_no: { label: '拍摄编号', unit: '' },
};

// ML Y 模式元数据
export const ML_Y_MODE_META = {
    pred: '模型预测',
    actual: '实测标签',
    residual: '偏差(实测-预测)',
};
