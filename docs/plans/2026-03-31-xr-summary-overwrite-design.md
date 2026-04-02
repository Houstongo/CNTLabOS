# XR Summary Overwrite Design

## Goal

将 `slice_standard_batch_20260331_005741/summary.csv` 中当前确认可用的 XR 新特征回写到 `images` 表：

- 旧字段直接覆盖
- 新口径特征新增字段保存
- 直径主字段使用平均值口径

## Confirmed Mapping

直接覆盖旧字段：

- `density <- summary.density`
- `alignment <- summary.alignment`
- `diameter <- summary.diameter_mean_nm`
- `curvature <- summary.l2_curvature_trimmed_mean_sqrt_length_nm * 1000` （统一存为 `um^-1`）
- `tortuosity <- summary.l2_tortuosity_v2`
- `waviness_ratio <- summary.l2_waviness_ratio_v2`

补充填充已有但基本为空的直径统计字段：

- `diameter_mean <- summary.diameter_mean_nm`
- `diameter_std <- summary.diameter_std_nm`
- `diameter_min <- summary.diameter_min_nm`
- `diameter_max <- summary.diameter_max_nm`
- `diameter_p50 <- summary.diameter_p50_nm`
- `diameter_p75 <- summary.diameter_p75_nm`

新增字段保存新口径：

- `junction_count`
- `junction_ratio`
- `skeleton_length_px`
- `skeleton_length_um`
- `branch_count`
- `curvature_label`
- `curvature_p70`
- `curvature_mean`
- `curvature_trimmed_mean`
- `diameter_p30_nm_v2`
- `l2_branch_count`
- `l2_curvature_label`
- `l2_curvature_p70_sqrt_length_nm`
- `l2_curvature_mean_sqrt_length_nm`
- `l2_curvature_trimmed_mean_sqrt_length_nm`
- `l2_waviness_ratio_v2`
- `l2_tortuosity_v2`
- `xr_feature_report_tag`

## Rationale

这次用户明确要求“直接覆盖”，因此旧的 `curvature/tortuosity/waviness_ratio/diameter` 不再保留 legacy 数值作为主值。为了避免后续论文和建模再次混淆，同时把当前方法的显式新口径以独立列保存在库内。

同时增加一组不带版本名的简洁主字段，供数据库查询、论文写作和建模直接使用：

- `branch_count`
- `curvature_label`
- `curvature_p70`
- `curvature_mean`
- `curvature_trimmed_mean`

其中这些简洁字段当前统一对应 L2 + sqrt-length 统计口径。
其中所有数据库主曲率字段统一使用 `um^-1`，不再使用 `nm^-1` 作为数据库展示/建模单位。

## Safety

- 只处理 `source='XR'`
- 只处理 `summary.csv` 中 `status='success'`
- 使用临时目录复制数据库后写回，规避当前环境偶发的 SQLite `disk I/O error`
- 脚本支持重复执行，采用幂等更新
