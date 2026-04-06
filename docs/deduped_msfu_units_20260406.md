# 去重后的 MSFU 单元整理

## 说明

- 数据来源：`database/cnta_knowledge_base.sqlite` 的 `kb_msfu` 表
- 统计日期：`2026-04-06`
- 去重规则：按 `source_entity + relation_type + target_entity` 视为同一 MSFU 单元
- 去重后 MSFU 单元总数：`203`
- 保留字段：出现频次、最高置信度、平均置信度、方向汇总

## 关系类型分布

- `促进`：94 个
- `影响`：49 个
- `抑制`：36 个
- `导致`：13 个
- `未知`：5 个
- `不导致`：1 个
- `增大`：1 个
- `模式化`：1 个
- `测量`：1 个
- `涉及`：1 个
- `表征`：1 个

## 促进

- `工艺:CVD` 促进 `形貌:高度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:aniline_concentration` 促进 `形貌:disorder_of_cnt_surface`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:aniline_concentration` 促进 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:antibody_binding` 促进 `工艺:cation_flux`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:calcination_temperature` 促进 `形貌:particle_size`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:capacitance` 促进 `形貌:area`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:capacity` 促进 `工艺:reversible_level`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:catalysis` 促进 `形貌:single-walled_nanotubes`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:catalyst` 促进 `工艺:SWNT_growth`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:catalyst` 促进 `形貌:formation`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:catalyst` 促进 `机理:nucleation`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:catalytic_patterning` 促进 `性能:electronic_devices`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:cdv` 促进 `工艺:dispersity`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:chemical_vapor_deposition` 促进 `形貌:patterned_growth`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:deposition` 促进 `工艺:characterization`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:deposition` 促进 `工艺:device_fabrication`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:deposition` 促进 `形貌:characteristics`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:deposition_condition` 促进 `形貌:surface_cleanliness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:exposure_to_solution` 促进 `形貌:size`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:filler_content` 促进 `机理:interfacial_interaction`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `positive`
- `工艺:flow_rate` 促进 `工艺:temperature`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:hydroxylamine_assisted_particle_deposition` 促进 `形貌:high_density_arrays`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:ilayer_radius` 促进 `工艺:bilayer_tension`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:immersion` 促进 `工艺:nanoparticle_formation`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:measurement` 促进 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:patterned_growth` 促进 `形貌:cleanliness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:peak_capacitive_currents` 促进 `工艺:monotonic_change`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `positive`
- `工艺:polytetrafluoroethylene_fibrillation` 促进 `形貌:dispersity`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:power` 促进 `工艺:contrast`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:raman_d_and_g_band_intensity_ratio` 促进 `工艺:nitrogen_concentration_during_cnt_growth`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:rate_performance` 促进 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:reaction` 促进 `形貌:nanoparticle_density`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:reaction` 促进 `形貌:percentage_of_cnts`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:reaction` 促进 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:redox_kinetics` 促进 `工艺:current_response`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:reduction` 促进 `性能:电导率`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:response` 促进 `工艺:amplitude`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:response` 促进 `工艺:speed`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:scan_rate` 促进 `工艺:peak_area`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:scan_rate` 促进 `性能:电导率`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:soaking_time` 促进 `形貌:nanoparticle_size`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:soaking_time` 促进 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:surface_area` 促进 `工艺:impedance_features`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:surface_diffusion_flux` 促进 `性能:电导率`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature` 促进 `工艺:electric_output`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:temperature` 促进 `工艺:ionic_conductivity`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature` 促进 `工艺:reaction_temperature`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature` 促进 `形貌:密度`；频次 `9`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature` 促进 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature_step_induction` 促进 `工艺:natural_temperature_change`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:transition_detection` 促进 `工艺:natural_transition_temperature`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:voc` 促进 `工艺:short_circuit_current_density`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `positive`
- `工艺:volume` 促进 `形貌:area`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:volumetric_performance` 促进 `形貌:carbon_content`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:wafer_thickness` 促进 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:zinc_ion_transport_rate` 促进 `工艺:electrochemical_performance`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:氩气流量` 促进 `形貌:密度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:生长时间` 促进 `形貌:取向度`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:生长时间` 促进 `形貌:密度`；频次 `7`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:生长时间` 促进 `形貌:波曲度`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:生长时间` 促进 `形貌:管径`；频次 `1`；最高置信度 `0.65`；平均置信度 `0.65`；方向 `positive`
- `工艺:生长时间` 促进 `形貌:高度`；频次 `6`；最高置信度 `0.65`；平均置信度 `0.567`；方向 `positive`
- `工艺:生长温度` 促进 `形貌:取向度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:生长温度` 促进 `形貌:管径`；频次 `9`；最高置信度 `0.65`；平均置信度 `0.561`；方向 `positive`
- `工艺:生长温度` 促进 `形貌:高度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:管径` 促进 `形貌:radial_breathing_mode`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:退火时间` 促进 `形貌:取向度`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:退火时间` 促进 `形貌:密度`；频次 `3`；最高置信度 `0.8`；平均置信度 `0.633`；方向 `positive`
- `工艺:退火时间` 促进 `形貌:管径`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:铁催化剂厚度` 促进 `形貌:取向度`；频次 `10`；最高置信度 `0.65`；平均置信度 `0.56`；方向 `positive`
- `工艺:铁催化剂厚度` 促进 `形貌:密度`；频次 `22`；最高置信度 `0.65`；平均置信度 `0.559`；方向 `positive`
- `工艺:铁催化剂厚度` 促进 `形貌:波曲度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `工艺:铁催化剂厚度` 促进 `形貌:管径`；频次 `18`；最高置信度 `0.65`；平均置信度 `0.556`；方向 `positive`
- `工艺:铁催化剂厚度` 促进 `形貌:高度`；频次 `10`；最高置信度 `0.65`；平均置信度 `0.56`；方向 `positive`
- `形貌:取向度` 促进 `性能:弹性模量`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:取向度` 促进 `性能:抗拉强度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:取向度` 促进 `性能:电导率`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:密度` 促进 `工艺:energy_volumetric_density`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `形貌:密度` 促进 `性能:弹性模量`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:密度` 促进 `性能:抗拉强度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:密度` 促进 `性能:电导率`；频次 `10`；最高置信度 `0.65`；平均置信度 `0.57`；方向 `positive`
- `形貌:曲折度` 促进 `性能:弹性模量`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:曲折度` 促进 `性能:电导率`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:波曲度` 促进 `性能:弹性模量`；频次 `4`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:波曲度` 促进 `性能:电导率`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:管径` 促进 `形貌:密度`；频次 `1`；最高置信度 `0.7`；平均置信度 `0.7`；方向 `positive`
- `形貌:管径` 促进 `性能:弹性模量`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:管径` 促进 `性能:抗拉强度`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:管径` 促进 `性能:电导率`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:高度` 促进 `性能:弹性模量`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `形貌:高度` 促进 `性能:抗拉强度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `positive`
- `性能:electrical_conductivity` 促进 `性能:internal_lithium_ion_transport_efficiency`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `性能:电导率` 促进 `性能:cycling_stability`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `机理:reversible_chemical_reaction` 促进 `性能:separation_properties`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `positive`

## 影响

- `工艺:SWNT_growth` 影响 `形貌:nanotube_pattern`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:SWNT_growth` 影响 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
- `工艺:XPS_study` 影响 `机理:oxidation_state_identification`；频次 `1`；最高置信度 `0.6`；平均置信度 `0.6`；方向 `unknown`
- `工艺:calcination` 影响 `形貌:管径`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `unknown`
- `工艺:calcination_temperature` 影响 `形貌:nanoparticle_height`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:catalyst` 影响 `形貌:pattern`；频次 `1`；最高置信度 `0.7`；平均置信度 `0.7`；方向 `unknown`
- `工艺:deposition` 影响 `形貌:管径`；频次 `1`；最高置信度 `0.7`；平均置信度 `0.7`；方向 `unknown`
- `工艺:hydroxylamine_assisted_deposition` 影响 `形貌:film_thickness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
- `工艺:metal_ion_type` 影响 `形貌:morphology`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `neutral`
- `工艺:nanoparticle_deposition` 影响 `工艺:solution_pH`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `unknown`
- `工艺:nanoparticle_deposition` 影响 `形貌:particle_distribution`；频次 `1`；最高置信度 `0.7`；平均置信度 `0.7`；方向 `unknown`
- `工艺:nanoparticle_deposition` 影响 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:nanoparticle_deposition` 影响 `形貌:高度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:nanoparticle_variation` 影响 `形貌:film_thickness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
- `工艺:nucleation` 影响 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `unknown`
- `工艺:pH` 影响 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:particle_deposition` 影响 `形貌:nanoparticle_deposition`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
- `工艺:patterned_regions` 影响 `形貌:cleanliness_of_catalyst_regions`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:placement_of_electrodes` 影响 `性能:yield_of_electrical_devices`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:reaction_solution` 影响 `形貌:密度`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `unknown`
- `工艺:reaction_time` 影响 `工艺:depositio`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `unknown`
- `工艺:reaction_time` 影响 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:soaking_time` 影响 `形貌:AFM_image`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:soaking_time` 影响 `形貌:nanoparticle_reduction`；频次 `1`；最高置信度 `0.7`；平均置信度 `0.7`；方向 `unknown`
- `工艺:soaking_time` 影响 `形貌:population`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `unknown`
- `工艺:soaking_time` 影响 `形貌:高度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:substrate` 影响 `形貌:cleanliness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:substrate` 影响 `形貌:imaging_quality`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:substrate` 影响 `形貌:particle_free`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:temperature` 影响 `形貌:密度`；频次 `2`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
- `工艺:voltage` 影响 `工艺:time_constant`；频次 `4`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `unknown`
- `机理:催化剂团聚` 影响 `形貌:取向度`；频次 `3`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `机理:催化剂团聚` 影响 `形貌:密度`；频次 `2`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `机理:催化剂团聚` 影响 `形貌:管径`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `机理:催化剂失活` 影响 `形貌:取向度`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative`
- `机理:催化剂失活` 影响 `形貌:密度`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `机理:催化剂失活` 影响 `形貌:波曲度`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative`
- `机理:催化剂失活` 影响 `形貌:管径`；频次 `3`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `机理:催化剂失活` 影响 `形貌:高度`；频次 `2`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `机理:扩散` 影响 `形貌:取向度`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `机理:扩散` 影响 `形貌:密度`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `机理:扩散` 影响 `形貌:曲折度`；频次 `2`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `机理:扩散` 影响 `形貌:管径`；频次 `3`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `机理:扩散` 影响 `形貌:高度`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `机理:生长动力学` 影响 `形貌:取向度`；频次 `2`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `机理:生长动力学` 影响 `形貌:密度`；频次 `6`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `机理:生长动力学` 影响 `形貌:曲折度`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative`
- `机理:生长动力学` 影响 `形貌:管径`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative`
- `机理:生长动力学` 影响 `形貌:高度`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`

## 抑制

- `工艺:antibody_binding` 抑制 `工艺:anion_flux`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:assembly` 抑制 `工艺:steps`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:capacity` 抑制 `工艺:early_cycles`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:catalyst_activity` 抑制 `机理:structure`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:critical_buckling_forces` 抑制 `形貌:length`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:gravimetric_capacity` 抑制 `形貌:capacity`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:hydrophilicity` 抑制 `工艺:evaporation`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`
- `工艺:nitrogen_concentration` 抑制 `形貌:compartment_distance`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `negative`
- `工艺:particle_size` 抑制 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:pre-bending_treatment` 抑制 `形貌:substrate_bending`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `negative`
- `工艺:reaction` 抑制 `形貌:percentage_of_scnms`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:soaking_time` 抑制 `形貌:particle_deposition`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:soaking_time` 抑制 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:temperature` 抑制 `形貌:crystallite_size`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:temperature` 抑制 `形貌:thermal_stability`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:temperature` 抑制 `形貌:密度`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:thickness` 抑制 `工艺:electrocompression`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:thickness` 抑制 `形貌:thickness`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:生长时间` 抑制 `形貌:取向度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `工艺:生长时间` 抑制 `形貌:密度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `工艺:生长温度` 抑制 `形貌:取向度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `工艺:退火时间` 抑制 `形貌:particle_deposition`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`
- `工艺:铁催化剂厚度` 抑制 `形貌:取向度`；频次 `10`；最高置信度 `0.65`；平均置信度 `0.59`；方向 `negative`
- `工艺:铁催化剂厚度` 抑制 `形貌:密度`；频次 `10`；最高置信度 `0.65`；平均置信度 `0.57`；方向 `negative`
- `工艺:铁催化剂厚度` 抑制 `形貌:管径`；频次 `5`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `工艺:铁催化剂厚度` 抑制 `形貌:高度`；频次 `1`；最高置信度 `0.65`；平均置信度 `0.65`；方向 `negative`
- `形貌:取向度` 抑制 `性能:抗拉强度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:取向度` 抑制 `性能:电导率`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:密度` 抑制 `性能:弹性模量`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:密度` 抑制 `性能:抗拉强度`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:密度` 抑制 `性能:电导率`；频次 `2`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:曲折度` 抑制 `性能:电导率`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:波曲度` 抑制 `性能:弹性模量`；频次 `3`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:波曲度` 抑制 `性能:抗拉强度`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:管径` 抑制 `性能:弹性模量`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`
- `形貌:管径` 抑制 `性能:电导率`；频次 `1`；最高置信度 `0.55`；平均置信度 `0.55`；方向 `negative`

## 导致

- `工艺:chemical_vapor_deposition` 导致 `形貌:bending_deformation`；频次 `1`；最高置信度 `0.9`；平均置信度 `0.9`；方向 `positive`
- `工艺:生长时间` 导致 `机理:催化剂团聚`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `工艺:生长时间` 导致 `机理:催化剂失活`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `工艺:生长时间` 导致 `机理:扩散`；频次 `8`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `工艺:生长时间` 导致 `机理:生长动力学`；频次 `3`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `工艺:生长温度` 导致 `机理:催化剂团聚`；频次 `2`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `工艺:生长温度` 导致 `机理:生长动力学`；频次 `3`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `工艺:退火时间` 导致 `机理:催化剂失活`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative`
- `工艺:退火时间` 导致 `机理:扩散`；频次 `1`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive`
- `工艺:铁催化剂厚度` 导致 `机理:催化剂团聚`；频次 `9`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `工艺:铁催化剂厚度` 导致 `机理:催化剂失活`；频次 `15`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `negative,positive`
- `工艺:铁催化剂厚度` 导致 `机理:扩散`；频次 `21`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`
- `工艺:铁催化剂厚度` 导致 `机理:生长动力学`；频次 `4`；最高置信度 `0.73`；平均置信度 `0.73`；方向 `positive,negative`

## 未知

- `工艺:reduction_potential` 未知 `形貌:unknown`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `unknown`
- `工艺:temperature` 未知 `形貌:cleanliness`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `neutral`
- `工艺:temperature` 未知 `形貌:取向度`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `neutral`
- `工艺:temperature` 未知 `形貌:密度`；频次 `3`；最高置信度 `0.8`；平均置信度 `0.6`；方向 `neutral,negative`
- `工艺:temperature` 未知 `形貌:管径`；频次 `1`；最高置信度 `0.5`；平均置信度 `0.5`；方向 `neutral`

## 不导致

- `工艺:exposure_to_solution` 不导致 `机理:nucleation`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `negative`

## 增大

- `工艺:reduction` 增大 `形貌:particle_size`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `positive`

## 模式化

- `工艺:electron_beam_lithography` 模式化 `形貌:wells`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`

## 测量

- `工艺:Raman_shift_omega_r` 测量 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`

## 涉及

- `工艺:deposition` 涉及 `工艺:surface_hydroxyl_groups`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`

## 表征

- `工艺:resonant_micro-Raman_spectroscopy` 表征 `形貌:管径`；频次 `1`；最高置信度 `0.8`；平均置信度 `0.8`；方向 `neutral`
