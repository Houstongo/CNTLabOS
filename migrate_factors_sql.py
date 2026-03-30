"""
因子名称中英文迁移 - SQL版本
直接使用SQL UPDATE语句替换因子名称
"""
import sqlite3

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite"


# 工艺因子映射
PROCESS_MAP = {
    'temperature': '温度',
    'calcination_temperature': '煅烧温度',
    'flow_rate': '流速',
    'power': '功率',
    'filler_content': '填料含量',
    'metal_ion_type': '金属离子类型',
    'nitrogen_concentration': '氮浓度',
    'chemical_vapor_deposition': '化学气相沉积',
    'deposition': '沉积',
    'deposition_condition': '沉积条件',
    'reaction_solution': '反应溶液',
    'reaction_time': '反应时间',
    'pre-bending_treatment': '预弯处理',
    'reaction': '反应',
    'reduction': '还原',
    'reduction_potential': '还原电位',
    'catalyst_deactivation': '催化剂失活',
    'catalyst_agglomeration': '催化剂团聚',
    'electron_beam_lithography': '电子束光刻',
    'exposure_to_solution': '溶液暴露',
    'hydrophilicity': '亲水性',
    'hydroxylamine_assisted_deposition': '羟胺辅助沉积',
    'hydroxylamine_assisted_particle_deposition': '羟胺辅助粒子沉积',
    'nanoparticle_deposition': '纳米粒子沉积',
    'nanoparticle_variation': '纳米粒子变化',
    'particle_deposition': '粒子沉积',
    'particle_size': '粒径',
    'pH': 'pH值',
    'immersion': '浸渍',
    'measurement': '测量',
    'assembly': '组装',
    'calcination': '煅烧',
    'capacitance': '电容',
    'capacity': '容量',
    'gravimetric_capacity': '重量容量',
    'catalysis': '催化',
    'catalyst': '催化剂',
    'catalyst_activity': '催化剂活性',
    'catalytic_patterning': '催化图案化',
    'antibody_binding': '抗体结合',
    'scan_rate': '扫描速率',
    'soaking_time': '浸泡时间',
    'substrate': '基底',
    'surface_area': '表面积',
    'surface_diffusion_flux': '表面扩散通量',
    'transition_temperature_induction': '转变温度诱导',
    'thickness': '厚度',
    'voltage': '电压',
    'volume': '体积',
    'voc': '开路电压',
    'wafer_thickness': '晶圆厚度',
    'zinc_ion_transport_rate': '锌离子传输速率',
    'cdv': '化学气相沉积',
    'CVD': '化学气相沉积',
    'Raman_shift_omega_r': '拉曼位移ω_r',
    'SWNT_growth': '单壁纳米管生长',
    'XPS_study': 'XPS研究',
    'aniline_concentration': '苯胺浓度',
    'critical_buckling_forces': '临界屈曲力',
    'electrical_conductivity': '电导率',
    'peak_capacitive_currents': '峰值容性电流',
    'placement_of_electrodes': '电极放置',
    'polytetrafluoroethylene_fibrillation': '聚四氟乙烯分丝',
    'rate_performance': '速率性能',
    'redox_kinetics': '氧化还原动力学',
    'resonant_micro_Raman_spectroscopy': '共振微拉曼光谱',
    'response': '响应',
    'reversible_chemical_reaction': '可逆化学反应',
}

# 形貌因子映射
MORPHOLOGY_MAP = {
    'alignment': '取向度',
    'density': '密度',
    'diameter': '管径',
    'curvature': '波曲度',
    'tortuosity': '曲折度',
    'height': '高度',
    'substrate_bending_deformation': '基底弯曲变形',
    'bending_deformation': '弯曲变形',
    'morphology': '形貌',
    'compartment_distance': '间距距离',
    'particle_size': '粒径',
    'patterned_growth': '图案化生长',
    'patterned_regions': '图案区域',
}

# 性能因子映射
PERFORMANCE_MAP = {
    'conductivity': '电导率',
    'separation_properties': '分离性能',
    'cycling_stability': '循环稳定性',
    'volumetric_performance': '体积性能',
    'interfacial_interaction': '界面相互作用',
    'sheet_resistance': '方块电阻',
    'tensile_strength': '抗拉强度',
    'modulus': '弹性模量',
    'resistivity': '电阻率',
    'electrical_conductivity': '电导率',
}

# 机理因子映射
MECHANISM_MAP = {
    'diffusion': '扩散',
    'interfacial_interaction': '界面相互作用',
    'filler_content': '填料含量',
    'growth_kinetics': '生长动力学',
    'boundary_layer_effect': '边界层效应',
    'reversible_chemical_reaction': '可逆化学反应',
}


def migrate_kb_links():
    """迁移kb_links表的因子字段"""
    conn = sqlite3.connect(DB_PATH)

    print("=== 开始迁移 kb_links 表因子字段 ===")
    updated_count = 0

    # 工艺因子
    for eng, cn in PROCESS_MAP.items():
        cursor = conn.execute(
            "UPDATE kb_links SET process_factor = ? WHERE process_factor = ?",
            (cn, eng)
        )
        updated_count += cursor.rowcount

    # 形貌因子
    for eng, cn in MORPHOLOGY_MAP.items():
        cursor = conn.execute(
            "UPDATE kb_links SET morphology_factor = ? WHERE morphology_factor = ?",
            (cn, eng)
        )
        updated_count += cursor.rowcount

    # 性能因子
    for eng, cn in PERFORMANCE_MAP.items():
        cursor = conn.execute(
            "UPDATE kb_links SET performance_factor = ? WHERE performance_factor = ?",
            (cn, eng)
        )
        updated_count += cursor.rowcount

    # 机理因子（用process_factor存储）
    for eng, cn in MECHANISM_MAP.items():
        cursor = conn.execute(
            "UPDATE kb_links SET process_factor = ? WHERE process_factor = ?",
            (cn, eng)
        )
        updated_count += cursor.rowcount

    conn.commit()
    print(f"kb_links 因子迁移完成: {updated_count} 条记录")


def migrate_kb_msfu():
    """迁移kb_msfu表的实体字段"""
    conn = sqlite3.connect(DB_PATH)

    print("\n=== 开始迁移 kb_msfu 表实体字段 ===")
    updated_count = 0

    # 读取所有需要迁移的实体
    rows = conn.execute("""
        SELECT id, source_entity, target_entity
        FROM kb_msfu
    """).fetchall()

    for row in rows:
        row_id = row['id']
        source_entity = row['source_entity']
        target_entity = row['target_entity']

        # 解析并转换source_entity
        if ':' in source_entity:
            entity_type, factor_name = source_entity.split(':', 1)
            entity_type = entity_type.strip().lower()

            # 转换因子名称
            factor_cn = None
            if factor_name in PROCESS_MAP:
                factor_cn = PROCESS_MAP[factor_name]
            elif factor_name in MORPHOLOGY_MAP:
                factor_cn = MORPHOLOGY_MAP[factor_name]
            elif factor_name in PERFORMANCE_MAP:
                factor_cn = PERFORMANCE_MAP[factor_name]
            elif factor_name in MECHANISM_MAP:
                factor_cn = MECHANISM_MAP[factor_name]

            # 转换实体类型
            entity_type_map = {
                'process': '工艺',
                'morphology': '形貌',
                'performance': '性能',
                'mechanism': '机理',
                'evidence': '证据'
            }
            entity_type_cn = entity_type_map.get(entity_type, entity_type)

            if factor_cn:
                source_entity_cn = f"{entity_type_cn}:{factor_cn}"
                conn.execute(
                    "UPDATE kb_msfu SET source_entity = ? WHERE id = ?",
                    (source_entity_cn, row_id)
                )
                updated_count += 1

        # 同样处理target_entity
        if ':' in target_entity:
            entity_type, factor_name = target_entity.split(':', 1)
            entity_type = entity_type.strip().lower()

            factor_cn = None
            if factor_name in PROCESS_MAP:
                factor_cn = PROCESS_MAP[factor_name]
            elif factor_name in MORPHOLOGY_MAP:
                factor_cn = MORPHOLOGY_MAP[factor_name]
            elif factor_name in PERFORMANCE_MAP:
                factor_cn = PERFORMANCE_MAP[factor_name]
            elif factor_name in MECHANISM_MAP:
                factor_cn = MECHANISM_MAP[factor_name]

            entity_type_map = {
                'process': '工艺',
                'morphology': '形貌',
                'performance': '性能',
                'mechanism': '机理',
                'evidence': '证据'
            }
            entity_type_cn = entity_type_map.get(entity_type, entity_type)

            if factor_cn:
                target_entity_cn = f"{entity_type_cn}:{factor_cn}"
                conn.execute(
                    "UPDATE kb_msfu SET target_entity = ? WHERE id = ?",
                    (target_entity_cn, row_id)
                )
                updated_count += 1

    conn.commit()
    print(f"kb_msfu 实体迁移完成: {updated_count} 条字段")

    conn.close()


if __name__ == "__main__":
    print("因子名称中英文迁移工具")
    print("=" * 50)

    # 迁移kb_links
    migrate_kb_links()

    # 迁移kb_msfu
    migrate_kb_msfu()

    print("\n所有迁移完成！")
    print("请重新启动后端服务以应用更改。")
