"""
增强的规则模式 - 扩展原有规则以提高MSFU提取覆盖率
解决：关系模式简单、实体匹配缺乏上下文、机制覆盖不足等问题
"""

from enum import Enum


class RelationCategory(Enum):
    """关系类别（新增）"""
    INCREASE = "increase"
    DECREASE = "decrease"
    CAUSE = "cause"
    AFFECT = "affect"
    PROMOTE = "promote"
    INHIBIT = "inhibit"
    NEGATION = "negation"  # 新增：否定关系


# ==================== 扩展的关系模式 ====================

ENHANCED_RELATION_PATTERNS = [
    # ========== 1. 原有模式（保留） ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            r"(?P<src>.+?)\s*(?:增加|提高|enhance|increase|improve)\s*(?P<tgt>.+?)(?:$|[。.!?;；])",
            r"(?P<tgt>.+?)\s*(?:随着|with)\s*(?P<src>.+?)\s*(?:增加|提高|increase)",
        ],
        "source": "original",
    },

    {
        "category": RelationCategory.DECREASE.value,
        "relation_type": "decreases",
        "direction": "negative",
        "patterns": [
            r"(?P<src>.+?)\s*(?:减少|降低|decrease|reduce)\s*(?P<tgt>.+?)(?:$|[。.!?;；])",
            r"(?P<tgt>.+?)\s*(?:随着|with)\s*(?P<src>.+?)\s*(?:减少|降低|decrease)",
        ],
        "source": "original",
    },

    # ========== 2. 新增：名词化结构 ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            # "Increase of X improves Y"
            r"(?:increase|increase of|increase in)\s+(?P<src>.+?)\s+(?:leads to|results in|improves|enhances)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "The increase in X..."
            r"(?:the\s+)?(?:increase|elevation|enhancement)\s+(?:of|in)\s+(?P<src>.+?)\s+(?:enhances|improves)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "X increase results in..."
            r"(?P<src>.+?)\s+(?:increase|elevation)\s+(?:leads to|results in)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_nominalization",
    },

    {
        "category": RelationCategory.DECREASE.value,
        "relation_type": "decreases",
        "direction": "negative",
        "patterns": [
            # "Decrease of X reduces Y"
            r"(?:decrease|decrease of|decrease in)\s+(?P<src>.+?)\s+(?:leads to|results in|reduces)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "The decrease in X..."
            r"(?:the\s+)?(?:decrease|reduction)\s+(?:of|in)\s+(?P<src>.+?)\s+(?:reduces)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_nominalization",
    },

    # ========== 3. 新增：复合谓词 ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            # "X enhances A and B"（拆分为两个关系）
            r"(?P<src>.+?)\s*(?:enhances|improves|increases)\s+(?P<tgt1>.+?)\s+(?:and|while|but)\s+(?:reduces|decreases)\s+(?P<tgt2>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_composite",
        "split_multiple": True,  # 标记为需要拆分
    },

    {
        "category": RelationCategory.DECREASE.value,
        "relation_type": "decreases",
        "direction": "negative",
        "patterns": [
            # "X reduces A and B"
            r"(?P<src>.+?)\s*(?:reduces|decreases)\s+(?P<tgt>.+?)\s+and\s+(?P<tgt2>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_composite",
        "split_multiple": True,
    },

    # ========== 4. 新增：被动语态 ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            # "Alignment is improved by higher temperature"
            r"(?P<tgt>.+?)\s+(?:is|was|are|were)\s+(?:improved|enhanced|increased)\s+by\s+(?P<src>.+?)(?:$|[。.!?;；])",
            # "Alignment can be improved by..."
            r"(?P<tgt>.+?)\s+can\s+be\s+(?:improved|enhanced)\s+by\s+(?P<src>.+?)(?:$|[。.!?;；])",
            # "中文被字句"
            r"(?P<tgt>.+?)(?:被|由)\s+(?P<src>.+?)(?:提高|增加|增强)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_passive",
    },

    # ========== 5. 新增：比较句式 ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            # "Higher X leads to better Y"
            r"(?:higher|greater|increased)\s+(?P<src>.+?)\s+(?:leads to|results in)\s+(?:higher|greater|better|improved)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "X increases compared to..."
            r"(?P<src>.+?)\s+(?:increases|improves)\s+(?:compared to|relative to|vs\.?)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_comparison",
    },

    # ========== 6. 新增：否定处理 ==========
    {
        "category": RelationCategory.NEGATION.value,
        "relation_type": "does_not_affect",
        "direction": "neutral",
        "patterns": [
            # "X does not affect Y"
            r"(?P<src>.+?)\s+(?:does not|doesn't|does\s+not)\s+(?:affect|influence|impact)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "X has no effect on Y"
            r"(?P<src>.+?)\s+(?:has no effect|has little effect)\s+on\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
            # "中文否定"
            r"(?P<src>.+?)\s+(?:没有|不|未)\s*(?:影响|affect)\s+(?P<tgt>.+?)(?:$|[。.!?;；])",
        ],
        "source": "enhanced_negation",
        "is_negation": True,  # 标记为否定关系
    },

    # ========== 7. 新增：条件句式 ==========
    {
        "category": RelationCategory.INCREASE.value,
        "relation_type": "increases",
        "direction": "positive",
        "patterns": [
            # "When X > 750, Y improves"
            r"(?:when|while|at)\s+(?P<src>.+?)\s*[>≥]\s*(?P<value>\d+\.?\d*)\s*(?:,|;)\s*(?P<tgt>.+?)\s+(?:improves|enhances|increases)",
            # "In the presence of X, Y..."
            r"(?:in the presence of|with)\s+(?P<src>.+?)\s*,\s*(?P<tgt>.+?)\s+(?:is improved|improves|enhances)",
        ],
        "source": "enhanced_conditional",
        "extract_condition": True,  # 标记为需要提取条件
    },
]


# ==================== 扩展的实体模式（添加上下文） ====================

ENHANCED_ENTITY_PATTERNS = {
    "process": {
        # 生长温度（添加上下文区分）
        "growth_temp": [
            r"\bgrowth\s+temperature\b",
            r"(?:反应|生长)\s*温度\b",
            r"(?:CVD|化学气相沉积).*?温度",
        ],
        # 退火温度（新增）
        "anneal_temp": [
            r"\bannealing\s+temperature\b",
            r"(?:退火|热处理)\s*温度\b",
            r"(?:post-anneal|post-annealing).*?温度",
        ],
        # 其他工艺参数（保持原样）
        "growth_time": [r"\bgrowth time\b", r"\btime\b", r"生长时间"],
        "ar_flow": [r"\bar\b", r"氩", r"氩气"],
        "h2_flow": [r"\bh2\b", r"氢", r"氢气"],
        "c2h4_flow": [r"\bc2h4\b", r"乙烯"],
        "fe_thickness": [r"\bfe\s*thickness\b", r"\bcatalyst\s*thickness\b", r"铁\s*厚度", r"催化剂厚度"],
        "al2o3_thickness": [r"\bal2o3\s*thickness\b", r"\bAl₂O₃\s*thickness\b", r"氧化铝\s*厚度", r"支撑层\s*厚度"],
    },

    "morphology": {
        # 添加更多同义词和变体
        "alignment": [
            r"\balignment\b",
            r"\borientation\b",  # 新增
            r"\bverticality\b",  # 新增
            r"取向",
            r"对齐",
        ],
        "density": [
            r"\bdensity\b",
            r"\bcoverage\b",  # 新增
            r"\bpacking density\b",  # 新增
            r"密度",
            r"覆盖率",
        ],
        "diameter": [
            r"\bdiameter\b",
            r"\btube\s*diameter\b",
            r"管径",
            r"直径",
            r"\bnanotube\s*size\b",  # 新增
        ],
        "curvature": [
            r"\bcurvature\b",
            r"\bwaviness\b",
            r"\btortuosity\b",  # 新增（曲折度）
            r"弯曲",
            r"波曲",
        ],
        # 新增：高度/长度
        "height": [
            r"\bheight\b",
            r"\blength\b",
            r"\btube\s*length\b",
            r"高度",
            r"长度",
        ],
    },

    "performance": {
        "conductivity": [
            r"\bconductiv",
            r"\belectrical conductivity\b",
            r"\bspecific conductivity\b",
            r"\bconductance\b",
            r"\bcurrent density\b",  # 新增
            r"电导",
            r"导电",
        ],
        "resistivity": [
            r"\bresistiv",
            r"\belectrical resist",
            r"电阻率",
        ],
        "sheet_resistance": [
            r"\bsheet resistance\b",
            r"\bsheet\s*R\b",
            r"方阻",
        ],
        "tensile_strength": [
            r"\btensile\b",
            r"\bmechanical strength\b",
            r"\bultimate strength\b",
            r"\bstrength\b",
            r"抗拉",
            r"强度",
        ],
        "modulus": [
            r"\bmodulus\b",
            r"\byoung'?s modulus\b",
            r"\belastic modulus\b",
            r"\bstiffness\b",
            r"模量",
        ],
    },

    "mechanism": {
        # 大幅扩展（覆盖论文表3.4）
        "diffusion": [
            r"\bdiffusion\b",
            r"\bmass transport\b",  # 新增
            r"\bgas diffusion\b",
            r"扩散",
            r"质量传输",
        ],
        "catalyst_deactivation": [
            r"\bdeactivation\b",
            r"\bpoison\b",
            r"\binactivation\b",  # 新增
            r"失活",
            r"中毒",
        ],
        "catalyst_agglomeration": [
            r"\bripening\b",
            r"\bagglomer",
            r"\bsinter",
            r"\bcoarsening\b",  # 新增
            r"\bOstwald ripening\b",  # 新增
            r"烧结",
            r"团聚",
            r"粗化",
        ],
        "growth_kinetics": [
            r"\bkinetic",
            r"\bactivation energy\b",
            r"\bgrowth\s*rate\b",  # 新增
            r"\breaction\s*rate\b",  # 新增
            r"动力学",
            r"生长速率",
            r"反应速率",
        ],
        "boundary_layer_effect": [
            r"\bboundary layer\b",
            r"\bdiffusion layer\b",  # 新增
            r"边界层",
            r"扩散层",
        ],
        # 新增的机制实体
        "nucleation": [
            r"\bnucleation\b",
            r"\bnuclear\b",
            r"\bnucleation density\b",
            r"\bnucleation rate\b",
            r"成核",
            r"成核密度",
        ],
        "catalyst_activation": [
            r"\bcatalyst\s+activation\b",
            r"\bcatalyst\s+reduction\b",  # 新增
            r"\bcatalyst\s+state\b",
            r"催化剂激活",
            r"催化剂活化",
            r"催化剂还原",
        ],
        "catalyst_particle_size": [
            r"\bcatalyst\s+particle\s*size\b",
            r"\bnanoparticle\s*size\b",
            r"\bparticle\s+diameter\b",
            r"催化剂粒径",
            r"颗粒尺寸",
        ],
        "stress_induced_bending": [
            r"\bstress.*?bending\b",
            r"\bcurvature.*?stress\b",
            r"\bthermal\s+stress\b",
            r"\bresidual\s+stress\b",
            r"应力.*?弯曲",
            r"热应力",
            r"残余应力",
        ],
        "van_der_waals": [
            r"\bvan der waals\b",
            r"\bvdW\b",
            r"\bweak.*?interaction\b",
            r"范德华",
            r"弱相互作用",
        ],
        "tip_growth": [
            r"\btip\s+growth\b",
            r"\bbase\s+growth\b",
            r"顶端生长",
            r"基底生长",
        ],
        "carbon_feedstock": [
            r"\bcarbon\s+source\b",
            r"\bfeedstock\b",
            r"\bprecursor\b",
            r"碳源",
            r"前驱体",
        ],
    },
}


# ==================== 扩展的条件模式 ====================

ENHANCED_CONDITION_PATTERNS = {
    # 温度条件（改进）
    "temperature_above": [
        r"(?:温度|temperature)\s*[>高于]\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*above\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:higher|elevated)\s*temperature\s*[>than]\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
    ],
    "temperature_below": [
        r"(?:温度|temperature)\s*[<低于]\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*below\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:lower|reduced)\s*temperature\s*[<than]\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
    ],
    "temperature_range": [
        r"(?:温度|temperature)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*[°~\-to]\s*(?P<max>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*and\s*(?P<max>\d+\.?\d*)\s*°?[C℃]?",
        # "750-800°C" 格式
        r"(?P<min>\d{3,4})[-~](?P<max>\d{3,4})\s*°?[C℃]?",
    ],
    "temperature_equal": [
        r"(?:温度|temperature)\s*[=等于]\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?",
        r"(?:at|保持)\s*(?P<value>\d+\.?\d*)\s*°?[C℃]?\s*(?:温度|temperature)",
    ],

    # 时间条件（改进）
    "time_above": [
        r"(?:时间|time|duration)\s*[>超过]\s*(?P<value>\d+\.?\d*)\s*(?:min|分钟|h|小时)",
        r"(?:时间|time|duration)\s*above\s*(?P<value>\d+\.?\d*)\s*(?:min|h)",
    ],
    "time_range": [
        r"(?:时间|time|duration)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*[~\-to]\s*(?P<max>\d+\.?\d*)\s*(?:min|h|分钟|小时)",
        r"(?P<min>\d+)[-~](?P<max>\d+)\s*min",
    ],

    # 流量条件（改进）
    "flow_above": [
        r"(?:流量|flow)\s*[>超过]\s*(?P<value>\d+\.?\d*)\s*sccm",
        r"(?:流量|flow)\s*above\s*(?P<value>\d+\.?\d*)\s*sccm",
    ],
    "flow_range": [
        r"(?:流量|flow)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*[~\-to]\s*(?P<max>\d+\.?\d*)\s*sccm",
        r"(?P<min>\d+)[-~](?P<max>\d+)\s*sccm",
    ],

    # 厚度条件（改进）
    "thickness_above": [
        r"(?:厚度|thickness)\s*[>超过]\s*(?P<value>\d+\.?\d*)\s*nm",
        r"(?:厚度|thickness)\s*above\s*(?P<value>\d+\.?\d*)\s*nm",
    ],
    "thickness_range": [
        r"(?:厚度|thickness)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*[~\-to]\s*(?P<max>\d+\.?\d*)\s*nm",
        r"(?P<min>\d+\.?\d*)[-~](?P<max>\d+\.?\d*)\s*nm",
    ],

    # 压力条件
    "pressure_above": [
        r"(?:压力|pressure)\s*[>超过]\s*(?P<value>\d+\.?\d*)\s*Pa",
        r"(?:压力|pressure)\s*above\s*(?P<value>\d+\.?\d*)\s*Pa",
    ],
    "pressure_range": [
        r"(?:压力|pressure)\s*(?:在|between|from)\s*(?P<min>\d+\.?\d*)\s*[~\-to]\s*(?P<max>\d+\.?\d*)\s*Pa",
    ],
}


# ==================== 转换函数 ====================

def convert_to_msfu_format():
    """
    将增强模式转换为现有MSFU格式

    Returns:
        dict: 兼容knowledge_base.py的格式
    """
    # 关系模式
    output = {
        "PROCESS_FACTOR_PATTERNS": ENHANCED_ENTITY_PATTERNS["process"],
        "MORPHOLOGY_FACTOR_PATTERNS": ENHANCED_ENTITY_PATTERNS["morphology"],
        "PERFORMANCE_FACTOR_PATTERNS": ENHANCED_ENTITY_PATTERNS["performance"],
        "MECHANISM_FACTOR_PATTERNS": ENHANCED_ENTITY_PATTERNS["mechanism"],
        "INCREASE_PATTERNS": [p for config in ENHANCED_RELATION_PATTERNS
                               if config["category"] == "increase"
                               for p in config["patterns"]],
        "DECREASE_PATTERNS": [p for config in ENHANCED_RELATION_PATTERNS
                               if config["category"] == "decrease"
                               for p in config["patterns"]],
    }

    return output


if __name__ == "__main__":
    # 测试代码
    print("增强模式统计:")
    print(f"- 实体模式总数: {sum(len(patterns) for patterns in ENHANCED_ENTITY_PATTERNS.values())}")
    print(f"- 关系模式总数: {len(ENHANCED_RELATION_PATTERNS)}")
    print(f"- 条件模式总数: {len(ENHANCED_CONDITION_PATTERNS)}")
    print(f"- 机制实体: {len(ENHANCED_ENTITY_PATTERNS['mechanism'])}个")
