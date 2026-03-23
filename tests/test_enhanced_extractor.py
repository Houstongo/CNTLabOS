"""
增强规则提取器测试脚本
测试新模式的覆盖率和准确率
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.rule_based_extractor_enhanced import EnhancedRuleExtractor
from backend.core.msfu_extractor import MSFUMetadata


# 测试句子（覆盖各种复杂句式）
TEST_SENTENCES = [
    # 1. 原有简单句式（应该都能提取）
    "Increasing growth temperature improves alignment.",
    "Higher density reduces conductivity.",
    "Fe thickness increases diameter.",
    "催化剂厚度增加导致直径增大。",

    # 2. 名词化结构（增强提取器应该能提取）
    "The increase of growth temperature enhances alignment.",
    "Decrease in catalyst thickness reduces diameter.",
    "Temperature elevation leads to improved conductivity.",
    "生长温度的提高改善了取向度。",

    # 3. 复合谓词（增强提取器应该能提取）
    "Increased temperature enhances alignment and reduces diameter.",
    "Higher Fe thickness improves density but decreases alignment.",
    "温度增加提高了密度但降低了取向度。",

    # 4. 被动语态（增强提取器应该能提取）
    "Alignment is improved by higher temperature.",
    "Density is increased by longer growth time.",
    "取向度被较高的温度改善。",

    # 5. 否定句式（增强提取器应该能识别）
    "Temperature does not significantly affect density.",
    "Fe thickness has no effect on alignment.",
    "温度对密度没有明显影响。",

    # 6. 条件句式（增强提取器应该能提取条件）
    "When temperature > 750°C, conductivity improves.",
    "Temperature in the range of 750-800°C enhances alignment.",
    "在温度高于750°C时，导电性改善。",

    # 7. 比较句式（增强提取器应该能提取）
    "Higher temperature leads to better alignment.",
    "Lower pressure results in decreased density.",
    "更高的温度导致更好的取向度。",

    # 8. 机制实体（增强提取器应该能识别更多）
    "Ostwald ripening leads to catalyst coarsening.",
    "Catalyst deactivation affects growth rate.",
    "Nucleation density determines tube diameter.",
    "Stress-induced bending increases curvature.",
    "Diffusion-limited growth reduces alignment.",

    # 9. 复杂条件（增强提取器应该能提取范围条件）
    "Temperature range 700-750°C improves density.",
    "Growth time 3-5 minutes increases diameter.",
    "温度在700-750°C范围内提高密度。",
]


def test_coverage():
    """测试覆盖率"""
    print("="*70)
    print("测试覆盖率")
    print("="*70)

    # 创建提取器
    try:
        original = OriginalExtractor(confidence_threshold=0.3)
        use_original = True
    except Exception as e:
        print(f"警告：无法初始化原有提取器: {e}")
        print("将只测试增强提取器")
        use_original = False

    enhanced = EnhancedRuleExtractor(confidence_threshold=0.3)

    metadata = MSFUMetadata(
        doc_id="test",
        chunk_id="0",
        doc_title="Test Document"
    )

    original_total = 0
    enhanced_total = 0
    original_coverage = []
    enhanced_coverage = []

    for i, sentence in enumerate(TEST_SENTENCES):
        # 原有提取器
        original_count = 0
        original_msfus = []
        if use_original:
            try:
                original_msfus = original.extract(sentence, metadata)
                original_count = len(original_msfus)
                original_total += original_count
            except Exception as e:
                print(f"  警告：原有提取器处理失败: {e}")

        # 增强提取器
        enhanced_msfus = enhanced.extract(sentence, metadata)
        enhanced_count = len(enhanced_msfus)
        enhanced_total += enhanced_count

        # 记录覆盖率
        if use_original:
            original_coverage.append(original_count > 0)
        enhanced_coverage.append(enhanced_count > 0)

        # 显示结果
        status = "+" if enhanced_count > original_count else "="
        print(f"\n[{status}] 句子 {i+1}:")
        print(f"  {sentence[:70]}...")
        if use_original:
            print(f"  原有提取器: {original_count} 个MSFU")
        print(f"  增强提取器: {enhanced_count} 个MSFU")
        if use_original and enhanced_count > original_count:
            print(f"  改进: +{enhanced_count - original_count} 个")

        # 显示提取的关系
        if enhanced_msfus:
            print(f"  提取的关系:")
            for msfu in enhanced_msfus[:2]:  # 只显示前2个
                print(f"    - {msfu.assertion.source_entity} → {msfu.assertion.target_entity}")
                print(f"      类型: {msfu.assertion.relation_type}, 方向: {msfu.assertion.direction}, 置信度: {msfu.evidence.confidence:.2f}")

    # 统计
    print("\n" + "="*70)
    print("统计结果:")
    print("="*70)
    print(f"总句子数: {len(TEST_SENTENCES)}")
    print(f"原有提取器:")
    print(f"  - 总提取: {original_total} 个MSFU")
    print(f"  - 覆盖率: {sum(original_coverage)}/{len(TEST_SENTENCES)} ({sum(original_coverage)/len(TEST_SENTENCES)*100:.1f}%)")
    print(f"  - 平均每句: {original_total/len(TEST_SENTENCES):.2f} 个MSFU")
    print(f"\n增强提取器:")
    print(f"  - 总提取: {enhanced_total} 个MSFU")
    print(f"  - 覆盖率: {sum(enhanced_coverage)}/{len(TEST_SENTENCES)} ({sum(enhanced_coverage)/len(TEST_SENTENCES)*100:.1f}%)")
    print(f"  - 平均每句: {enhanced_total/len(TEST_SENTENCES):.2f} 个MSFU")
    print(f"\n改进:")
    print(f"  - 增加数量: +{enhanced_total - original_total} 个MSFU")
    print(f"  - 提升率: {(enhanced_total - original_total)/max(original_total, 1)*100:.1f}%")
    print(f"  - 覆盖率提升: {sum(enhanced_coverage) - sum(original_coverage)} 句")


def test_entity_recognition():
    """测试实体识别"""
    print("\n" + "="*70)
    print("测试实体识别")
    print("="*70)

    enhanced = EnhancedRuleExtractor(confidence_threshold=0.3)

    test_entities = [
        "growth temperature",
        "annealing temperature",
        "catalyst thickness",
        "alignment",
        "conductivity",
        "Ostwald ripening",
        "catalyst deactivation",
        "nucleation density",
    ]

    print("\n实体识别测试:")
    for entity_text in test_entities:
        normalized = enhanced._normalize_entity(entity_text)
        category = normalized.split(":")[0] if ":" in normalized else "unknown"
        entity_type = normalized.split(":")[1] if ":" in normalized else normalized
        print(f"  '{entity_text}' → {normalized} ({category}/{entity_type})")


def test_condition_extraction():
    """测试条件提取"""
    print("\n" + "="*70)
    print("测试条件提取")
    print("="*70)

    enhanced = EnhancedRuleExtractor(confidence_threshold=0.3)

    test_conditions = [
        "Temperature > 750°C improves alignment.",
        "Temperature range 700-750°C increases density.",
        "Growth time > 5 minutes reduces diameter.",
        "Pressure below 100 Pa affects growth.",
    ]

    print("\n条件提取测试:")
    for sentence in test_conditions:
        conditions = enhanced._extract_conditions(sentence)
        print(f"\n  句子: {sentence}")
        if conditions:
            for cond in conditions:
                print(f"    提取条件: {cond.parameter} {cond.operator} {cond.value} {cond.unit or ''}")
        else:
            print(f"    未提取到条件")


def run_all_tests():
    """运行所有测试"""
    test_coverage()
    test_entity_recognition()
    test_condition_extraction()

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
    print("\n建议:")
    print("1. 查看覆盖率提升情况")
    print("2. 检查实体识别准确性")
    print("3. 验证条件提取正确性")
    print("4. 如有误判，调整模式权重和阈值")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
