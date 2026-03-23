"""
简化版增强提取器测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.rule_based_extractor_enhanced import EnhancedRuleExtractor
from backend.core.msfu_extractor import MSFUMetadata


# 简化的测试句子
TEST_SENTENCES = [
    ("Simple", "Increasing growth temperature improves alignment."),
    ("Nominalization", "The increase of growth temperature enhances alignment."),
    ("Composite", "Increased temperature enhances alignment and reduces diameter."),
    ("Passive", "Alignment is improved by higher temperature."),
    ("Negation", "Temperature does not significantly affect density."),
    ("Conditional", "When temperature > 750°C, conductivity improves."),
    ("Chinese", "生长温度的提高改善了取向度。"),
    ("Mechanism", "Ostwald ripening leads to catalyst coarsening."),
]


def test_enhanced_extractor():
    """测试增强提取器"""
    print("="*60)
    print("Enhanced Rule Extractor Test")
    print("="*60)

    enhanced = EnhancedRuleExtractor(confidence_threshold=0.3)

    metadata = MSFUMetadata(
        doc_id="test",
        chunk_id="0",
        doc_title="Test Document"
    )

    total_msfus = 0
    coverage = 0

    for i, (test_type, sentence) in enumerate(TEST_SENTENCES):
        print(f"\n[{test_type}] Sentence {i+1}:")
        print(f"  Text: {sentence[:60]}...")

        msfus = enhanced.extract(sentence, metadata)
        count = len(msfus)
        total_msfus += count
        if count > 0:
            coverage += 1

        print(f"  Extracted: {count} MSFU(s)")

        if msfus:
            for msfu in msfus[:2]:  # Show first 2
                print(f"    - {msfu.assertion.source_entity} -> {msfu.assertion.target_entity}")
                print(f"      Type: {msfu.assertion.relation_type}, Direction: {msfu.assertion.direction}, Confidence: {msfu.evidence.confidence:.2f}")

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total sentences: {len(TEST_SENTENCES)}")
    print(f"Coverage: {coverage}/{len(TEST_SENTENCES)} ({coverage/len(TEST_SENTENCES)*100:.1f}%)")
    print(f"Total MSFUs: {total_msfus}")
    print(f"Average per sentence: {total_msfus/len(TEST_SENTENCES):.2f}")
    print("="*60)

    # Test specific patterns
    print("\nPattern Coverage Test:")
    test_patterns = [
        "Nominalization (increase of...)",
        "Passive voice (is improved by...)",
        "Composite predicates (A affects B and C)",
        "Negation (does not affect)",
        "Mechanism entities (Ostwald ripening)",
    ]
    print("Enhanced patterns should cover:")
    for pattern in test_patterns:
        print(f"  - {pattern}")
    print("="*60)


if __name__ == "__main__":
    test_enhanced_extractor()
