"""测试规则提取"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.msfu_extractor import RuleBasedExtractor, MSFUMetadata

# 测试文本
test_text = """
The growth temperature was increased from 600 to 800 C.
Increasing the temperature leads to higher carbon nanotube density.
At higher temperatures, the catalyst particles become more active.
The CNT diameter decreases when the flow rate is reduced.
"""

metadata = MSFUMetadata(
    doc_id="test",
    chunk_id="test",
    doc_title="Test Document",
    doc_type="pdf"
)

# 创建提取器
extractor = RuleBasedExtractor()

# 提取
msfus = extractor.extract(test_text, metadata, "Test")

print(f"提取到 {len(msfus)} 个 MSFU")
for i, msfu in enumerate(msfus, 1):
    print(f"[{i}] {msfu.assertion.source_entity} --[{msfu.assertion.relation_type}]--> {msfu.assertion.target_entity}")
    print(f"    direction: {msfu.assertion.direction}, confidence: {msfu.evidence.confidence:.2f}")
