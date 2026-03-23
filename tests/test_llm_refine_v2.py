"""测试 LLM 精炼功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.msfu_extractor import MSFUExtractor, MSFUMetadata
from backend.core.ai_interpreter import AIInterpreter

# 初始化 LLM client
api_key = "37g04jzR931b4ae8b8e8a04da8127e66.nixWPPYNKwW4FHHM"
llm_client = AIInterpreter(provider="glm", api_key=api_key)

# 创建提取器
extractor = MSFUExtractor(llm_client=llm_client, use_llm_refinement=True)

# 测试文本
test_text = """
The growth temperature was varied from 600 to 800°C.
Increasing the temperature leads to higher carbon nanotube density due to enhanced catalyst activity.
At higher temperatures, the catalyst particles become more active, promoting faster CNT growth.
However, excessive temperature can decrease the alignment quality of the CNT arrays.
"""

metadata = MSFUMetadata(
    doc_id="test_doc",
    chunk_id="test_chunk",
    doc_title="Test Document",
    doc_type="pdf"
)

# 提取 MSFU
print("提取 MSFU...")
msfus = extractor.extract(test_text, metadata, "Test Document")

print(f"提取到 {len(msfus)} 个 MSFU")

for i, msfu in enumerate(msfus, 1):
    print(f"\n[{i}] {msfu.assertion.source_entity} --[{msfu.assertion.relation_type}]--> {msfu.assertion.target_entity}")
    print(f"    direction: {msfu.assertion.direction}")
    print(f"    confidence: {msfu.evidence.confidence:.2f}")
    print(f"    method: {msfu.evidence.extraction_method}")
    print(f"    content: {msfu.content[:100]}...")
