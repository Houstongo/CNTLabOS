"""测试 LLM MSFU 精炼"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.ai_interpreter import AIInterpreter
import json

# 测试 API
api_key = "37g04jzR931b4ae8b8e8a04da8127e66.nixWPPYNKwW4FHHM"
interpreter = AIInterpreter(provider="glm", api_key=api_key)

# 模拟候选 MSFU
candidates = [{
    "content": "Increasing the growth temperature leads to higher carbon nanotube density.",
    "assertion": {
        "source_entity": "process:temperature",
        "relation_type": "increases",
        "target_entity": "morphology:density",
        "direction": "positive"
    }
}]

context = "The growth temperature was varied from 600 to 800°C. Increasing the growth temperature leads to higher carbon nanotube density due to enhanced catalyst activity."

print("测试 LLM 精炼...")
try:
    result = interpreter.refine_msfu_batch(candidates, context)
    print("返回类型:", type(result))
    print("返回内容:", json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    print("错误:", e)
