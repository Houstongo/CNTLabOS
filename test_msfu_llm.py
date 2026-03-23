"""测试 MSFU LLM 精炼功能"""
import sys
sys.path.insert(0, 'CNTA_ML_Project')

from backend.core.msfu_extractor import MSFUExtractor
from backend.core.ai_interpreter import AIInterpreter

print('模块导入成功')
print('MSFUExtractor._refine_with_llm 方法存在:', hasattr(MSFUExtractor, '_refine_with_llm'))
print('AIInterpreter.refine_msfu_batch 方法存在:', hasattr(AIInterpreter, 'refine_msfu_batch'))
