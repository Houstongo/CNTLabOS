"""
测试知识驱动预测模型
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.knowledge_driven_predictor import KnowledgeDrivenPredictor
from backend.core.knowledge_rag import RAGRetriever


def test_basic_prediction():
    """测试基础预测功能"""
    print("=" * 60)
    print("测试1：基础预测功能")
    print("=" * 60)

    # 初始化组件
    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

    rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
    predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)

    # 测试ZZY数据预测
    print("\n[ZZY数据预测]")
    zzy_params = {
        'source': 'ZZY',
        'growth_temp': 750,
        'growth_time': 3,
        'fe_thickness': 1.5,
        'al2o3_thickness': 10,
        'ar_flow': 500,
        'h2_flow': 100,
        'c2h4_flow': 50,
    }

    for target in ['diameter', 'density', 'alignment', 'curvature']:
        result = predictor.predict(zzy_params, target=target, use_knowledge=True)
        print(f"{target:12s}: {result.predicted_value:.2f} (置信度: {result.confidence:.2f})")
        print(f"  知识基线: {result.knowledge_baseline:.2f}, ML残差: {result.ml_residual:.2f}")

    # 测试XR数据预测
    print("\n[XR数据预测]")
    xr_params = {
        'source': 'XR',
        'actual_temp': 750,
        'growth_time': 3,
        'membrane_pos_cm': 18.0,
    }

    for target in ['diameter', 'density', 'alignment', 'curvature']:
        result = predictor.predict(xr_params, target=target, use_knowledge=True)
        print(f"{target:12s}: {result.predicted_value:.2f} (置信度: {result.confidence:.2f})")
        print(f"  知识基线: {result.knowledge_baseline:.2f}, ML残差: {result.ml_residual:.2f}")


def test_physical_constraints():
    """测试物理约束检查"""
    print("\n" + "=" * 60)
    print("测试2：物理约束检查")
    print("=" * 60)

    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

    rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
    predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)

    # 测试违反约束的参数
    print("\n[违反约束的参数]")
    invalid_params = {
        'source': 'ZZY',
        'growth_temp': 500,  # 温度过低
        'fe_thickness': 0.3,  # Fe厚度过薄
        'ar_flow': 100,  # Ar流量过低
        'c2h4_flow': 100,  # C2H4流量过高
    }

    result = predictor.predict(invalid_params, target='density', use_knowledge=True)
    print(f"预测密度: {result.predicted_value:.2f}%")
    print(f"置信度: {result.confidence:.2f}")
    print(f"物理约束: {result.physical_constraints}")


def test_knowledge_enhanced_features():
    """测试知识增强特征"""
    print("\n" + "=" * 60)
    print("测试3：知识增强特征工程")
    print("=" * 60)

    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

    rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
    predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)

    params = {
        'source': 'ZZY',
        'growth_temp': 750,
        'growth_time': 3,
        'fe_thickness': 1.5,
        'al2o3_thickness': 10,
        'ar_flow': 500,
        'h2_flow': 100,
        'c2h4_flow': 50,
    }

    features = predictor.build_knowledge_enhanced_features(params)
    print("\n知识增强特征:")
    for name, value in features.items():
        print(f"  {name:25s}: {value:.4f}")


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 60)
    print("测试4：批量预测")
    print("=" * 60)

    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

    rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
    predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)

    # 批量预测不同温度下的直径
    params_list = [
        {'source': 'ZZY', 'growth_temp': 700, 'growth_time': 3, 'fe_thickness': 1.5},
        {'source': 'ZZY', 'growth_temp': 750, 'growth_time': 3, 'fe_thickness': 1.5},
        {'source': 'ZZY', 'growth_temp': 800, 'growth_time': 3, 'fe_thickness': 1.5},
    ]

    results = predictor.batch_predict(params_list, target='diameter')

    print("\n温度对直径的影响:")
    for i, result in enumerate(results):
        temp = params_list[i]['growth_temp']
        print(f"  {temp}℃: {result.predicted_value:.2f}nm (置信度: {result.confidence:.2f})")


def test_rag_integration():
    """测试RAG集成"""
    print("\n" + "=" * 60)
    print("测试5：RAG文献检索集成")
    print("=" * 60)

    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

    rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
    predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)

    # 检查文献库状态
    documents = rag_retriever.list_documents()
    print(f"\nRAG文献库状态: {len(documents)} 篇文献")

    if documents:
        print("\n文献列表:")
        for doc in documents[:3]:
            # 兼容不同的键名
            filename = doc.get('filename', 'unknown')
            chunk_count = doc.get('chunk_count', 0)
            print(f"  ID={doc['id']}: {filename} ({chunk_count} chunks)")

    # 使用RAG检索进行预测
    params = {
        'source': 'ZZY',
        'growth_temp': 750,
        'growth_time': 3,
        'fe_thickness': 1.5,
        'ar_flow': 500,
        'h2_flow': 100,
        'c2h4_flow': 50,
    }

    query = "temperature effect on CNT diameter"
    result = predictor.predict(params, target='diameter', query=query)

    print(f"\n使用RAG检索的预测:")
    print(f"  预测直径: {result.predicted_value:.2f}nm")
    print(f"  RAG证据: {len(result.rag_evidence)} 条")
    for i, evidence in enumerate(result.rag_evidence, 1):
        print(f"    [{i}] {evidence.get('filename', 'unknown')}: {evidence.get('text', '')[:100]}...")


if __name__ == '__main__':
    try:
        test_basic_prediction()
        test_physical_constraints()
        test_knowledge_enhanced_features()
        test_batch_prediction()
        test_rag_integration()

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()