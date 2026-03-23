"""
知识驱动预测模型
结合RAG文献、专家知识和机器学习，进行CNT形貌特征预测
"""
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


@dataclass
class PredictionResult:
    """预测结果"""
    predicted_value: float
    confidence: float
    knowledge_baseline: float
    ml_residual: float
    similar_experiments: List[Dict]
    rag_evidence: List[Dict]
    physical_constraints: List[str]


class KnowledgeDrivenPredictor:
    """知识驱动预测器"""

    def __init__(self, db_path: str, rag_retriever):
        self.db_path = db_path
        self.rag = rag_retriever
        self.models = {}
        self.scalers = {}
        self.init_models()

    def init_models(self):
        """初始化ML模型"""
        targets = ['diameter', 'density', 'alignment', 'curvature']
        for target in targets:
            self.models[target] = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            self.scalers[target] = StandardScaler()

    # ------------------------------------------------------------------ #
    #  知识层：物理约束规则
    # ------------------------------------------------------------------ #
    def check_physical_constraints(self, params: Dict) -> List[str]:
        """
        检查物理约束，返回违反的约束列表
        基于CNT生长机理的专家知识
        """
        constraints = []

        # 温度约束
        temp = params.get('growth_temp') or params.get('actual_temp')
        if temp:
            if temp < 600:
                constraints.append("温度过低(<600℃)，碳源裂解不足，形核困难")
            elif temp > 900:
                constraints.append("温度过高(>900℃)，催化剂易团聚，密度下降")

        # Fe厚度约束
        fe = params.get('fe_thickness')
        if fe:
            if fe < 0.5:
                constraints.append("Fe厚度过薄(<0.5nm)，催化活性低，生长困难")
            elif fe > 3.0:
                constraints.append("Fe厚度过厚(>3.0nm)，颗粒团聚，密度降低")

        # 气流约束
        ar = params.get('ar_flow')
        h2 = params.get('h2_flow')
        c2h4 = params.get('c2h4_flow')
        if ar and c2h4:
            ratio = ar / c2h4 if c2h4 > 0 else float('inf')
            if ratio < 5:
                constraints.append(f"Ar/C2H4比过低({ratio:.1f}<5)，碳浓度过高易致无定形碳")
            elif ratio > 50:
                constraints.append(f"Ar/C2H4比过高({ratio:.1f}>50)，碳浓度不足密度降低")

        return constraints

    # ------------------------------------------------------------------ #
    #  数据层：知识增强特征工程
    # ------------------------------------------------------------------ #
    def build_knowledge_enhanced_features(self, params: Dict) -> Dict:
        """
        基于专家知识构造复合特征
        """
        features = {}

        # 基础参数
        temp = params.get('growth_temp') or params.get('actual_temp', 750)
        time = params.get('growth_time', 3)
        fe = params.get('fe_thickness', 1.0)
        al2o3 = params.get('al2o3_thickness', 10)
        ar = params.get('ar_flow', 500)
        h2 = params.get('h2_flow', 100)
        c2h4 = params.get('c2h4_flow', 50)

        # 专家知识驱动的复合特征
        features['temp_normalized'] = temp / 750.0  # 归一化温度
        features['fe_thickness_sq'] = fe ** 2  # 厚度平方（影响团聚）

        # 催化剂效率：Fe/Al2O3比
        features['catalyst_ratio'] = fe / al2o3 if al2o3 > 0 else 0

        # 温度稳定性：温度/时间
        features['temp_stability'] = temp / time if time > 0 else 0

        # 碳供应能力：C2H4/Ar比
        features['carbon_supply'] = c2h4 / ar if ar > 0 else 0

        # 还原氛围：H2/Ar比
        features['reduction_ratio'] = h2 / ar if ar > 0 else 0

        # 温度梯度（XR特有）
        pos = params.get('membrane_pos_cm')
        if pos:
            features['position_normalized'] = pos / 36.0

        return features

    # ------------------------------------------------------------------ #
    #  模型层：知识基线 + ML残差混合预测
    # ------------------------------------------------------------------ #
    def predict(
        self,
        params: Dict,
        target: str = 'diameter',
        query: Optional[str] = None,
        use_knowledge: bool = True
    ) -> PredictionResult:
        """
        混合预测：知识基线 + ML残差
        """
        # 1. 物理约束检查
        constraints = self.check_physical_constraints(params)
        if constraints:
            # 严重违反物理约束，返回保守估计
            return self._get_conservative_estimate(params, target, constraints)

        # 2. RAG证据检索
        rag_evidence = []
        if query:
            rag_evidence = self.rag.retrieve_from_pdf(query, top_k=2)

        # 3. 相似实验检索（知识基线）
        similar_exps = self.rag.retrieve_from_db({}, params, top_k=5)

        # 4. 计算知识基线预测
        knowledge_baseline = self._compute_knowledge_baseline(
            similar_exps, target, params
        )

        # 5. ML残差预测
        ml_residual = 0.0
        if use_knowledge and similar_exps:
            ml_residual = self._predict_ml_residual(params, target, similar_exps)

        # 6. 混合预测
        predicted_value = knowledge_baseline + ml_residual

        # 7. 计算置信度
        confidence = self._compute_confidence(
            similar_exps, rag_evidence, params, target
        )

        return PredictionResult(
            predicted_value=predicted_value,
            confidence=confidence,
            knowledge_baseline=knowledge_baseline,
            ml_residual=ml_residual,
            similar_experiments=similar_exps,
            rag_evidence=rag_evidence,
            physical_constraints=[]
        )

    def _compute_knowledge_baseline(
        self,
        similar_exps: List[Dict],
        target: str,
        params: Dict
    ) -> float:
        """
        基于相似实验的加权平均（知识基线）
        权重由参数相似度决定
        """
        if not similar_exps:
            return self._get_default_baseline(params, target)

        # 计算每条相似实验的权重
        weights = []
        values = []
        for exp in similar_exps:
            weight = self._compute_similarity_weight(exp, params)
            value = exp.get(target)
            if value is not None:
                weights.append(weight)
                values.append(value)

        if not values:
            return self._get_default_baseline(params, target)

        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # 加权平均
        baseline = sum(v * w for v, w in zip(values, normalized_weights))
        return baseline

    def _compute_similarity_weight(self, exp: Dict, params: Dict) -> float:
        """计算相似度权重"""
        weight = 1.0

        # 温度权重
        temp_param = params.get('growth_temp') or params.get('actual_temp')
        temp_exp = exp.get('growth_temp') or exp.get('actual_temp')
        if temp_param and temp_exp:
            weight *= np.exp(-abs(temp_param - temp_exp) / 50.0)

        # Fe厚度权重
        fe_param = params.get('fe_thickness')
        fe_exp = exp.get('fe_thickness')
        if fe_param and fe_exp:
            weight *= np.exp(-abs(fe_param - fe_exp) / 0.5)

        # 时间权重
        time_param = params.get('growth_time')
        time_exp = exp.get('growth_time')
        if time_param and time_exp:
            weight *= np.exp(-abs(time_param - time_exp) / 1.0)

        return max(0.1, weight)

    def _predict_ml_residual(
        self,
        params: Dict,
        target: str,
        similar_exps: List[Dict]
    ) -> float:
        """
        预测ML残差
        使用知识增强的特征
        """
        # 构造特征
        features = self.build_knowledge_enhanced_features(params)

        # 这里简化处理，实际应该训练模型
        # 基于专家知识的简单规则预测残差
        temp = params.get('growth_temp') or params.get('actual_temp', 750)
        fe = params.get('fe_thickness', 1.0)

        if target == 'diameter':
            # 高温+厚催化剂 → 大直径
            residual = (temp - 750) * 0.02 + (fe - 1.0) * 3.0
        elif target == 'density':
            # 适中Fe厚度 → 高密度
            optimal_fe = 1.5
            residual = -abs(fe - optimal_fe) * 5.0
        elif target == 'alignment':
            # 高温 → 好取向
            residual = (temp - 750) * 0.001
        else:  # curvature
            # 适中温度 → 低曲率
            residual = -abs(temp - 780) * 0.0001

        return residual

    def _compute_confidence(
        self,
        similar_exps: List[Dict],
        rag_evidence: List[Dict],
        params: Dict,
        target: str
    ) -> float:
        """计算预测置信度"""
        confidence = 0.5  # 基础置信度

        # 相似实验数量
        n_similar = len([e for e in similar_exps if e.get(target) is not None])
        confidence += min(n_similar * 0.1, 0.3)

        # RAG证据数量
        n_evidence = len(rag_evidence)
        confidence += min(n_evidence * 0.05, 0.1)

        # 参数是否在训练范围内
        temp = params.get('growth_temp') or params.get('actual_temp')
        if temp and 600 <= temp <= 900:
            confidence += 0.05

        return min(confidence, 1.0)

    def _get_default_baseline(self, params: Dict, target: str) -> float:
        """获取默认基线值"""
        baselines = {
            'diameter': 30.0,  # nm
            'density': 50.0,   # %
            'alignment': 0.6,  # 无量纲
            'curvature': 1.1,  # 无量纲
        }
        return baselines.get(target, 0.0)

    def _get_conservative_estimate(
        self,
        params: Dict,
        target: str,
        constraints: List[str]
    ) -> PredictionResult:
        """违反物理约束时的保守估计"""
        baseline = self._get_default_baseline(params, target)

        # 根据约束类型调整
        constraint_text = " ".join(constraints)
        if "温度过低" in constraint_text:
            if target == 'density':
                baseline *= 0.5
            elif target == 'diameter':
                baseline *= 1.5
        elif "温度过高" in constraint_text:
            if target == 'density':
                baseline *= 0.7
            elif target == 'diameter':
                baseline *= 1.3

        return PredictionResult(
            predicted_value=baseline,
            confidence=0.3,  # 低置信度
            knowledge_baseline=baseline,
            ml_residual=0.0,
            similar_experiments=[],
            rag_evidence=[],
            physical_constraints=constraints
        )

    # ------------------------------------------------------------------ #
    #  模型训练（可选）
    # ------------------------------------------------------------------ #
    def train_models(self, source: Optional[str] = None):
        """
        训练ML模型
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # 查询训练数据
        query = """
            SELECT * FROM images
            WHERE processed = 1
        """
        if source:
            query += f" AND source = '{source}'"

        rows = conn.execute(query).fetchall()
        conn.close()

        if len(rows) < 50:
            print(f"数据量不足({len(rows)}条)，跳过训练")
            return

        # 准备训练数据
        for target in ['diameter', 'density', 'alignment', 'curvature']:
            X = []
            y = []

            for row in rows:
                # 构造特征
                params = dict(row)
                features = self.build_knowledge_enhanced_features(params)

                # 构造特征向量
                feature_vector = [
                    features.get('temp_normalized', 0),
                    features.get('fe_thickness_sq', 0),
                    features.get('catalyst_ratio', 0),
                    features.get('temp_stability', 0),
                    features.get('carbon_supply', 0),
                    features.get('reduction_ratio', 0),
                ]

                if row[target] is not None:
                    X.append(feature_vector)
                    y.append(row[target])

            if len(X) < 20:
                continue

            X = np.array(X)
            y = np.array(y)

            # 训练模型
            X_scaled = self.scalers[target].fit_transform(X)

            self.models[target]['rf'].fit(X_scaled, y)
            self.models[target]['gb'].fit(X_scaled, y)

            # 交叉验证
            rf_score = np.mean(cross_val_score(
                self.models[target]['rf'], X_scaled, y, cv=5, scoring='r2'
            ))
            gb_score = np.mean(cross_val_score(
                self.models[target]['gb'], X_scaled, y, cv=5, scoring='r2'
            ))

            print(f"{target}: RF R2={rf_score:.3f}, GB R2={gb_score:.3f}")

    # ------------------------------------------------------------------ #
    #  批量预测
    # ------------------------------------------------------------------ #
    def batch_predict(
        self,
        params_list: List[Dict],
        target: str = 'diameter',
        query: Optional[str] = None
    ) -> List[PredictionResult]:
        """批量预测"""
        return [
            self.predict(params, target, query)
            for params in params_list
        ]
