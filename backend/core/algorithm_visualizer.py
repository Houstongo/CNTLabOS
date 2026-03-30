"""
算法可视化模块 — 传统阈值分割路径的薄壳包装。
实际算法逻辑已迁移到 visualization_adapter.py，
此类保持向后兼容接口。
"""
import numpy as np


class AlgorithmVisualizer:
    """算法可视化：委托给 FeatureVisualizationAdapter。"""

    def __init__(self, magnification=50000):
        self.mag = magnification
        self.steps = []
        self.reference_gray = None
        self.reference_bgr = None
        self.current_image = None
        self.current_step = 0
        self._adapter = None

    def visualize_extraction(self, img_gray):
        """生成完整的特征提取可视化流程（12 步）。"""
        from backend.core.visualization_adapter import FeatureVisualizationAdapter
        adapter = FeatureVisualizationAdapter(magnification=self.mag)
        self.steps = adapter.visualize(img_gray)
        self._adapter = adapter
        return self.steps

    def add_step(self, name, image, description=""):
        """兼容旧接口：添加步骤到列表。"""
        from backend.core.viz_rendering import encode_step
        step = encode_step(image, name, description)
        self.steps.append(step)

    def get_steps(self):
        return self.steps

    def get_step(self, index):
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
