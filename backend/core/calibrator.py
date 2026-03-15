import os
import sys
from typing import Dict, Any, Optional

# 确保能导入 src.analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.analysis.furnace_model import FurnaceModel

class DataCalibrator:
    def __init__(self):
        self.model = FurnaceModel()
        
        # XR 均匀分布：C1=0, C10=41.2, 之间均匀分布
        # 对应用户确认的：C1-C10 在 0-41.2 cm 均匀分布
        total_len = 41.2
        num_membranes = 10
        self.xr_base_pos = {
            i: (i - 1) * (total_len / (num_membranes - 1)) 
            for i in range(1, num_membranes + 1)
        }
        
        # ZZY 标签到 cm 的映射 (通常取测量曲线的核心点)
        self.zzy_pos_map = {
            'mid': 29.4,
            'middle': 29.4,
            'top': 11.7,
            'bottom': 41.2
        }

    def calibrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据原始数据校准实际温度和物理位置
        """
        source = data.get('source')
        growth_temp = data.get('growth_temp')
        
        pos_cm = data.get('membrane_pos_cm')
        
        # 如果没有手动指定位置，则根据标签/ID自动计算
        if pos_cm is None:
            if source == 'XR':
                mid = data.get('membrane_id')
                if mid in self.xr_base_pos:
                    pos_cm = self.xr_base_pos[mid]
            elif source == 'ZZY':
                label = data.get('position_label', '').lower()
                for key, val in self.zzy_pos_map.items():
                    if key in label:
                        pos_cm = val
                        break
        
        if pos_cm is not None:
            data['membrane_pos_cm'] = pos_cm
            if growth_temp and data.get('actual_temp') is None:
                data['actual_temp'] = self.model.get_actual_temp(growth_temp, pos_cm)
                
        return data

# 全局单例
calibrator = DataCalibrator()
