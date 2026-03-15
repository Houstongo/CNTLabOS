import numpy as np
from scipy.interpolate import interp1d

class FurnaceModel:
    """
    根据《管式炉温区.xlsx》建立的温度预测模型
    """
    def __init__(self):
        # 预存 Excel 抽提的数据
        self.data = {
            750: {
                'pos': [0, 7, 14, 20, 28, 35, 42],
                'temp': [665, 715, 758, 775, 789, 775, 734]
            },
            800: {
                'pos': [0, 5.886, 11.771, 17.657, 23.543, 29.429, 35.314, 41.2],
                'temp': [674, 746, 790, 822, 842, 845, 831, 786]
            },
            850: {
                'pos': [7, 14, 21, 28, 35, 42],
                'temp': [795, 851, 878, 880, 880, 855]
            }
        }
        self.f_models = {}
        for set_temp, vals in self.data.items():
            self.f_models[set_temp] = interp1d(vals['pos'], vals['temp'], kind='cubic', fill_value="extrapolate")

    def get_actual_temp(self, set_temp, position_cm):
        """
        输入设定温度(℃)和距离(cm)，返回预测的实际局部温度
        """
        # 寻找最接近的设定温度模型，或者进行线性插值
        temps = sorted(self.f_models.keys())
        if set_temp in self.f_models:
            return float(self.f_models[set_temp](position_cm))
        
        # 如果设定温度介于两者之间，进行双线性插值
        if set_temp < temps[0]: return float(self.f_models[temps[0]](position_cm))
        if set_temp > temps[-1]: return float(self.f_models[temps[-1]](position_cm))
        
        # 线性插值设定温度
        for i in range(len(temps)-1):
            t1, t2 = temps[i], temps[i+1]
            if t1 < set_temp < t2:
                v1 = self.f_models[t1](position_cm)
                v2 = self.f_models[t2](position_cm)
                return float(v1 + (v2 - v1) * (set_temp - t1) / (t2 - t1))
        return set_temp

if __name__ == "__main__":
    model = FurnaceModel()
    test_pos = 20
    test_set = 800
    print(f"设定温度 {test_set}℃, 位置 {test_pos}cm -> 实际温度: {model.get_actual_temp(test_set, test_pos):.2f}℃")
