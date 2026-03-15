import os
import re
import pandas as pd

def parse_folder_name(folder_name):
    """
    解析文件夹名称中的参数。
    格式示例: 250403 T800 3h L250 1.5g
    """
    params = {
        'folder_name': folder_name,
        'date': None,
        'temperature_C': None,
        'time_h': None,
        'flow_L': None,
        'weight_g': None
    }
    
    # 日期 (前面6位数字)
    date_match = re.search(r'^(\d{6})', folder_name)
    if date_match:
        params['date'] = date_match.group(1)
        
    # 温度 (Txxx)
    temp_match = re.search(r'[tT](\d+)', folder_name)
    if temp_match:
        params['temperature_C'] = int(temp_match.group(1))
        
    # 时间 (xh)
    time_match = re.search(r'(\d+)h', folder_name, re.IGNORECASE)
    if time_match:
        params['time_h'] = int(time_match.group(1))
        
    # 流量 (Lxxx)
    flow_match = re.search(r'[lL](\d+)', folder_name)
    if flow_match:
        params['flow_L'] = int(flow_match.group(1))
        
    # 质量 (x.xg)
    weight_match = re.search(r'(\d+\.?\d*)g', folder_name)
    if weight_match:
        params['weight_g'] = float(weight_match.group(1))
        
    return params

def main():
    root_dir = r'd:\CNTDATA'
    all_data = []
    
    # 遍历主目录
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item != 'CNTA_ML_Project':
            params = parse_folder_name(item)
            
            # 统计文件夹内的图像数量
            tiff_files = [f for f in os.listdir(item_path) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
            params['image_count'] = len(tiff_files)
            
            all_data.append(params)
            
    # 转为 DataFrame 并保存
    df = pd.DataFrame(all_data)
    output_path = r'd:\CNTDATA\CNTA_ML_Project\data\dataset_index.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"成功解析 {len(all_data)} 个实验批次，索引已保存至: {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()
