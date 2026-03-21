import os
import sys
# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_manager import CNTADataParser, DatabaseManager
from calibrator import calibrator

def convert_time_to_h(time_str):
    """将 15min 或 3h 统一转换为小时(float)"""
    if 'min' in time_str:
        return float(time_str.replace('min', '')) / 60.0
    if 'h' in time_str:
        return float(time_str.replace('h', ''))
    try:
        return float(time_str)
    except:
        return 0.0

def populate(clear=False):
    root_data_dirs = {
        'XR': r'd:\CNTDATA\XR',
        'ZZY': r'd:\CNTDATA\ZZY'
    }
    db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
    
    parser = CNTADataParser()
    db = DatabaseManager(db_path)
    
    if clear:
        db.clear_tables()
    
    for source, root_dir in root_data_dirs.items():
        if not os.path.exists(root_dir):
            print(f"Directory not found: {root_dir}")
            continue
            
        print(f"Scanning {source} data in {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.tiff', '.tif')):
                    file_path = os.path.join(root, file)
                    data = None
                    if source == 'ZZY':
                        data = parser.parse_zzy_filename(file)
                        if data:
                            if parser.should_include_zzy_record(data):
                                data['anneal_time'] = convert_time_to_h(str(data['anneal_time']))
                                data['growth_time'] = convert_time_to_h(str(data['growth_time']))
                            else:
                                data = None
                    elif source == 'XR':
                        data = parser.parse_xr_filename(
                            file,
                            folder_name=os.path.basename(root),
                        )
                    
                    if data:
                        data['file_path'] = file_path
                        data = calibrator.calibrate(data)
                        db.insert_image(data)

    print("Database population complete.")

if __name__ == "__main__":
    populate()
