import sqlite3
import os
import pytesseract
from PIL import Image
import re
from tqdm import tqdm

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'

def extract_mag(image_path):
    try:
        import cv2
        import numpy as np
        
        if not os.path.exists(image_path):
            return None
        
        # 使用 OpenCV 读取以支持更多处理
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            # 尝试使用 PIL 读取作为回退
            try:
                img_pil = Image.open(image_path).convert('L')
                img_cv = np.array(img_pil)
            except:
                return None
            
        h, w = img_cv.shape
        # 截取底部区域 (约 10% 的高度)
        crop_h = int(h * 0.1)
        data_bar = img_cv[h - crop_h : h, :]
        
        # 放大图像以提高 OCR 准确率
        data_bar = cv2.resize(data_bar, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        
        # 尝试几种不同的预处理方法
        # 1. 大津法自动阈值
        _, thresh_otsu = cv2.threshold(data_bar, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 2. 固定阈值 150
        _, thresh_f150 = cv2.threshold(data_bar, 150, 255, cv2.THRESH_BINARY_INV)
        # 3. 固定阈值 180 (原算法)
        _, thresh_f180 = cv2.threshold(data_bar, 180, 255, cv2.THRESH_BINARY_INV)
        
        methods = [
            ("Otsu", thresh_otsu),
            ("F150", thresh_f150),
            ("F180", thresh_f180)
        ]
        
        for name, processed_img in methods:
            text = pytesseract.image_to_string(processed_img, config='--psm 6')
            # 匹配数字后跟 x 或 k (如 20 000 x, 5.0k x)
            matches = re.findall(r'([\d\s\.,]+)\s*([xk])\b', text, re.IGNORECASE)
            
            for val_str, unit in matches:
                clean_val = val_str.replace(' ', '').replace(',', '.')
                try:
                    # 处理可能的 double dots
                    if clean_val.count('.') > 1:
                        parts = clean_val.split('.')
                        clean_val = parts[0] + '.' + ''.join(parts[1:])
                        
                    value = float(clean_val)
                    unit = unit.lower()
                    if value < 0.1: continue
                    
                    # 处理 20 000 x 这种中间有空格的情况 (20 000 被 OCR 为 20 和 000)
                    # 这里的 regex 已经包含了空格，但 float(clean_val) 会处理掉它们
                    
                    if unit == 'k':
                        actual_mag = int(value * 1000)
                    elif ' ' in val_str.strip() and value < 1000:
                        # 启发式：如果中间有空格且数值较小，可能是把 10 000 识别成了 10
                        # 但这个比较危险，先按正常处理
                        actual_mag = int(value)
                        if actual_mag < 100: # 可能是 20 000 丢了 000
                             actual_mag = int(value * 1000)
                    else:
                        actual_mag = int(value)
                    
                    if 100 <= actual_mag <= 1000000:
                        return actual_mag
                except:
                    continue
    except Exception as e:
        # print(f"Error: {e}")
        pass
    return None

def update_all_magnifications():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 获取所有 magnification 为空或为 0 的记录
    cursor.execute("SELECT id, file_path FROM images WHERE magnification IS NULL OR magnification = 0")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} records needing magnification extraction.")
    
    updated_count = 0
    for img_id, file_path in tqdm(rows):
        mag = extract_mag(file_path)
        if mag:
            cursor.execute("UPDATE images SET magnification = ? WHERE id = ?", (mag, img_id))
            updated_count += 1
            # 每 10 个 commit 一次
            if updated_count % 10 == 0:
                conn.commit()
    
    conn.commit()
    conn.close()
    print(f"Update complete. Successfully extracted {updated_count} magnifications.")

if __name__ == "__main__":
    update_all_magnifications()
