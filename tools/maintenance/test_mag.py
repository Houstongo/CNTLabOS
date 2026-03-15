import pytesseract
from PIL import Image
import os
import re

def extract_mag(image_path):
    try:
        img = Image.open(image_path)
        w, h = img.size
        # 截取底部区域
        crop_h = int(h * 0.08) # 截取底部 8%
        data_bar = img.crop((0, h - crop_h, w, h))
        
        # 放大图片提高识别率
        data_bar = data_bar.resize((data_bar.width * 2, data_bar.height * 2), Image.Resampling.LANCZOS)
        
        # 转灰度
        gray = data_bar.convert('L')
        # 二值化：因为是黑底白字，我们将白字转黑
        # 提高对比度
        thresh = gray.point(lambda x: 0 if x > 180 else 255)
        
        # thresh.save(r"d:\CNTDATA\CNTA_ML_Project\src\analysis\debug_mag.png") # 调试用
        
        # psm 6: Assume a single uniform block of text.
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        print(f"OCR Raw Text:\n{text}")
        
        # 寻找形如 20.000 x, 20 000 x, 5.00 k, 5000 x 的模式
        # 匹配 数字（允许空格、逗号、点） + 空格? + x 或 k
        matches = re.findall(r'([\d\s\.,]+)\s*([xk])\b', text, re.IGNORECASE)
        print(f"Found matches: {matches}")
        
        for val_str, unit in matches:
            # 清理数值：移除空格，将逗号转点（如果是小数点的话，或者直接移除如果是千分位）
            # SEM 的 20 000 x 通常是 20.000 (2万倍)
            clean_val = val_str.replace(' ', '').replace(',', '.')
            try:
                value = float(clean_val)
                unit = unit.lower()
                # 过滤掉不合理的数值（比如 8um 的 8）
                if value < 0.1: continue
                
                # 如果是 20.000 x 且单位是 x，通常意味着 20000 倍（取决于 SEM 习惯）
                # 或者是 20 000 x 读成了 20.000 x
                # 检查原字符是否有空格间隔
                if ' ' in val_str.strip() and value < 1000:
                   # 可能是 20 000 读成了 20.000
                   actual_mag = int(value * 1000)
                elif unit == 'k':
                    actual_mag = int(value * 1000)
                else:
                    actual_mag = int(value)
                
                # CNTA SEM 通常是 500x 到 100000x
                if 100 < actual_mag < 500000:
                    return actual_mag
            except:
                continue
            
    except Exception as e:
        print(f"Error during extraction: {e}")
    return None

if __name__ == "__main__":
    test_path = r'd:\CNTDATA\XR\250301 T800 3h L250\C3A2.tiff'
    mag = extract_mag(test_path)
    print(f"Extracted Magnification: {mag}")
