import cv2
import numpy as np
import os

class SEMImageProcessor:
    def __init__(self, environmental_name="lab_agent"):
        self.env = environmental_name

    def preprocess(self, img):
        """增强图像对比度，方便特征提取"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 使用 CLAHE (有限对比适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    def detect_scale_bar(self, img):
        """
        定位 SEM 图像底部的标尺。
        通常标尺是一根明显的白线。
        返回: (pixel_length, physical_value, unit)
        """
        # SEM 标尺通常在底部的 10% 区域
        h, w = img.shape[:2]
        bottom_area = img[int(h*0.9):, :]
        
        # 转换为灰度
        if len(bottom_area.shape) == 3:
            gray_bottom = cv2.cvtColor(bottom_area, cv2.COLOR_BGR2GRAY)
        else:
            gray_bottom = bottom_area
            
        # 二值化以寻找白线
        _, thresh = cv2.threshold(gray_bottom, 200, 255, cv2.THRESH_BINARY)
        
        # 使用形态学操作连接断裂的线（如果有）
        kernel = np.ones((1, 50), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选出最像标尺的线（细长的矩形）
        scale_line_contour = None
        max_width = 0
        
        for cnt in contours:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            # 标尺线通常比较长且窄
            if w_cnt > 50 and h_cnt < 20: 
                if w_cnt > max_width:
                    max_width = w_cnt
                    scale_line_contour = (x, y, w_cnt, h_cnt)
        
        if scale_line_contour:
            # 返回检测到的像素长度
            return scale_line_contour[2] 
        return None

def test_on_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    processor = SEMImageProcessor()
    px_len = processor.detect_scale_bar(img)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Detected Scale Bar Pixel Length: {px_len}")
    
    # 获取文件名信息
    # 比如 C10A1.tiff
    return px_len

if __name__ == "__main__":
    # 测试一张典型的图像
    test_img = r'd:\CNTDATA\250313 T750 3h L250\C10A1.tiff'
    test_on_single_image(test_img)
