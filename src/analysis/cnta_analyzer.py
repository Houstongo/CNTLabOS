import cv2
import numpy as np
import os
import pandas as pd

class CNTAAnalyzer:
    def __init__(self, processor):
        self.processor = processor

    def analyze_orientation(self, img):
        """
        使用快速傅里叶变换 (FFT) 分析 CNT 的取向性。
        返回: 取向分布的熵或主要方向。
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 1. 预处理
        gray = self.processor.preprocess(gray)
        
        # 2. 计算 FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # 3. 分析频谱的角分布
        # 在频率域中，如果 CNT 是垂直排列的，频谱会呈现水平延伸。
        h, w = magnitude_spectrum.shape
        center = (h // 2, w // 2)
        
        # 简化版：计算图像梯度的直方图 (HOG 思想)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # 过滤掉低梯度的区域（背景）
        mask = mag > np.mean(mag)
        angles = ang[mask]
        
        # 计算角度分布的方差，方差越小，取向越一致
        orientation_consistency = 1.0 / (np.std(angles) + 1e-6)
        
        return {
            'orientation_consistency': orientation_consistency,
            'mean_angle': np.mean(angles)
        }

    def estimate_height(self, img):
        """
        针对截面图（Cross-section），估计 CNT 阵列的高度。
        假设阵列是两个平行的边界。
        """
        gray = self.processor.preprocess(img)
        # 使用边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 统计每一行的像素密度
        row_density = np.sum(edges, axis=1)
        
        # 寻找密度突变的位置（顶部和底部衬底）
        # 这里使用简单的阈值，实际可能需要更复杂的峰值检测
        threshold = np.max(row_density) * 0.3
        active_rows = np.where(row_density > threshold)[0]
        
        if len(active_rows) > 0:
            pixel_height = active_rows[-1] - active_rows[0]
            return pixel_height
        return None

if __name__ == "__main__":
    import sys
    # 将 utils 目录添加到路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from image_processor import SEMImageProcessor
    
    proc = SEMImageProcessor()
    analyzer = CNTAAnalyzer(proc)
    
    # 测试文件
    test_img_path = r'd:\CNTDATA\250313 T750 3h L250\C10A1.tiff'
    img = cv2.imread(test_img_path)
    
    if img is not None:
        orientation = analyzer.analyze_orientation(img)
        height_px = analyzer.estimate_height(img)
        px_scale = proc.detect_scale_bar(img)
        
        print(f"--- Analysis Results for {os.path.basename(test_img_path)} ---")
        print(f"Orientation Consistency: {orientation['orientation_consistency']:.4f}")
        print(f"Estimated Height (pixels): {height_px}")
        print(f"Scale Bar (pixels): {px_scale}")
        
        if px_scale and height_px:
            # 假设检测到的是 1um 的标尺（实际需 OCR 配合）
            actual_height = height_px / px_scale 
            print(f"Estimated Height (normalized to scale unit): {actual_height:.2f}")
