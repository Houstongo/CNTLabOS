import os
import zipfile
import shutil

def extract_images_from_pptx(pptx_path, output_folder):
    if not os.path.exists(pptx_path):
        print(f"File not found: {pptx_path}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    with zipfile.ZipFile(pptx_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith('ppt/media/'):
                # 提取图像
                filename = os.path.basename(file_info.filename)
                target_path = os.path.join(output_folder, filename)
                with zip_ref.open(file_info.filename) as source, open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted: {filename}")

if __name__ == "__main__":
    with open(r"d:\CNTDATA\CNTA_ML_Project\data\temp\pptx_path.txt", "r", encoding="utf-8") as f:
        path = f.read().strip()
    
    extract_images_from_pptx(path, r"d:\CNTDATA\CNTA_ML_Project\backend\data\temp\ppt_assets")
