import zipfile
import xml.etree.ElementTree as ET

def parse_docx(file_path):
    with zipfile.ZipFile(file_path) as docx:
        xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        paragraphs = []
        for p in tree.findall('.//w:p', ns):
            texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)

try:
    text = parse_docx(r"d:\CNTDATA\doc\3机器学习辅助的碳纳米管阵列的数字特征提取及生长工艺优化(1).docx")
    with open(r"d:\CNTDATA\CNTA_ML_Project\data\temp\doc_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Extract success")
except Exception as e:
    print(f"Error: {e}")
