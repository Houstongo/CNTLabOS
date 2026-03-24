"""测试管径分布API"""
import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 测试查询ZZY数据
response = client.get('/api/images?source=ZZY&limit=1')
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'找到 {len(data)} 条记录')
    if data:
        item = data[0]
        print(f'ID: {item["id"]}')
        print(f'diameter_mean: {item.get("diameter_mean")}')
        print(f'diameter_std: {item.get("diameter_std")}')
        print(f'distribution: {item.get("diameter_distribution")}')
