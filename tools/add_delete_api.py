"""添加逻辑删除功能的API端点"""
import sqlite3

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"

# 要添加的API端点代码
api_code = '''
# 标记图像为已删除
@app.put("/api/images/{image_id}/delete")
async def soft_delete_image(image_id: int):
    """
    标记图像为已删除（逻辑删除）
    - is_deleted = 1: 已删除
    - is_deleted = 0: 正常
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (image_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    conn.commit()
    conn.close()

    return {"status": "success", "deleted_id": image_id}


# 恢复已删除的图像
@app.put("/api/images/{image_id}/restore")
async def restore_deleted_image(image_id: int):
    """
    恢复已删除的图像
    - is_deleted = 0: 恢复正常状态
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("UPDATE images SET is_deleted = 0 WHERE id = ?", (image_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    conn.commit()
    conn.close()

    return {"status": "success", "restored_id": image_id}
'''

# 查找插入位置
with open(r'd:\CNTDATA\CNTA_ML_Project\backend\main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 @app.get("/api/summary") 之后插入
insert_marker = '@app.get("/api/summary")'

if insert_marker in content:
    # 在marker之后插入新代码
    insert_pos = content.find(insert_marker) + len(insert_marker)

    new_content = content[:insert_pos] + '\n' + api_code + '\n' + content[insert_pos:]

    with open(r'd:\CNTDATA\CNTA_ML_Project\backend\main.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("已添加以下API端点:")
    print("  PUT /api/images/{image_id}/delete  - 标记图像为已删除")
    print("  PUT /api/images/{image_id}/restore - 恢复已删除的图像")
    print("\n请重启FastAPI服务以使更改生效")
else:
    print(f"未找到插入点: {insert_marker}")
