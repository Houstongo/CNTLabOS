import pandas as pd
import os

xlsx_path = r"D:\电脑管家迁移文件\xwechat_files\wxid_697jsgpwq0zg22_414a\msg\file\2026-03\管式炉温区.xlsx"

def read_xlsx(path):
    print(f"\n--- Reading Excel: {os.path.basename(path)} ---")
    try:
        df_dict = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in df_dict.items():
            print(f"\nSheet: {sheet_name}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.head(20)) # Show more rows
            # Save a summary if needed
            df.to_csv(f"d:/CNTDATA/CNTA_ML_Project/scripts/{sheet_name}.csv", index=False)
    except Exception as e:
        print(f"Error reading Excel: {e}")

if __name__ == "__main__":
    if os.path.exists(xlsx_path):
        read_xlsx(xlsx_path)
    else:
        print(f"File not found: {xlsx_path}")
