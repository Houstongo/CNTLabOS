import os

def create_structure(base_path):
    directories = [
        "backend/core",
        "backend/utils",
        "backend/data/db",
        "backend/data/temp",
        "frontend/src",
        "database"
    ]
    for d in directories:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

if __name__ == "__main__":
    create_structure(r"d:\CNTDATA\CNTA_ML_Project")
