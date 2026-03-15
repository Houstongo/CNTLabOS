# CNTA ML Project Context

## 项目目标
利用机器学习辅助碳纳米管阵列 (CNTA) 的数字特征提取与生长工艺优化。

## 核心规则 (MUST FOLLOW)

### 1. 脚本与工具管理规则
- **禁止在根目录或 `src` 目录随意生成临时脚本。**
- **统一入口**：所有功能性任务通过根目录的 `manage.py` 调度。
- **工具存放**：
    - 通用工具类代码存放在 `src/utils/`。
    - 具体的任务脚本存放在 `tools/` 目录下。
    - 数据清理和初始化脚本存放在 `tools/maintenance/`。
- **命名规范**：工具脚本应使用描述性名称，如 `tools/db_maintenance.py` 而非 `tools/test1.py`。

### 2. 代码组织结构
- `backend/`: FastAPI 后端服务及核心业务逻辑。
- `frontend/`: 基于 HTML/JS 的前端驾驶舱。
- `src/`: 核心算法、数据模型和解析模块（被其他模块引用）。
- `tools/`: 独立的命令行工具和自动化脚本。
- `database/`: SQLite 数据库文件存放地。
- `data/`: 原始数据存储路径。
- `docs/`: 项目文档和技术规范。

### 3. 技术协议
- **后端**: FastAPI, Python 3.x
- **前端**: Vanilla JS, TailwindCSS, Fetch API
- **数据库**: SQLite (路径: `database/cnta_experiments.sqlite`)
- **通讯**: 默认 API 运行在 `http://localhost:8000`

---
## 常用命令
- 启动后端: `python manage.py run-backend`
- 启动前端服务: `python manage.py run-frontend`
- 初始化数据库: `python manage.py init-db`
