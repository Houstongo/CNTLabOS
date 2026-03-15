# Tools 目录说明

该目录用于存放项目中所有的可执行工具脚本。为了维护项目整洁，请遵循以下规则：

## 目录结构
- `/maintenance`: 数据库维护、系统初始化、元数据同步等运维脚本。
- `/data_processing`: 原始数据（Excel, PPTX, CSV）的解析与预处理脚本。
- `/experiments`: 临时的算法测试或实验性质的脚本（注意：长期稳定的代码应移至 `src/`）。

## 使用方式
推荐通过根目录的 `manage.py` 来统一运行常用任务。例如：
```bash
python manage.py analyze  # 运行批量分析
python manage.py init-db  # 初始化数据库
```

## 开发规范
1. **禁止在根目录生成临时脚本**。
2. 脚本应具备基本的命令行参数提示（使用 `argparse`）。
3. 如果脚本依赖项目内部模块，请确保在运行前设置好 `PYTHONPATH` 或通过 `manage.py` 调度。
4. 所有的工具脚本注释必须使用 **中文**。
