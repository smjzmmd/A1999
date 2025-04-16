# A1999
# 全部由ai生成
mumu_automation/
│
├── config/                  # 配置文件目录
│   ├── config.yaml          # 主配置文件
│   ├── device_config.yaml   # 设备特定配置
│   └── game_config.yaml     # 游戏特定配置
│
├── core/                    # 核心功能模块
│   ├── __init__.py
│   ├── adb_controller.py    # ADB命令封装
│   ├── image_processor.py   # 图像处理模块
│   ├── ocr_processor.py     # OCR文字识别模块
│   └── mumu_controller.py   # Mumu模拟器控制核心
│
├── scripts/                 # 自动化脚本
│   ├── __init__.py
│   ├── common_operations/   # 通用操作
│   │   ├── login.py
│   │   ├── daily_tasks.py
│   │   └── ...
│   └── game_specific/       # 游戏特定脚本
│       ├── game1/
│       └── game2/
│
├── resources/               # 资源文件
│   ├── images/              # 图片模板
│   │   ├── buttons/
│   │   ├── icons/
│   │   └── ...
│   ├── data/                # 数据文件
│   └── logs/                # 日志目录
│
├── tests/                   # 测试代码
│   ├── unit_tests/
│   └── integration_tests/
│
├── utils/                   # 工具类
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   ├── helper.py            # 辅助函数
│   └── scheduler.py         # 任务调度
│
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明