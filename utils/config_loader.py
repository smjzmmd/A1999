import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "A1999" / "config" / "config.yaml"
        
        # 检查配置文件是否存在
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        # 加载YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # # 设置ADB路径为绝对路径
        # if 'emulator' in self._config and 'adb_path' in self._config['emulator']:
        #     adb_path = Path(self._config['emulator']['adb_path'])
        #     if not adb_path.is_absolute():
        #         adb_path = project_root / adb_path
        #     self._config['emulator']['adb_path'] = str(adb_path.resolve())
    
    def get(self, *keys) -> Any:
        """获取嵌套配置值"""
        result = self._config
        for key in keys:
            result = result[key]
        return result
    
    @property
    def all_config(self) -> Dict:
        """获取全部配置"""
        return self._config

# 单例实例
config = ConfigLoader()