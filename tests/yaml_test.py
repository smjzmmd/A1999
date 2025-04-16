from ..utils.config_loader import config

def main():
    # 获取配置值
    adb_path = config.get('emulator', 'adb_path')
    port = config.get('emulator', 'port')
    package_name = config.get('game', 'package_name')
    
    print(f"ADB路径: {adb_path}")
    print(f"模拟器端口: {port}")
    print(f"游戏包名: {package_name}")
    
    # 获取整个emulator配置节
    emulator_config = config.get('emulator')
    print(f"完整模拟器配置: {emulator_config}")

if __name__ == "__main__":
    main()