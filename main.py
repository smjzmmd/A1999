from utils.config_loader import config
from core.adb_controller import ADBController

def main():
    # 获取配置值
    adb = ADBController(config.get('emulator', 'adb_path'), config.get('emulator', 'mumu_serial'))
    # 执行操作
    print(adb.connect())

if __name__ == "__main__":
    main()