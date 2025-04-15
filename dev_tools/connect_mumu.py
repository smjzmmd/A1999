import subprocess

# MuMu模拟器默认ADB端口
MUMU_ADB_PORT = 7555
# ADB路径（根据实际情况修改）
ADB_PATH = "E:\\Program Files\\Netease\\MuMu Player 12\\shell\\adb.exe"

def connect_mumu():
    """连接MuMu模拟器ADB"""
    try:
        subprocess.run(f'"{ADB_PATH}" connect 127.0.0.1:{MUMU_ADB_PORT}', shell=True, check=True)
        print("成功连接MuMu模拟器")
        return True
    except subprocess.CalledProcessError as e:
        print(f"连接MuMu模拟器失败: {e}")
        return False
