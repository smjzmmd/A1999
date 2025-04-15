import subprocess

def connect_mumu(ADB_PATH, MUMU_ADB_PORT):
    """连接MuMu模拟器ADB"""
    try:
        subprocess.run(f'"{ADB_PATH}" connect 127.0.0.1:{MUMU_ADB_PORT}', shell=True, check=True)
        print("成功连接MuMu模拟器")
        return True
    except subprocess.CalledProcessError as e:
        print(f"连接MuMu模拟器失败: {e}")
        return False

if __name__ == "__main__":
    connect_mumu()