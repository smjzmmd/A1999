import subprocess

# ADB路径（根据实际情况修改）
ADB_PATH = "E:\\Program Files\\Netease\\MuMu Player 12\\shell\\adb.exe"
# 1999包名 adb shell cmd package resolve-activity --brief com.shenlan.m.reverse1999
REVERSE1999_PACKAGE = "com.shenlan.m.reverse1999"
REVERSE1999_ACTIVITY = "com.ssgame.mobile.gamesdk.frame.AppStartUpActivity"

def open_1999():
    """打开1999"""
    try:
        # 启动游戏
        subprocess.run(f'"{ADB_PATH}" shell am start -n {REVERSE1999_PACKAGE}/{REVERSE1999_ACTIVITY}', shell=True, check=True)
        print("1999启动中...")
    except subprocess.CalledProcessError as e:
        print(f"1999启动失败: {e}")
        return False
    return True

if __name__ == "__main__":
    open_1999()