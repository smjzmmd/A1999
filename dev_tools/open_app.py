import subprocess

def open_app():
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
    open_app()