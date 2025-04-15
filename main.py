from dev_tools.connect_mumu import connect_mumu
from dev_tools.open_app import open_app
from dev_tools.Image_calibration import find_image_in_screen

class Main:
    ADB_PATH = ''
    MUMU_ADB_PORT= 0
    REVERSE1999_PACKAGE= ''
    REVERSE1999_ACTIVITY= ''
    def __init__(self):
        # ADB 路径
        self.ADB_PATH = r"D:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe"

        # MuMu模拟器默认ADB端口
        self.MUMU_ADB_PORT = 7555

        # 1999包名 adb shell cmd package resolve-activity --brief com.shenlan.m.reverse1999
        self.REVERSE1999_PACKAGE = "com.shenlan.m.reverse1999"
        self.REVERSE1999_ACTIVITY = "com.ssgame.mobile.gamesdk.frame.AppStartUpActivity"


if __name__ == "__main__":
    m = Main()
    connect_mumu(m.ADB_PATH,m.MUMU_ADB_PORT)
    # open_1999()

    # template_path = r"assets\login\logout.png"
    # if find_image_in_screen(template_path):
    #     print("找到目标图片")
    # else:
    #     print("未找到目标图片")
    