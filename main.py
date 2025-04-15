from dev_tools.connect_mumu import connect_mumu
from dev_tools.apprun import open_1999
from dev_tools.Image_calibration import find_image_in_screen

if __name__ == "__main__":
    connect_mumu()
    open_1999()

    template_path = "assets\login\logout.png"
    if find_image_in_screen(template_path):
        print("找到目标图片")
    else:
        print("未找到目标图片")
    