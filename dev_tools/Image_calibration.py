import os
import tempfile
import cv2
import subprocess       

# ADB路径（根据实际情况修改）
ADB_PATH = "E:\\Program Files\\Netease\\MuMu Player 12\\shell\\adb.exe"

def capture_screen():
    """截取模拟器屏幕"""
    try:
        # 创建临时文件保存截图
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()
        
        # 使用ADB截图
        subprocess.run(f'"{ADB_PATH}" shell screencap -p /sdcard/screen.png', shell=True, check=True)
        subprocess.run(f'"{ADB_PATH}" pull /sdcard/screen.png "{temp_file.name}"', shell=True, check=True)
        
        # 读取截图
        screen = cv2.imread(temp_file.name)
        os.unlink(temp_file.name)  # 删除临时文件
        return screen
    except Exception as e:
        print(f"截图失败: {e}")
        return None

def find_image_in_screen(template_path, threshold=0.8):
    """
    在模拟器屏幕中查找模板图片
    :param template_path: 模板图片路径
    :param threshold: 匹配阈值(0-1)
    :return: 找到返回True，否则False
    """
    # if not connect_mumu():
    #     return False
    
    # 读取模板图片
    template = cv2.imread(template_path)
    if template is None:
        print(f"无法读取模板图片: {template_path}")
        return False
    
    # 获取屏幕截图
    screen = capture_screen()
    if screen is None:
        return False
    
    # 模板匹配
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_val >= threshold

if __name__ == "__main__":
    # 模板图片路径
    template_path = "assets\login\logout.png"
    if find_image_in_screen(template_path):
        print("找到目标图片")
    else:
        print("未找到目标图片")