import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .adb_controller import ADBController
from .image_processor import ImageProcessor

class MuMuController:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MuMu 模拟器控制器
        
        :param config: 配置字典，包含以下关键字段:
            - adb_path: ADB 可执行文件路径
            - mumu_serial: 模拟器设备序列号
            - resolution: 模拟器分辨率 (width, height)
            - screenshot_dir: 截图保存目录
            - default_wait: 默认等待时间(秒)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 初始化ADB控制器
        self.adb = ADBController(
            adb_path=config['adb_path'],
            device_serial=config['mumu_serial']
        )
        
        # 初始化图像处理器
        self.image_processor = ImageProcessor()
        
        # 设备分辨率
        self.width, self.height = config['resolution']
        self.logger.info(f"设备分辨率: {self.width}x{self.height}")
        
        # 状态标志
        self._connected = False
        self._game_running = False
        
        # 连接设备
        self.connect()

    def connect(self) -> bool:
        """连接模拟器设备"""
        try:
            if self.adb.connect():
                self._connected = True
                self.logger.info("模拟器连接成功")
                return True
        except Exception as e:
            self.logger.error(f"连接模拟器失败: {str(e)}")
            self._connected = False
        return False

    def disconnect(self):
        """断开模拟器连接"""
        self.adb.disconnect()
        self._connected = False
        self.logger.info("已断开模拟器连接")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def take_screenshot(self, save: bool = False, filename: str = None) -> np.ndarray:
        """
        获取屏幕截图
        
        :param save: 是否保存截图文件
        :param filename: 保存的文件名(不含扩展名)
        :return: 截图numpy数组(BGR格式)
        """
        if not self._connected:
            self.logger.warning("未连接模拟器，无法截图")
            return None
            
        try:
            # 使用ADB获取截图
            screenshot = self.adb.screencap()
            
            if save and filename:
                save_path = Path(self.config['screenshot_dir']) / f"{filename}.png"
                cv2.imwrite(str(save_path), screenshot)
                self.logger.debug(f"截图已保存到: {save_path}")
                
            return screenshot
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            return None

    def find_image(self, template_path: str, threshold: float = 0.8, 
                  region: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int]]:
        """
        在屏幕上查找图像
        
        :param template_path: 模板图像路径
        :param threshold: 匹配阈值(0-1)
        :param region: 搜索区域(x1,y1,x2,y2)
        :return: 匹配位置的(x,y)坐标，未找到返回None
        """
        screenshot = self.take_screenshot()
        if screenshot is None:
            return None
            
        try:
            # 加载模板图像
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                self.logger.error(f"无法加载模板图像: {template_path}")
                return None
                
            # 如果在特定区域搜索
            if region:
                x1, y1, x2, y2 = region
                screenshot = screenshot[y1:y2, x1:x2]
                
            # 使用图像处理器进行匹配
            result = self.image_processor.find_template(
                screenshot, template, threshold
            )
            
            if result:
                x, y = result
                if region:  # 如果在区域内找到，需要加上偏移量
                    x += region[0]
                    y += region[1]
                self.logger.debug(f"在位置 ({x}, {y}) 找到图像: {template_path}")
                return x, y
                
            return None
        except Exception as e:
            self.logger.error(f"图像查找失败: {str(e)}")
            return None

    def click(self, x: int, y: int, delay: float = 0.5):
        """
        点击屏幕指定位置
        
        :param x: 横坐标
        :param y: 纵坐标
        :param delay: 点击后延迟(秒)
        """
        if not self._connected:
            self.logger.warning("未连接模拟器，无法点击")
            return
            
        try:
            self.adb.tap(x, y)
            time.sleep(delay)
            self.logger.debug(f"点击位置: ({x}, {y})")
        except Exception as e:
            self.logger.error(f"点击失败: {str(e)}")

    def swipe(self, start_x: int, start_y: int, 
             end_x: int, end_y: int, duration: int = 300):
        """
        滑动屏幕
        
        :param start_x: 起始x坐标
        :param start_y: 起始y坐标
        :param end_x: 结束x坐标
        :param end_y: 结束y坐标
        :param duration: 滑动持续时间(毫秒)
        """
        if not self._connected:
            self.logger.warning("未连接模拟器，无法滑动")
            return
            
        try:
            self.adb.swipe(start_x, start_y, end_x, end_y, duration)
            self.logger.debug(f"从 ({start_x}, {start_y}) 滑动到 ({end_x}, {end_y})")
        except Exception as e:
            self.logger.error(f"滑动失败: {str(e)}")

    def wait_until_find(self, template_path: str, timeout: int = 10, 
                       interval: float = 0.5, **kwargs) -> Optional[Tuple[int, int]]:
        """
        等待直到找到指定图像
        
        :param template_path: 模板图像路径
        :param timeout: 超时时间(秒)
        :param interval: 检查间隔(秒)
        :param kwargs: 传递给find_image的其他参数
        :return: 找到的位置坐标，超时返回None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos = self.find_image(template_path, **kwargs)
            if pos:
                return pos
            time.sleep(interval)
            
        self.logger.warning(f"等待 {template_path} 超时 ({timeout}秒)")
        return None

    def click_image(self, template_path: str, timeout: int = 10, **kwargs):
        """
        找到图像并点击
        
        :param template_path: 模板图像路径
        :param timeout: 查找超时时间(秒)
        :param kwargs: 传递给find_image的其他参数
        :return: 是否成功点击
        """
        pos = self.wait_until_find(template_path, timeout=timeout, **kwargs)
        if pos:
            self.click(*pos)
            return True
        return False

    def input_text(self, text: str):
        """
        输入文本
        
        :param text: 要输入的文本
        """
        if not self._connected:
            self.logger.warning("未连接模拟器，无法输入文本")
            return
            
        try:
            self.adb.input_text(text)
            self.logger.debug(f"输入文本: {text}")
        except Exception as e:
            self.logger.error(f"文本输入失败: {str(e)}")

    def restart_app(self, package_name: str):
        """
        重启应用
        
        :param package_name: 应用包名
        """
        if not self._connected:
            self.logger.warning("未连接模拟器，无法重启应用")
            return
            
        try:
            self.adb.stop_app(package_name)
            time.sleep(2)
            self.adb.start_app(package_name)
            self.logger.info(f"已重启应用: {package_name}")
        except Exception as e:
            self.logger.error(f"重启应用失败: {str(e)}")