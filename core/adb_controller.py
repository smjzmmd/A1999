import subprocess
import re
import time
import logging
from typing import Optional, Tuple, List, Union, Any
import cv2
import numpy as np
from pathlib import Path

class ADBController:
    def __init__(self, adb_path: str, device_serial: str = None, timeout: int = 30):
        """
        初始化ADB控制器
        
        :param adb_path: ADB可执行文件路径
        :param device_serial: 设备序列号(可选)
        :param timeout: 命令超时时间(秒)
        """
        self.logger = logging.getLogger(__name__)
        self.adb_path = adb_path
        self.device_serial = device_serial
        self.timeout = timeout
        self._connected = False
        
        # 验证ADB路径
        if not self._validate_adb_path():
            raise FileNotFoundError(f"ADB可执行文件不存在: {self.adb_path}")

    def _run_command(self, command: Union[str, List[str]], adb_args: str = "") -> Tuple[bool, str]:
        """
        执行ADB命令
        
        :param command: 要执行的命令
        :param adb_args: 额外的ADB参数
        :return: (成功状态, 命令输出)
        """
        if isinstance(command, str):
            command = command.split()
            
        full_cmd = [str(self.adb_path)]
        if self.device_serial:
            full_cmd.extend(["-s", self.device_serial])
        if adb_args:
            full_cmd.extend(adb_args.split())
        full_cmd.extend(command)
        
        self.logger.debug(f"执行ADB命令: {' '.join(full_cmd)}")
        
        try:
            result = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                self.logger.error(f"ADB命令失败: {' '.join(full_cmd)}\n错误: {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"ADB命令超时: {' '.join(full_cmd)}")
            return False, "Command timeout"
        except Exception as e:
            self.logger.error(f"执行ADB命令异常: {str(e)}")
            return False, str(e)

    def _validate_adb_path(self) -> bool:
        """验证ADB路径是否有效"""
        try:
            success, _ = self._run_command("version")
            return success
        except:
            return False

    def connect(self) -> bool:
        """连接到设备"""
        if self.device_serial and ":" in self.device_serial:
            # 如果是网络设备，先尝试连接
            success, _ = self._run_command(f"connect {self.device_serial}")
            if not success:
                return False
                
        # 检查设备是否已连接
        success, output = self._run_command("devices")
        if not success:
            return False
            
        devices = re.findall(r"(\S+)\tdevice", output)
        if not devices:
            self.logger.error("没有找到已连接的设备")
            return False
            
        if self.device_serial and self.device_serial not in devices:
            self.logger.error(f"设备 {self.device_serial} 未连接")
            return False
            
        self._connected = True
        self.logger.info(f"已连接到设备: {self.device_serial or devices[0]}")
        return True

    def disconnect(self):
        """断开设备连接"""
        if self.device_serial and ":" in self.device_serial:
            self._run_command(f"disconnect {self.device_serial}")
        self._connected = False
        self.logger.info("已断开设备连接")

    def is_connected(self) -> bool:
        """检查设备是否已连接"""
        return self._connected

    def screencap(self, save_path: str = None) -> Optional[np.ndarray]:
        """
        获取屏幕截图
        
        :param save_path: 保存路径(可选)
        :return: 截图numpy数组(BGR格式)，失败返回None
        """
        try:
            # 使用ADB命令截图并传输到本地
            temp_file = "/sdcard/screencap.png"
            self._run_command(f"shell screencap -p {temp_file}")
            
            # 从设备拉取截图
            local_path = save_path or "/tmp/adb_screencap.png"
            self._run_command(f"pull {temp_file} {local_path}")
            
            # 删除设备上的临时文件
            self._run_command(f"shell rm {temp_file}")
            
            # 读取图像文件
            img = cv2.imread(local_path)
            if img is None:
                raise ValueError("无法读取截图文件")
                
            if not save_path:
                Path(local_path).unlink()  # 删除临时文件
                
            return img
            
        except Exception as e:
            self.logger.error(f"截图失败: {str(e)}")
            return None

    def tap(self, x: int, y: int):
        """点击屏幕位置"""
        return self._run_command(f"shell input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """
        滑动屏幕
        
        :param duration: 滑动持续时间(毫秒)
        """
        return self._run_command(
            f"shell input swipe {x1} {y1} {x2} {y2} {duration}"
        )

    def input_text(self, text: str):
        """输入文本"""
        # 转义特殊字符
        text = text.replace(" ", "%s").replace("'", "'\"'\"'")
        return self._run_command(f"shell input text '{text}'")

    def key_event(self, keycode: int):
        """发送按键事件"""
        return self._run_command(f"shell input keyevent {keycode}")

    def start_app(self, package_name: str, activity: str = None):
        """启动应用"""
        if activity:
            return self._run_command(
                f"shell am start -n {package_name}/{activity}"
            )
        return self._run_command(f"shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1")

    def stop_app(self, package_name: str):
        """停止应用"""
        return self._run_command(f"shell am force-stop {package_name}")

    def get_current_activity(self) -> Optional[str]:
        """获取当前Activity"""
        success, output = self._run_command("shell dumpsys window windows | grep -E 'mCurrentFocus'")
        if not success:
            return None
            
        match = re.search(r"\{.*?\s+u0\s+(\S+?)/(\S+?)\}", output)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return None

    def get_device_info(self) -> dict:
        """获取设备信息"""
        info = {}
        
        # 获取设备型号
        success, model = self._run_command("shell getprop ro.product.model")
        if success:
            info['model'] = model.strip()
            
        # 获取Android版本
        success, version = self._run_command("shell getprop ro.build.version.release")
        if success:
            info['android_version'] = version.strip()
            
        # 获取分辨率
        success, display = self._run_command("shell dumpsys window displays | grep -E 'init='")
        if success:
            match = re.search(r"init=(\d+)x(\d+)", display)
            if match:
                info['resolution'] = (int(match.group(1)), int(match.group(2)))
                
        return info

    def install_app(self, apk_path: str) -> bool:
        """安装APK"""
        success, output = self._run_command(f"install {apk_path}")
        if success:
            self.logger.info(f"成功安装APK: {apk_path}")
        else:
            self.logger.error(f"安装APK失败: {output}")
        return success

    def uninstall_app(self, package_name: str) -> bool:
        """卸载应用"""
        success, output = self._run_command(f"uninstall {package_name}")
        if success:
            self.logger.info(f"成功卸载应用: {package_name}")
        else:
            self.logger.error(f"卸载应用失败: {output}")
        return success

    def list_packages(self) -> List[str]:
        """列出所有已安装包名"""
        success, output = self._run_command("shell pm list packages")
        if not success:
            return []
        return [line.split(":")[1] for line in output.splitlines() if line.startswith("package:")]

    def push_file(self, local_path: str, device_path: str) -> bool:
        """推送文件到设备"""
        return self._run_command(f"push {local_path} {device_path}")[0]

    def pull_file(self, device_path: str, local_path: str) -> bool:
        """从设备拉取文件"""
        return self._run_command(f"pull {device_path} {local_path}")[0]