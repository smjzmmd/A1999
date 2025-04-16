import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pytesseract
from PIL import Image
import re

class OCRProcessor:
    def __init__(self, 
                 tesseract_path: Optional[str] = None,
                 lang: str = 'chi_sim+eng',
                 debug: bool = False):
        """
        初始化OCR处理器
        
        :param tesseract_path: Tesseract OCR路径(如果不在系统PATH中)
        :param lang: 使用的语言(默认中文简体+英文)
        :param debug: 是否启用调试模式
        """
        self.logger = logging.getLogger(__name__)
        self.lang = lang
        self.debug = debug
        self.debug_dir = Path("debug_ocr")
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)
            
        # 白名单配置(可根据需要调整)
        self.digit_whitelist = "0123456789"
        self.alphanumeric_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        self.chinese_whitelist = None  # 中文不使用白名单
        
        self._validate_tesseract()

    def _validate_tesseract(self):
        """验证Tesseract是否可用"""
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract OCR 版本: {version}")
            return True
        except Exception as e:
            self.logger.error(f"Tesseract OCR 初始化失败: {str(e)}")
            raise RuntimeError("Tesseract OCR 不可用") from e

    def _save_debug_image(self, image: np.ndarray, name: str):
        """保存调试图像"""
        if self.debug:
            timestamp = int(time.time() * 1000)
            path = self.debug_dir / f"{timestamp}_{name}.png"
            cv2.imwrite(str(path), image)
            self.logger.debug(f"保存OCR调试图像: {path}")

    def preprocess_image(self, image: np.ndarray, 
                        denoise: bool = True,
                        binarize: bool = True,
                        invert: bool = False) -> np.ndarray:
        """
        预处理图像以提高OCR准确率
        
        :param image: 输入图像(BGR格式)
        :param denoise: 是否降噪
        :param binarize: 是否二值化
        :param invert: 是否反转颜色(黑底白字)
        :return: 处理后的图像
        """
        # 转为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 降噪
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
        # 二值化
        if binarize:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
        # 反转颜色
        if invert:
            gray = cv2.bitwise_not(gray)
            
        if self.debug:
            self._save_debug_image(gray, "preprocessed")
            
        return gray

    def recognize_text(self, image: np.ndarray,
                      config: Optional[str] = None,
                      preprocess: bool = True) -> str:
        """
        识别图像中的文字
        
        :param image: 输入图像
        :param config: 自定义Tesseract配置
        :param preprocess: 是否自动预处理图像
        :return: 识别出的文本
        """
        try:
            if preprocess:
                image = self.preprocess_image(image)
                
            # 转换为PIL图像
            pil_img = Image.fromarray(image)
            
            # 默认配置
            if config is None:
                config = f'--oem 3 --psm 6 -l {self.lang}'
                
            text = pytesseract.image_to_string(pil_img, config=config)
            
            # 后处理: 去除多余空格和换行
            text = ' '.join(text.split()).strip()
            
            self.logger.debug(f"OCR识别结果: {text}")
            return text
            
        except Exception as e:
            self.logger.error(f"文字识别失败: {str(e)}")
            return ""

    def recognize_digits(self, image: np.ndarray) -> str:
        """
        专门识别数字
        
        :return: 识别出的数字字符串
        """
        config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={self.digit_whitelist}'
        return self.recognize_text(image, config=config)

    def recognize_alphanumeric(self, image: np.ndarray) -> str:
        """
        专门识别字母和数字
        
        :return: 识别出的字母数字字符串
        """
        config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={self.alphanumeric_whitelist}'
        return self.recognize_text(image, config=config)

    def recognize_specific_text(self, image: np.ndarray, 
                              expected_text: str,
                              similarity_threshold: float = 0.8) -> bool:
        """
        识别特定文本是否存在
        
        :param expected_text: 期望的文本
        :param similarity_threshold: 相似度阈值
        :return: 是否存在
        """
        recognized = self.recognize_text(image)
        return self._text_similarity(recognized, expected_text) >= similarity_threshold

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度(简单实现)
        
        :return: 相似度(0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # 转换为相同大小写
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # 完全匹配
        if text1 == text2:
            return 1.0
            
        # 部分匹配
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = set1 & set2
        union = set1 | set2
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

    def find_text_position(self, image: np.ndarray,
                          target_text: str,
                          region: Optional[Tuple[int, int, int, int]] = None,
                          partial_match: bool = False) -> Optional[Tuple[int, int, int, int]]:
        """
        查找文本在图像中的位置
        
        :param target_text: 要查找的文本
        :param region: 搜索区域(x1,y1,x2,y2)
        :param partial_match: 是否允许部分匹配
        :return: 文本区域(x1,y1,x2,y2)或None
        """
        try:
            if region:
                x1, y1, x2, y2 = region
                roi = image[y1:y2, x1:x2]
            else:
                roi = image.copy()
                x1, y1 = 0, 0
                
            # 获取OCR数据
            data = pytesseract.image_to_data(
                Image.fromarray(roi),
                output_type=pytesseract.Output.DICT,
                config=f'--oem 3 --psm 6 -l {self.lang}'
            )
            
            # 遍历所有识别到的文本
            for i, text in enumerate(data['text']):
                text = text.strip()
                if not text:
                    continue
                    
                conf = int(data['conf'][i])
                if conf < 60:  # 置信度阈值
                    continue
                    
                # 检查文本匹配
                if (text == target_text) or \
                   (partial_match and target_text.lower() in text.lower()):
                    
                    # 获取文本位置
                    x = data['left'][i] + x1
                    y = data['top'][i] + y1
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    if self.debug:
                        debug_img = image.copy()
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        self._save_debug_image(debug_img, f"text_{target_text}")
                        
                    return (x, y, x+w, y+h)
                    
            return None
            
        except Exception as e:
            self.logger.error(f"查找文本位置失败: {str(e)}")
            return None

    def extract_text_from_region(self, image: np.ndarray,
                               region: Tuple[int, int, int, int],
                               config: Optional[str] = None) -> str:
        """
        从指定区域提取文本
        
        :param region: (x1, y1, x2, y2)
        :return: 识别出的文本
        """
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        if self.debug:
            self._save_debug_image(roi, "ocr_roi")
            
        return self.recognize_text(roi, config=config)

    def recognize_with_confidence(self, image: np.ndarray,
                                min_confidence: int = 70) -> List[Dict[str, Any]]:
        """
        识别文本并返回带置信度的结果
        
        :param min_confidence: 最小置信度阈值(0-100)
        :return: 字典列表,每个包含text, confidence, position
        """
        try:
            # 获取详细的OCR数据
            data = pytesseract.image_to_data(
                Image.fromarray(image),
                output_type=pytesseract.Output.DICT,
                config=f'--oem 3 --psm 6 -l {self.lang}'
            )
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf >= min_confidence:
                    results.append({
                        'text': text,
                        'confidence': conf,
                        'position': (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        )
                    })
                    
            return results
            
        except Exception as e:
            self.logger.error(f"带置信度的OCR识别失败: {str(e)}")
            return []

    def recognize_digit_from_image(self, image: np.ndarray) -> Optional[int]:
        """
        从图像中识别单个数字
        
        :return: 识别出的数字或None
        """
        text = self.recognize_digits(image)
        digits = re.findall(r'\d+', text)
        if digits:
            try:
                return int(digits[0])
            except ValueError:
                return None
        return None

    def recognize_health_value(self, health_bar_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        专门识别血条数值(如: 1250/2000)
        
        :return: (当前值, 最大值) 或 None
        """
        text = self.recognize_text(health_bar_image)
        match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if match:
            try:
                current = int(match.group(1))
                max_val = int(match.group(2))
                return current, max_val
            except ValueError:
                return None
        return None