import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
import logging
from pathlib import Path

class ImageProcessor:
    def __init__(self, default_threshold: float = 0.8, debug: bool = False):
        """
        初始化图像处理器
        
        :param default_threshold: 默认图像匹配阈值(0-1)
        :param debug: 是否启用调试模式(保存中间图像)
        """
        self.logger = logging.getLogger(__name__)
        self.default_threshold = default_threshold
        self.debug = debug
        self.debug_dir = Path("debug_images")
        
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)

    def _save_debug_image(self, image: np.ndarray, name: str):
        """保存调试图像"""
        if self.debug:
            timestamp = int(time.time() * 1000)
            path = self.debug_dir / f"{timestamp}_{name}.png"
            cv2.imwrite(str(path), image)
            self.logger.debug(f"保存调试图像: {path}")

    def find_template(self, screenshot: np.ndarray, 
                     template: np.ndarray, 
                     threshold: float = None,
                     method: int = cv2.TM_CCOEFF_NORMED) -> Optional[Tuple[int, int]]:
        """
        在屏幕截图中查找模板图像
        
        :param screenshot: 屏幕截图(BGR格式)
        :param template: 模板图像(BGR格式)
        :param threshold: 匹配阈值(0-1)
        :param method: 匹配方法(默认cv2.TM_CCOEFF_NORMED)
        :return: (x, y)坐标或None
        """
        if screenshot is None or template is None:
            self.logger.error("无效的输入图像")
            return None
            
        if threshold is None:
            threshold = self.default_threshold
            
        try:
            # 转换为灰度图像(提高性能)
            screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # 执行模板匹配
            res = cv2.matchTemplate(screen_gray, template_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # 根据匹配方法确定最佳匹配位置
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                match_val = 1 - min_val
            else:
                top_left = max_loc
                match_val = max_val
                
            # 检查匹配分数
            if match_val >= threshold:
                # 返回中心坐标
                h, w = template.shape[:2]
                center_x = top_left[0] + w // 2
                center_y = top_left[1] + h // 2
                
                if self.debug:
                    # 绘制匹配区域
                    debug_img = screenshot.copy()
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(debug_img, f"{match_val:.2f}", 
                               (top_left[0], top_left[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self._save_debug_image(debug_img, "template_match")
                
                return center_x, center_y
                
            self.logger.debug(f"模板匹配分数不足: {match_val:.2f} < {threshold:.2f}")
            return None
            
        except Exception as e:
            self.logger.error(f"模板匹配失败: {str(e)}")
            return None

    def find_all_templates(self, screenshot: np.ndarray, 
                          template: np.ndarray, 
                          threshold: float = None,
                          method: int = cv2.TM_CCOEFF_NORMED) -> List[Tuple[int, int]]:
        """
        查找所有匹配的模板位置
        
        :return: 匹配位置列表[(x1,y1), (x2,y2), ...]
        """
        if screenshot is None or template is None:
            return []
            
        if threshold is None:
            threshold = self.default_threshold
            
        try:
            screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = template.shape[:2]
            
            res = cv2.matchTemplate(screen_gray, template_gray, method)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                res = 1 - res
                loc = np.where(res >= threshold)
            else:
                loc = np.where(res >= threshold)
                
            positions = []
            for pt in zip(*loc[::-1]):  # 交换x,y
                center_x = pt[0] + w // 2
                center_y = pt[1] + h // 2
                positions.append((center_x, center_y))
                
            # 非极大值抑制，去除重叠的匹配
            positions = self._non_max_suppression(positions, w, h)
            
            if self.debug and positions:
                debug_img = screenshot.copy()
                for x, y in positions:
                    top_left = (x - w//2, y - h//2)
                    bottom_right = (x + w//2, y + h//2)
                    cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
                self._save_debug_image(debug_img, "multi_template_match")
                
            return positions
            
        except Exception as e:
            self.logger.error(f"多模板匹配失败: {str(e)}")
            return []

    def _non_max_suppression(self, positions: List[Tuple[int, int]], 
                           width: int, height: int,
                           overlap_thresh: float = 0.5) -> List[Tuple[int, int]]:
        """
        非极大值抑制，去除重叠的匹配
        
        :param positions: 匹配位置列表
        :param width: 模板宽度
        :param height: 模板高度
        :param overlap_thresh: 重叠阈值
        :return: 过滤后的位置列表
        """
        if len(positions) == 0:
            return []
            
        # 转换为(x1, y1, x2, y2)格式
        boxes = []
        for (x, y) in positions:
            boxes.append([x - width//2, y - height//2, 
                         x + width//2, y + height//2])
            
        boxes = np.array(boxes)
        pick = []
        
        # 计算每个box的面积
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        
        # 按y坐标排序
        idxs = np.argsort(boxes[:, 1])
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # 计算与其他box的交集
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[:last], 3])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            # 删除重叠超过阈值的box
            idxs = np.delete(idxs, np.concatenate(([last], 
                         np.where(overlap > overlap_thresh)[0])))
                         
        return [positions[i] for i in pick]

    def compare_images(self, img1: np.ndarray, img2: np.ndarray, 
                      threshold: float = 0.95) -> bool:
        """
        比较两张图像是否相似
        
        :param threshold: 相似度阈值(0-1)
        :return: 是否相似
        """
        if img1.shape != img2.shape:
            return False
            
        try:
            # 计算结构相似性指数
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 使用SSIM比较
            score = self._ssim(gray1, gray2)
            self.logger.debug(f"图像相似度: {score:.2f}")
            
            if self.debug:
                diff = cv2.absdiff(gray1, gray2)
                self._save_debug_image(diff, "image_diff")
                
            return score >= threshold
            
        except Exception as e:
            self.logger.error(f"图像比较失败: {str(e)}")
            return False

    def _ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性指数(简化版)"""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def find_color(self, screenshot: np.ndarray, 
                  color: Tuple[int, int, int], 
                  tolerance: int = 10) -> Optional[Tuple[int, int]]:
        """
        查找指定颜色的位置
        
        :param color: 目标颜色(BGR格式)
        :param tolerance: 颜色容差
        :return: 第一个匹配点的(x,y)坐标
        """
        try:
            lower = np.array([max(0, c - tolerance) for c in color])
            upper = np.array([min(255, c + tolerance) for c in color])
            
            mask = cv2.inRange(screenshot, lower, upper)
            
            if self.debug:
                self._save_debug_image(mask, "color_mask")
                
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 返回最大轮廓的中心点
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY
                    
            return None
            
        except Exception as e:
            self.logger.error(f"颜色查找失败: {str(e)}")
            return None

    def extract_text_region(self, screenshot: np.ndarray, 
                          region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        提取指定区域的图像(用于OCR)
        
        :param region: (x1, y1, x2, y2)
        :return: 区域图像
        """
        x1, y1, x2, y2 = region
        return screenshot[y1:y2, x1:x2]

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        为OCR预处理图像:
        1. 转为灰度
        2. 二值化
        3. 降噪
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        if self.debug:
            self._save_debug_image(denoised, "ocr_preprocessed")
            
        return denoised

    def get_dominant_color(self, image: np.ndarray, top_n: int = 1) -> List[Tuple[int, int, int]]:
        """
        获取图像中的主要颜色
        
        :param top_n: 返回前N种主要颜色
        :return: 主要颜色列表(BGR格式)
        """
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, top_n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        dominant_colors = [tuple(color) for color in centers]
        
        return dominant_colors