import numpy as np
import os
import json
from PIL import Image
from typing import List, Tuple, Dict
from io import BytesIO
import re
from urllib.parse import quote
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'assignments'))

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

class ImageRetrievalSystem:
    def __init__(self, weights_path: str, gallery_dir: str = None):
        """
        初始化图像检索系统
        
        Args:
            weights_path: 预训练权重文件路径
            gallery_dir: 图库目录路径
        """
        print(f"Loading weights from {weights_path}")
        self.weights = np.load(weights_path)
        self.vit = Dinov2Numpy(self.weights)
        
        self.gallery_dir = gallery_dir
        self.gallery_features = None
        self.image_paths = []
        
        if gallery_dir and os.path.exists(gallery_dir):
            self.load_gallery_features()
    
    def extract_feature(self, image_path: str) -> np.ndarray:
        """从图像文件提取特征向量"""
        try:
            pixel_values = resize_short_side(image_path)
            feature = self.vit(pixel_values)
            return feature.flatten()
        except Exception as e:
            print(f"Error extracting feature from {image_path}: {e}")
            return None
    
    def extract_feature_from_pil(self, image: Image.Image) -> np.ndarray:
        """从PIL图像对象提取特征向量"""
        try:
            # 保存为临时文件
            temp_path = "temp_upload.jpg"
            image.save(temp_path)
            feature = self.extract_feature(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return feature
        except Exception as e:
            print(f"Error extracting feature from PIL image: {e}")
            return None
    
    def build_gallery_features(self, gallery_dir: str, save_path: str = "gallery_features.npy"):
        """构建图库特征数据库"""
        self.gallery_dir = gallery_dir
        image_features = []
        self.image_paths = []
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
        
        for filename in os.listdir(gallery_dir):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(gallery_dir, filename)
                print(f"Processing {image_path}")
                
                feature = self.extract_feature(image_path)
                if feature is not None:
                    image_features.append(feature)
                    self.image_paths.append(image_path)
        
        if image_features:
            self.gallery_features = np.array(image_features)
            np.save(save_path, self.gallery_features)
            
            # 保存路径映射
            with open("image_paths.json", "w") as f:
                json.dump(self.image_paths, f)
            
            print(f"Gallery features built: {len(image_features)} images")
        else:
            print("No valid images found in gallery directory")
    
    def load_gallery_features(self, features_path: str = "gallery_features.npy", 
                             paths_path: str = "image_paths.json"):
        """加载图库特征数据库"""
        if os.path.exists(features_path) and os.path.exists(paths_path):
            self.gallery_features = np.load(features_path)
            with open(paths_path, "r") as f:
                self.image_paths = json.load(f)
            print(f"Gallery features loaded: {len(self.image_paths)} images")
        else:
            print("Gallery features not found, building from directory...")
            if self.gallery_dir:
                self.build_gallery_features(self.gallery_dir)
    
    def cosine_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    
    def search_similar_images(self, query_feature: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """在图库中搜索相似图像"""
        if self.gallery_features is None or len(self.gallery_features) == 0:
            return []
        
        similarities = []
        for i, gallery_feature in enumerate(self.gallery_features):
            similarity = self.cosine_similarity(query_feature, gallery_feature)
            # 根据image_paths的数据结构获取路径
            if isinstance(self.image_paths, dict):
                # 如果image_paths是字典格式（键为数字索引）
                path = self.image_paths.get(str(i), f"unknown_path_{i}")
            elif isinstance(self.image_paths, list):
                # 如果image_paths是列表格式
                path = self.image_paths[i] if i < len(self.image_paths) else f"unknown_path_{i}"
            else:
                # 其他情况，使用索引作为后备
                path = f"gallery_image_{i}"
            similarities.append((path, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    

    
    def _extract_search_terms(self, caption: str) -> List[str]:
        """从图像描述中提取搜索关键词"""
        # 简单的关键词提取：移除停用词，提取有意义的词
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        # 清理和分词
        words = re.findall(r'\b\w+\b', caption.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:3]  # 返回前3个关键词
    
    def retrieve_similar_images(self, query_image_path: str = None, query_pil_image: Image.Image = None, 
                               top_k: int = 10) -> Dict:
        """
        完整的图像检索功能
        
        Args:
            query_image_path: 查询图像路径
            query_pil_image: PIL图像对象（二选一）
            top_k: 返回结果数量
            
        Returns:
            包含本地搜索结果的字典
        """
        results = {
            "local_results": [],
            "query_feature": None,
            "analysis": {}
        }
        
        # 提取查询图像特征
        if query_pil_image:
            query_feature = self.extract_feature_from_pil(query_pil_image)
        elif query_image_path:
            query_feature = self.extract_feature(query_image_path)
        else:
            return results
        
        if query_feature is None:
            return results
        
        results["query_feature"] = query_feature.tolist()
        
        # 本地图库搜索
        if self.gallery_features is not None:
            local_results = self.search_similar_images(query_feature, top_k)
            results["local_results"] = [
                {"path": path, "similarity": float(sim)} 
                for path, sim in local_results
            ]
        
        # 分析查询图像特征
        results["analysis"] = {
            "feature_dimension": len(query_feature),
            "feature_norm": float(np.linalg.norm(query_feature)),
            "feature_mean": float(np.mean(query_feature)),
            "feature_std": float(np.std(query_feature))
        }
        
        return results
    
    def get_image_analysis(self, image_path):
        """获取图像的详细分析信息"""
        try:
            # 提取特征
            features = self.extract_feature(image_path)
            
            # 获取图像基本信息
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                
            # 计算特征统计信息
            feature_stats = {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features)),
                'sparsity': float(np.sum(np.abs(features) < 0.1) / len(features))
            }
            
            return {
                'image_info': {
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'size': os.path.getsize(image_path)
                },
                'feature_stats': feature_stats,
                'feature_vector': features.tolist()
            }
            
        except Exception as e:
            print(f"图像分析失败: {e}")
            return None