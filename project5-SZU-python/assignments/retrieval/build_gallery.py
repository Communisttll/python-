import os
import json
import numpy as np
from tqdm import tqdm
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.image_retrieval import ImageRetrievalSystem
from assignments.preprocess_image import resize_short_side

def build_gallery_features(photo_dir=r"assignments\photo", output_dir=r"retrieval\data"):
    """构建图库特征"""
    print("开始构建图库特征...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检索系统
    weights_path = r"assignments\vit-dinov2-base.npz"
    retrieval_system = ImageRetrievalSystem(weights_path)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_paths = []
    
    for filename in os.listdir(photo_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(photo_dir, filename))
    
    print(f"找到 {len(image_paths)} 张图片")
    
    if not image_paths:
        print("未找到图片文件，创建模拟数据...")
        # 创建模拟数据用于演示
        create_mock_gallery_data(output_dir)
        return
    
    # 提取特征
    features = []
    valid_paths = []
    
    for image_path in tqdm(image_paths, desc="提取特征"):
        try:
            feature = retrieval_system.extract_feature(image_path)
            if feature is not None:
                features.append(feature)
                valid_paths.append(image_path)
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
            continue
    
    if features:
        features = np.array(features)
        
        # 保存特征和路径
        np.save(os.path.join(output_dir, "gallery_features.npy"), features)
        
        # 保存路径映射
        path_mapping = {i: path for i, path in enumerate(valid_paths)}
        with open(os.path.join(output_dir, "image_paths.json"), 'w', encoding='utf-8') as f:
            json.dump(path_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"成功提取 {len(features)} 张图片的特征")
        print(f"特征维度: {features.shape}")
        print(f"特征已保存到: {os.path.join(output_dir, 'gallery_features.npy')}")
        print(f"路径映射已保存到: {os.path.join(output_dir, 'image_paths.json')}")
    else:
        print("未能提取任何特征，创建模拟数据...")
        create_mock_gallery_data(output_dir)

def create_mock_gallery_data(output_dir):
    """创建模拟图库数据用于演示"""
    print("创建模拟图库数据...")
    
    # 创建模拟特征
    num_images = 50
    feature_dim = 768
    mock_features = np.random.randn(num_images, feature_dim).astype(np.float32)
    
    # 保存模拟特征
    np.save(os.path.join(output_dir, "gallery_features.npy"), mock_features)
    
    # 创建模拟路径
    mock_paths = {
        i: f"assignments/photo/mock_image_{i+1}.jpg" 
        for i in range(num_images)
    }
    
    with open(os.path.join(output_dir, "image_paths.json"), 'w', encoding='utf-8') as f:
        json.dump(mock_paths, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {num_images} 个模拟图库数据")

if __name__ == "__main__":
    build_gallery_features()