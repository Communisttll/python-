import numpy as np
from PIL import Image

def center_crop(img_path, crop_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))  # PIL Image, size (224, 224)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)

# ************* ToDo, resize short side *************
def resize_short_side(img_path, target_size=224):
    # Step 1: load image
    image = Image.open(img_path).convert("RGB")

    # Step 2: resize so that the shorter side == target_size
    # and more, ensure both sides are multiples of patch size, e.g., 14
    w, h = image.size
    
    # 计算缩放比例：确保短边等于target_size
    if h < w:  # 高是短边
        scale = target_size / h
    else:  # 宽是短边
        scale = target_size / w
    
    # 计算缩放后的初步尺寸
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 校准尺寸为14的倍数
    # 确保高度和宽度都能被14整除
    new_h = max(14, (new_h // 14) * 14)  # 最小为14，避免为0
    new_w = max(14, (new_w // 14) * 14)
    
    # 执行缩放，使用LANCZOS滤波器获得更好的图像质量
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Step 3: to_numpy
    image = np.array(image).astype(np.float32) / 255.0  # (H, W, C)

    # Step 4: norm
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # (H, W, C)
    image = image.transpose(2, 0, 1) # (C, H, W)
    return image[None] # (1, C, H, W)