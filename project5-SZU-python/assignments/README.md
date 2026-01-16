# AI图像检索系统

## 项目概述
本项目旨在构建一个高效、智能的AI图像检索系统，利用先进的深度学习技术，使用户能够通过上传图片来查找本地图库中的相似图片。系统基于Vision Transformer (ViT) 和 DINOv2 模型进行图像特征提取，并通过余弦相似度进行相似性匹配。前端提供直观的用户界面，支持图片拖拽上传、实时预览和结果可视化。

## 功能特性

### 核心功能
- **图像特征提取**: 利用预训练的DINOv2模型（基于Vision Transformer架构）从图像中提取高维特征向量。这些特征能够捕捉图像的深层语义信息，为相似性匹配提供基础。
- **相似度计算**: 采用余弦相似度算法，高效计算查询图片特征与图库图片特征之间的相似度。相似度值介于0到1之间，值越高表示两张图片越相似。
- **本地图库搜索**: 用户上传图片后，系统会在预先构建的本地图像特征库中进行快速搜索，返回与查询图片最相似的K张图片及其相似度得分。
- **图像分析**: 提供上传图片的详细分析报告，包括特征向量的维度、处理时间、本地匹配数量等，帮助用户了解检索过程。

### 前端功能
- **现代化响应式界面**: 采用Bootstrap框架构建，界面简洁、美观，并能自适应不同设备屏幕大小，提供良好的用户体验。
- **图片拖拽上传**: 支持用户通过拖拽方式上传图片，简化操作流程。
- **实时预览**: 用户上传图片后，系统会立即显示图片预览，确保上传正确。
- **可视化结果展示**: 搜索结果以图片列表形式展示，每张图片下方显示其与查询图片的相似度，直观明了。
- **特征可视化**: （待实现/扩展）未来可考虑展示图像特征向量的统计信息或降维可视化，帮助用户理解特征表示。

## 技术架构

### 后端技术
- **Django**: 强大的Python Web框架，用于构建RESTful API和处理Web请求，提供稳定、安全的后端服务。
- **NumPy**: Python科学计算库，用于高效处理图像特征向量的数值计算，如余弦相似度计算。
- **PIL/Pillow**: Python图像处理库，用于图像的加载、预处理（如尺寸调整、格式转换）等操作。
- **DINOv2 (Vision Transformer)**: 最先进的自监督学习视觉模型，能够提取高质量的图像特征，是本系统实现高精度检索的核心。预训练权重 `vit-dinov2-base.npz` 用于加载模型。

### 前端技术
- **HTML5/CSS3**: 构建页面结构和样式。
- **Bootstrap**: 流行的前端UI框架，提供丰富的组件和响应式布局，加速开发并确保界面美观。
- **JavaScript/jQuery**: 实现前端交互逻辑，如图片上传、实时预览、AJAX请求等。
- **AJAX**: 用于前端与后端API进行异步数据交互，实现无刷新页面更新，提升用户体验。

## 项目结构
```
image_retrieval_system/
├── assignments/                    # 核心算法模块及预训练模型
│   ├── dinov2_numpy.py            # DINOv2模型在NumPy上的实现
│   ├── preprocess_image.py          # 图像预处理工具函数
│   └── vit-dinov2-base.npz        # DINOv2基础模型的预训练权重文件
├── image_retrieval_system/         # Django项目主配置目录
│   ├── settings.py                # Django项目设置
│   ├── urls.py                    # Django项目URL路由
│   └── wsgi.py                    # WSGI入口
├── retrieval/                      # Django应用：图像检索模块
│   ├── image_retrieval.py         # 图像检索核心逻辑，包括特征提取和相似度计算
│   ├── build_gallery.py           # 用于构建本地图库特征（.npy）和路径（.json）的脚本
│   ├── templates/retrieval/        # 存放Django模板文件（HTML）
│   │   ├── upload.html            # 图片上传页面
│   │   └── gallery.html           # 图库展示页面
│   ├── static/retrieval/           # 存放静态文件（CSS, JS, 图片等）
│   └── views.py                    # 处理Web请求和API逻辑的视图函数
├── media/                          # 用户上传图片和图库图片存储目录
│   ├── gallery/                   # 同步后的图库图片
│   └── uploads/                   # 用户临时上传图片
├── manage.py                       # Django项目管理脚本，用于运行服务器、数据库迁移等
├── gallery_features.npy            # 存储本地图库所有图片的特征向量
├── image_paths.json                # 存储本地图库所有图片的路径信息
├── db.sqlite3                      # Django默认的SQLite数据库文件
└── README.md                       # 项目说明文件
```

## 快速开始

### 1. 环境准备
确保您的系统已安装Python 3.8+。
```bash
# 克隆项目仓库
# git clone <your-repo-url>
# cd image_retrieval_system

# 创建并激活虚拟环境 (推荐)
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装所有必要的Python依赖
pip install django pillow numpy requests tqdm pandas
```

### 2. 构建图库特征
在运行服务器之前，您需要构建本地图库的图像特征。这将遍历 `assignments/photo` 目录下的所有图片，提取特征并保存到 `retrieval/data/gallery_features.npy` 和 `retrieval/data/image_paths.json`。
```bash
python retrieval/build_gallery.py
```
**注意**: 首次运行可能需要一些时间，具体取决于图库图片数量和您的硬件性能。

### 3. 启动Django服务器
```bash
python manage.py runserver 0.0.0.0:8000
```
服务器启动后，您将在终端看到类似 `Starting development server at http://127.0.0.1:8000/` 的输出。

### 4. 访问应用
打开您的Web浏览器，访问以下URL：
- **主页**: `http://localhost:8000/`
- **图片上传与检索**: `http://localhost:8000/upload/`
- **图库展示**: `http://localhost:8000/gallery/`

## 使用说明

### 图像上传与检索
1. 访问 `http://localhost:8000/upload/` 页面。
2. 您可以通过点击区域选择图片，或直接将图片文件拖拽到指定区域。
3. 上传图片后，系统会自动进行特征提取和相似图片检索。
4. 页面将实时显示检索结果，包括本地图库中最相似的图片及其相似度得分。

### 结果解读
- **相似度**: 每个结果图片下方会显示一个0到1之间的相似度值。值越接近1，表示该图片与您上传的查询图片在视觉上越相似。
- **本地结果**: 这些图片是从您本地图库中找到的相似图片。
- **特征分析**: 在结果区域，您还可以看到关于上传图片的一些分析数据，例如特征维度和处理时间。

## 核心算法实现细节

### 特征提取 (`retrieval/image_retrieval.py`)
```python
def extract_features(self, image):
    # 图像预处理：将输入图像调整为DINOv2模型所需的尺寸（例如224x224），并进行归一化处理。
    processed_image = resize_short_side(image, 224)
    # 使用加载的DINOv2模型对预处理后的图像进行前向传播，提取其特征向量。
    features = dinov2_model(processed_image)
    return features
```
此方法负责将原始图像转换为DINOv2模型可理解的数值表示。

### 相似度计算 (`retrieval/image_retrieval.py`)
```python
def calculate_similarity(self, query_features, gallery_features):
    # 计算查询特征向量与图库中所有特征向量的余弦相似度。
    # 余弦相似度通过计算两个向量夹角的余弦值来衡量它们的相似性，与向量的长度无关。
    similarities = np.dot(gallery_features, query_features) / (
        np.linalg.norm(gallery_features, axis=1) * np.linalg.norm(query_features)
    )
    return similarities
```
此方法是检索过程的核心，它量化了查询图片与图库中每张图片的相似程度。

## 性能指标
- **特征维度**: 768维。DINOv2模型提取的特征向量具有768个浮点数，能够提供丰富的图像语义信息。
- **处理速度**: 单张图片特征提取和检索通常在0.1秒左右完成，具体取决于图片大小和服务器性能。
- **检索精度**: 基于DINOv2高质量的自监督学习特征，系统能够实现较高的检索精度，有效识别视觉相似的图片。
- **并发支持**: Django框架和Python的异步特性使得系统能够支持多用户同时进行图片上传和检索操作。

## 扩展功能与未来展望

### 1. 数据库支持
- **图像元数据存储**: 将图片路径、特征文件路径、上传时间、标签等元数据存储到PostgreSQL或MySQL等关系型数据库中，便于管理和查询。
- **用户上传历史**: 记录用户的上传和搜索历史，提供个性化服务。
- **搜索记录与分析**: 存储搜索日志，用于分析用户行为和优化检索算法。

### 2. 高级检索功能
- **多特征融合**: 结合DINOv2特征与其他视觉特征（如颜色、纹理、形状）进行融合，进一步提升检索精度。
- **语义搜索**: 允许用户通过文本描述进行图片搜索，实现跨模态检索。
- **实时索引更新**: 实现图库的动态更新，当有新图片添加到图库时，能够自动提取特征并更新索引，无需手动重建。
- **用户个性化**: 根据用户历史行为和偏好，调整搜索结果的排序或推荐相关内容。

## 注意事项
- **生产环境部署**: 建议在生产环境中使用Gunicorn、uWSGI等专业的WSGI服务器来部署Django应用，以获得更好的性能和稳定性。
- **图像版权**: 在使用和分发图片时，请务必注意图像的版权和使用权限。
- **DINOv2模型**: `vit-dinov2-base.npz` 文件较大，请确保下载完整。
