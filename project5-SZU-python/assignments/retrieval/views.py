from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import json
import time
from datetime import datetime

# 导入图像检索系统
from .image_retrieval import ImageRetrievalSystem

def index(request):
    """首页视图"""
    return render(request, 'retrieval/index.html')

def sync_gallery_to_media():
    """同步图库图片到媒体目录以便通过URL访问"""
    gallery_dir = r'D:\BaiduNetdiskDownload\project5-SZU-python\assignments\photo'
    media_gallery_dir = r'D:\BaiduNetdiskDownload\project5-SZU-python\media\gallery'
    
    # 确保媒体图库目录存在
    os.makedirs(media_gallery_dir, exist_ok=True)
    
    # 复制所有图片文件
    if os.path.exists(gallery_dir):
        for filename in os.listdir(gallery_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                src_path = os.path.join(gallery_dir, filename)
                dst_path = os.path.join(media_gallery_dir, filename)
                try:
                    # 如果目标文件不存在或源文件较新，则复制
                    if not os.path.exists(dst_path) or os.path.getmtime(src_path) > os.path.getmtime(dst_path):
                        import shutil
                        shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"复制文件失败 {filename}: {e}")

def upload(request):
    """上传和检索视图"""
    # 同步图库到媒体目录
    sync_gallery_to_media()
    
    if request.method == 'POST':
        try:
            # 获取上传的图片
            if 'image' not in request.FILES:
                return JsonResponse({'success': False, 'error': '未找到上传的图片'})
            
            uploaded_file = request.FILES['image']
            
            # 保存上传的图片
            fs = FileSystemStorage()
            filename = fs.save(f"uploads/{uploaded_file.name}", uploaded_file)
            uploaded_file_path = fs.path(filename)
            
            # 初始化检索系统
            weights_path = r'D:\BaiduNetdiskDownload\project5-SZU-python\assignments\vit-dinov2-base.npz'
            gallery_dir = r'D:\BaiduNetdiskDownload\project5-SZU-python\assignments\photo'
            
            retrieval_system = ImageRetrievalSystem(
                weights_path=weights_path,
                gallery_dir=gallery_dir
            )
            
            # 明确指定加载正确的图库特征文件
            retrieval_system.load_gallery_features(
                'retrieval/data/gallery_features.npy',
                'retrieval/data/image_paths.json'
            )
            
            # 开始计时
            start_time = time.time()
            
            # 执行检索
            results = retrieval_system.retrieve_similar_images(
                uploaded_file_path,
                top_k=10
            )
            
            processing_time = round(time.time() - start_time, 2)
            
            # 构建响应数据
            response_data = {
                'success': True,
                'analysis': {
                    'feature_dim': 768,
                    'processing_time': processing_time,
                    'local_matches': len(results.get('local_results', []))
                },
                'features': results.get('query_feature', [])[:50],  # 前50维特征
                'local_results': []
            }
            
            # 处理本地结果
            for result in results.get('local_results', []):
                # 获取相对路径用于URL
                image_path = result['path']
                image_filename = os.path.basename(image_path)
                
                # 对于本地图库中的图片，使用正确的URL路径
                image_url = f"/media/gallery/{image_filename}"
                
                response_data['local_results'].append({
                    'image_url': image_url,
                    'caption': result.get('caption', f'本地图库图片 - {image_filename}'),
                    'similarity': float(result['similarity']),
                    'source': '本地图库',
                    'original_path': image_path,
                    'filename': image_filename
                })
            
            # 清理上传的文件
            if os.path.exists(uploaded_file_path):
                os.remove(uploaded_file_path)
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    # GET请求显示上传页面
    return render(request, 'retrieval/upload.html')

def gallery(request):
    """图库视图"""
    gallery_dir = r'D:\BaiduNetdiskDownload\project5-SZU-python\assignments\photo'
    images = []
    
    if os.path.exists(gallery_dir):
        for filename in os.listdir(gallery_dir)[:20]:  # 只显示前20张
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                images.append({
                    'filename': filename,
                    'path': f"/media/gallery/{filename}"
                })
    
    return render(request, 'retrieval/gallery.html', {'images': images})

@csrf_exempt
def api_search(request):
    """API接口：搜索相似图像"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_url = data.get('image_url')
            top_k = data.get('top_k', 5)
            
            if not image_url:
                return JsonResponse({'success': False, 'error': '缺少图片URL'})
            
            # 这里可以实现基于URL的图像检索
            # 为了演示，返回模拟数据
            mock_results = {
                'success': True,
                'results': [
                    {
                        'image_url': 'https://via.placeholder.com/300x200?text=Similar+Image+1',
                        'caption': '相似图像1',
                        'similarity': 0.95
                    },
                    {
                        'image_url': 'https://via.placeholder.com/300x200?text=Similar+Image+2',
                        'caption': '相似图像2',
                        'similarity': 0.88
                    }
                ]
            }
            
            return JsonResponse(mock_results)
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': '只支持POST请求'})

@csrf_exempt
def api_analyze(request):
    """API接口：分析图像特征"""
    if request.method == 'POST':
        try:
            # 这里可以实现图像特征分析
            # 返回模拟数据
            analysis_data = {
                'success': True,
                'features': {
                    'feature_vector': [0.1, 0.2, 0.3, 0.4, 0.5] * 10,  # 50维特征
                    'dominant_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    'texture_features': {
                        'contrast': 0.75,
                        'correlation': 0.82,
                        'energy': 0.63,
                        'homogeneity': 0.91
                    }
                }
            }
            
            return JsonResponse(analysis_data)
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': '只支持POST请求'})