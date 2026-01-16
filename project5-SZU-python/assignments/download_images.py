import pandas as pd
import requests
import os
from tqdm import tqdm
import re

def download_images_from_csv(csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    
    # 过滤掉空的或无效的URL
    df = df.dropna(subset=['image_url'])
    df = df[df['image_url'].str.startswith('http')]

    print(f"Found {len(df)} images to download.")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        image_url = row['image_url']
        caption = row['caption']
        
        # 清理标题，移除特殊字符，确保文件名合法
        cleaned_caption = re.sub(r'[\\/*?:"<>|]', "", str(caption))
        if not cleaned_caption: # 如果清理后为空，则使用索引作为文件名
            cleaned_caption = f"image_{index}"

        # 尝试从URL中获取文件扩展名
        ext = image_url.split('.')[-1].split('?')[0].split('&')[0]
        if ext.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            ext = 'jpg' # 默认使用jpg

        filename = f"{cleaned_caption}.{ext}"
        filepath = os.path.join(output_dir, filename)

        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status() # 检查HTTP请求是否成功

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            # print(f"Error downloading {image_url}: {e}")
            pass # 忽略下载失败的图片
        except Exception as e:
            # print(f"An unexpected error occurred for {image_url}: {e}")
            pass # 忽略其他异常

if __name__ == "__main__":
    csv_file = r"d:\BaiduNetdiskDownload\project5-SZU-python\assignments\data.csv"
    output_directory = r"d:\BaiduNetdiskDownload\project5-SZU-python\assignments\photo"
    download_images_from_csv(csv_file, output_directory)