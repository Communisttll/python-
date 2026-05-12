"""内容解析器 - PDF/网页/视频内容提取"""
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import fitz  # PyMuPDF


class ContentParser:
    """统一的内容解析接口"""

    @staticmethod
    def parse_pdf(file_path: str) -> dict:
        """解析PDF文件，返回文本块列表"""
        doc = fitz.open(file_path)
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                # 按段落分割
                paragraphs = re.split(r'\n\n+', text)
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if para and len(para) > 20:  # 过滤短内容
                        chunks.append({
                            "content": para,
                            "page": page_num + 1,
                            "paragraph": i,
                            "source": Path(file_path).name
                        })

        total_pages = len(doc)
        doc.close()
        return {
            "type": "pdf",
            "title": Path(file_path).stem,
            "chunks": chunks,
            "total_pages": total_pages
        }

    @staticmethod
    def parse_web(url: str, html_content: str = None) -> dict:
        """解析网页内容"""
        if html_content is None:
            response = requests.get(url, timeout=10)
            html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        # 移除script和style标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 获取标题
        title = soup.title.string if soup.title else "Untitled"

        # 获取正文段落
        paragraphs = soup.find_all("p")
        chunks = []

        for i, p in enumerate(paragraphs):
            text = p.get_text(strip=True)
            if text and len(text) > 50:
                chunks.append({
                    "content": text,
                    "paragraph": i,
                    "source": url
                })

        return {
            "type": "web",
            "title": title,
            "url": url,
            "chunks": chunks
        }

    @staticmethod
    def parse_srt(file_path: str) -> dict:
        """解析SRT字幕文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = []
        blocks = re.split(r'\n\n+', content)

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                text_lines = []
                for line in lines:
                    if re.match(r'^\d+$', line):
                        continue
                    if re.match(r'\d{2}:\d{2}:\d{2}', line):
                        continue
                    text_lines.append(line)

                text = ' '.join(text_lines).strip()
                if text and len(text) > 20:
                    chunks.append({
                        "content": text,
                        "source": Path(file_path).name
                    })

        return {
            "type": "video",
            "title": Path(file_path).stem,
            "chunks": chunks
        }


if __name__ == "__main__":
    print("ContentParser 模块已加载")
