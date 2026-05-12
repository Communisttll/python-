"""AI-Native 记忆系统核心"""
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import config
from parser import ContentParser
from vector_store import VectorStore
from llm_router import LLMRouter, MockLLMRouter


class Memory:
    """单条记忆"""

    def __init__(self, memory_id: str, memory_type: str, title: str,
                 source_path: str, chunks: list):
        self.id = memory_id
        self.type = memory_type
        self.title = title
        self.source_path = source_path
        self.chunks = chunks
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "source_path": self.source_path,
            "chunk_count": len(self.chunks),
            "created_at": self.created_at
        }


class MemorySystem:
    """AI-Native 记忆系统"""

    def __init__(self, use_mock_llm: bool = True):
        """初始化记忆系统

        Args:
            use_mock_llm: 是否使用Mock LLM（演示用，不依赖真实API）
        """
        self.vector_store = VectorStore(str(config.chroma_path))
        self.use_mock_llm = use_mock_llm

        if use_mock_llm:
            self.llm = MockLLMRouter()
            print("[MemorySystem] 使用Mock LLM进行演示")
        else:
            self.llm = LLMRouter(
                provider=config.llm_provider,
                model=config.llm_model
            )
            print(f"[MemorySystem] 使用真实LLM: {config.llm_provider}/{config.llm_model}")

    def add_memory(self, file_path: str, memory_type: str = None) -> Memory:
        """添加记忆"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 自动检测类型
        if memory_type is None:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                memory_type = "pdf"
            elif suffix in [".html", ".htm"]:
                memory_type = "web"
            elif suffix in [".srt", ".vtt", ".txt"]:
                memory_type = "video"
            else:
                memory_type = "unknown"

        # 解析内容
        if memory_type == "pdf":
            parsed = ContentParser.parse_pdf(str(path))
        elif memory_type == "web":
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            parsed = ContentParser.parse_web(str(path), html)
        elif memory_type == "video":
            parsed = ContentParser.parse_srt(str(path))
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        # 创建记忆
        memory_id = str(uuid.uuid4())[:8]
        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            title=parsed["title"],
            source_path=str(path),
            chunks=parsed["chunks"]
        )

        # 存储到向量库
        self.vector_store.add_chunks(memory_id, parsed["chunks"], {
            "type": memory_type,
            "title": parsed["title"]
        })

        print(f"[MemorySystem] 添加记忆: {memory.title} ({len(parsed['chunks'])} chunks)")
        return memory

    def search(self, query: str, n_results: int = 5) -> dict:
        """语义检索记忆"""
        results = self.vector_store.search(query, n_results)

        formatted = {
            "query": query,
            "results": []
        }

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]

            formatted["results"].append({
                "chunk_id": chunk_id,
                "content": document,
                "similarity": 1 - distance,
                "source": metadata.get("source", ""),
                "page": metadata.get("page", ""),
                "memory_id": metadata.get("memory_id", "")
            })

        return formatted

    def generate_draft(self, query: str, n_chunks: int = 5) -> dict:
        """基于记忆生成初稿"""
        search_results = self.search(query, n_results=n_chunks)

        if not search_results["results"]:
            return {
                "draft": "未找到相关记忆片段，请先添加相关学习资料。",
                "citations": [],
                "search_results": []
            }

        chunks = [r["content"] for r in search_results["results"]]
        source_refs = []
        for i, r in enumerate(search_results["results"]):
            ref = f"{r['source']}"
            if r.get("page"):
                ref += f", p.{r['page']}"
            source_refs.append(ref)

        draft = self.llm.generate_draft(query, chunks, source_refs)

        return {
            "draft": draft,
            "citations": source_refs,
            "search_results": search_results["results"]
        }

    def stats(self) -> dict:
        """获取系统统计"""
        return self.vector_store.stats()

    def list_memories(self) -> list:
        """列出所有记忆"""
        all_data = self.vector_store.collection.get()
        memories = {}

        for meta in all_data["metadatas"]:
            mid = meta["memory_id"]
            if mid not in memories:
                memories[mid] = {
                    "id": mid,
                    "type": meta["source_type"],
                    "source": meta["source"],
                    "chunk_count": 0
                }
            memories[mid]["chunk_count"] += 1

        return list(memories.values())


if __name__ == "__main__":
    print("MemorySystem 核心模块已加载")
