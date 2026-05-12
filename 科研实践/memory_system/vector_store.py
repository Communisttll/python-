"""向量存储模块 - 基于ChromaDB"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import hashlib
import numpy as np


class VectorStore:
    """ChromaDB向量存储封装"""

    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 记忆集合
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"description": "AI-Native Memory System"}
        )

    def add_chunks(self, memory_id: str, chunks: list, metadata: dict):
        """添加文本块到向量库"""
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{memory_id}_{i}"
            ids.append(chunk_id)
            documents.append(chunk["content"])

            embedding = self._simple_embedding(chunk["content"])
            embeddings.append(embedding)

            meta = {
                "memory_id": memory_id,
                "source_type": metadata.get("type", "unknown"),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", ""),
                "paragraph": chunk.get("paragraph", i)
            }
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def _simple_embedding(self, text: str) -> list:
        """简化的embedding实现（基于词频）
        实际应该使用专门的embedding模型如 nomic-embed-text"""
        hash_val = hashlib.md5(text.encode()).digest()
        vec = np.frombuffer(hash_val, dtype=np.float32)
        while len(vec) < 768:
            vec = np.concatenate([vec, vec])
        vec = vec[:768]
        # L2归一化，使向量模长为1，内积等价于余弦相似度
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def search(self, query: str, n_results: int = 5) -> dict:
        """语义检索"""
        query_embedding = self._simple_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def get_memory_chunks(self, memory_id: str) -> list:
        """获取指定记忆的所有chunk"""
        results = self.collection.get(
            where={"memory_id": memory_id}
        )
        return results

    def delete_memory(self, memory_id: str):
        """删除指定记忆"""
        chunks = self.get_memory_chunks(memory_id)
        if chunks["ids"]:
            self.collection.delete(ids=chunks["ids"])

    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name
        }


if __name__ == "__main__":
    print("VectorStore 模块已加载")
