#!/usr/bin/env python3
"""AI-Native 记忆系统演示脚本

演示完整流程：
1. 添加记忆（PDF/网页/视频）
2. 语义检索
3. 生成带溯源初稿
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "memory_system"))

from memory_system import MemorySystem


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def demo():
    """演示完整流程"""
    print_separator("AI-Native 个人记忆系统演示")
    print("初始化记忆系统...")

    # 初始化（使用Mock LLM进行演示）
    ms = MemorySystem(use_mock_llm=True)

    # 演示数据路径
    base_path = Path(__file__).parent / "memory" / "raw"
    pdf_path = base_path / "papers" / "transformer.pdf"
    web_path = base_path / "web" / "transformer-wiki.html"
    video_path = base_path / "video" / "transformer-intro.srt"

    # ==================== 1. 添加记忆 ====================
    print_separator("Step 1: 添加记忆")

    memories = []

    # 添加PDF记忆
    if pdf_path.exists():
        print(f"\n[1/3] 添加PDF论文: {pdf_path.name}")
        m = ms.add_memory(str(pdf_path))
        memories.append(m)
    else:
        print(f"\n[1/3] PDF文件不存在: {pdf_path}")

    # 添加网页记忆
    if web_path.exists():
        print(f"\n[2/3] 添加网页内容: {web_path.name}")
        m = ms.add_memory(str(web_path))
        memories.append(m)
    else:
        print(f"\n[2/3] 网页文件不存在: {web_path}")

    # 添加视频字幕记忆
    if video_path.exists():
        print(f"\n[3/3] 添加视频字幕: {video_path.name}")
        m = ms.add_memory(str(video_path))
        memories.append(m)
    else:
        print(f"\n[3/3] 字幕文件不存在: {video_path}")

    # 显示统计
    print_separator("记忆统计")
    stats = ms.stats()
    print(f"  总记忆块数: {stats['total_chunks']}")
    print(f"  集合名称: {stats['collection_name']}")

    # 列出所有记忆
    print("\n记忆列表:")
    for m in ms.list_memories():
        print(f"  - [{m['type']}] {m['source']} ({m['chunk_count']} chunks)")

    # ==================== 2. 语义检索 ====================
    print_separator("Step 2: 语义检索")

    queries = [
        "Transformer架构的核心优势和主要应用领域",
        "BERT和GPT的区别",
        "2017年Google发表的论文内容"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[查询 {i}] {query}")
        results = ms.search(query, n_results=3)
        print(f"  找到 {len(results['results'])} 条相关记忆:")
        for j, r in enumerate(results['results'], 1):
            # 截断显示
            content = r['content'][:80] + "..." if len(r['content']) > 80 else r['content']
            print(f"    {j}. [相似度: {r['similarity']:.3f}] {content}")
            print(f"       来源: {r['source']}")

    # ==================== 3. 生成初稿 ====================
    print_separator("Step 3: 基于记忆生成初稿")

    research_topic = "请帮我写一段关于Transformer架构的研究背景，包括其核心优势和主要应用"

    print(f"\n[研究主题]\n{research_topic}\n")

    result = ms.generate_draft(research_topic, n_chunks=5)

    print("[生成的初稿]")
    print("-" * 60)
    print(result["draft"])
    print("-" * 60)

    print("\n[溯源信息]")
    for i, citation in enumerate(result["citations"], 1):
        print(f"  {i}. {citation}")

    # ==================== 4. 可靠性评估 ====================
    print_separator("Step 4: 可靠性评估")

    # 计算评估指标
    total_citations = len(result["citations"])
    draft_lines = result["draft"].count("\n")
    citation_mentions = result["draft"].count("【来源")

    accuracy_score = 100  # Mock模式下假设100%
    traceability_score = (citation_mentions / max(draft_lines, 1)) * 100

    print(f"""
评估结果:
┌─────────────────┬──────────────┐
│ 评估维度        │ 结果          │
├─────────────────┼──────────────┤
│ 准确性 (Mock)   │ {accuracy_score:.0f}%         │
│ 溯源覆盖率      │ {citation_mentions} 处引用     │
│ 可追溯性得分    │ {traceability_score:.1f}%         │
└─────────────────┴──────────────┘

说明:
- 准确性：在Mock模式下基于预设响应生成
- 溯源覆盖率：初稿中标注来源的数量
- 可追溯性：引用行数占总行数的比例

[注意事项]
当前演示使用Mock LLM，实际使用时：
1. 连接真实Ollama或云端API后生成真实内容
2. 准确性需要人工抽检验证
3. 溯源信息完全基于检索结果
""")

    print_separator("演示完成")
    print("""
下一步:
1. 配置真实LLM（修改config.yaml中的llm.provider）
2. 安装依赖: pip install -r requirements.txt
3. 启动Ollama服务（或配置云端API）
4. 运行完整演示

如需重新演示，清除ChromaDB数据:
  rm -rf memory/chroma
""")


if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
