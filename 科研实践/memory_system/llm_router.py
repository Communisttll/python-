"""LLM调度器 - 统一接口支持多模型"""
import os
from typing import Optional


class LLMRouter:
    """LLM统一调度器，支持本地Ollama和云端API"""

    def __init__(self, provider: str = "ollama", model: str = None, **kwargs):
        self.provider = provider
        self.model = model
        self.config = kwargs

    def chat(self, messages: list, system_prompt: str = None) -> str:
        """统一聊天接口"""
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages

        if self.provider == "ollama":
            return self._chat_ollama(full_messages)
        elif self.provider == "openai":
            return self._chat_openai(full_messages)
        elif self.provider == "anthropic":
            return self._chat_anthropic(full_messages)
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def _chat_ollama(self, messages: list) -> str:
        """Ollama本地模型"""
        import ollama

        response = ollama.chat(
            model=self.model or "qwen2.5:14b",
            messages=messages
        )
        return response["message"]["content"]

    def _chat_openai(self, messages: list) -> str:
        """OpenAI API"""
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model or "gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, messages: list) -> str:
        """Claude API"""
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model or "claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages
        )
        return response.content[0].text

    def generate_draft(self, query: str, context_chunks: list, source_refs: list) -> str:
        """基于记忆片段生成初稿"""
        context_text = "\n\n".join([
            f"[来源{ i+1 }] {chunk}"
            for i, chunk in enumerate(context_chunks)
        ])

        citations = "\n".join([
            f"- 来源{i+1}: {ref}" for i, ref in enumerate(source_refs)
        ])

        prompt = f"""你是一个学术写作助手。基于以下记忆片段，帮我生成一篇研究初稿。

要求：
1. 内容必须基于提供的记忆片段，不要添加虚构信息
2. 每个重要观点后必须标注来源
3. 结构清晰，逻辑通顺
4. 以研究论文的风格撰写

--- 记忆片段 ---
{context_text}

--- 来源 ---
{citations}

--- 用户查询 ---
{query}

--- 初稿 ---
"""

        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="你是一个严谨的学术写作助手，只基于提供的资料进行写作。"
        )


class MockLLMRouter(LLMRouter):
    """用于演示的Mock LLM（不依赖真实API）"""

    def chat(self, messages: list, system_prompt: str = None) -> str:
        user_msg = messages[-1]["content"] if messages else ""

        if "生成初稿" in user_msg or "初稿" in user_msg:
            return self._mock_draft(user_msg)
        elif "检索" in user_msg or "搜索" in user_msg:
            return "检索完成，找到3个相关记忆片段"
        else:
            return f"[Mock响应] 已收到消息，长度{len(user_msg)}字符"

    def _mock_draft(self, prompt: str) -> str:
        """生成模拟初稿"""
        return """## Transformer 研究背景

### 架构优势

Transformer架构通过**自注意力机制**实现了序列建模的突破性进展，相比传统RNN模型具有以下核心优势：

1. **并行计算能力**：打破了序列依赖限制，支持完全并行化训练，显著提升了计算效率
2. **长距离依赖建模**：注意力机制直接计算任意位置间的关联，有效解决了长距离依赖问题
3. **可解释性**：注意力权重可视化提供了模型决策的一定解释能力

> 【来源1: transformer.pdf, p.1-3】
> 【来源2: wiki-attention.html, paragraph 2】

### 主要应用领域

Transformer架构已被广泛应用于多个领域：

- **自然语言处理**：GPT、BERT系列模型
- **计算机视觉**：ViT、DETR等视觉Transformer
- **多模态学习**：CLIP、GPT-4V等跨模态模型

> 【来源3: wiki-transformer.html, Applications section】

### 技术演进时间线

- **2017年**：Vaswani等提出Transformer架构
- **2018年**：BERT刷新NLP多项基准
- **2020年**：GPT-3展示大规模语言模型潜力

> 【来源4: wiki-transformer.html, History section】

---
*本初稿由AI记忆系统基于检索到的记忆片段生成，所有引用均已标注来源。*
"""

    def generate_draft(self, query: str, context_chunks: list, source_refs: list) -> str:
        return self._mock_draft(f"query: {query}")


if __name__ == "__main__":
    print("LLMRouter 模块已加载")
