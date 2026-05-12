# AI-Native 个人记忆系统设计文档

> 版本：v1.0
> 日期：2026-05-11
> 状态：已批准

---

## 1. 系统概述

### 1.1 系统定位

**AI-Native 个人记忆系统** —— 用于存取日常学习中的无结构化信息，核心作为**写作/研究助手**，同时具备信息捕获检索和知识图谱构建能力。

### 1.2 核心价值

| 能力 | 描述 |
|------|------|
| 信息捕获与检索 | 快速保存PDF/网页/视频内容，支持语义检索 |
| 知识图谱构建 | 将零散信息自动关联成知识网络 |
| 写作/研究助手 | **核心能力** —— 基于记忆内容生成带溯源的初稿 |

### 1.3 输入源

| 来源 | 解析方式 | 输出格式 |
|------|----------|----------|
| arXiv PDF | PyMuPDF + OCR | 文本块 + 页码 + 元数据 |
| Wikipedia | 抓取 + HTML解析 | 标题 + 正文段落 |
| B站视频 | 字幕抓取 / Whisper转写 | 时间戳文本 |

---

## 2. 系统架构

### 2.1 整体架构图

> **图片说明**：需要制作一张系统架构图，建议使用draw.io或Mermaid制作。
>
> **图片生成 Prompt（用于AI绘图工具或设计工具）**：
>
> ```
> 设计一张清晰的系统架构图，风格为技术架构图（Tech Architecture Diagram），
> 使用蓝灰色调，专业简洁。
>
> 图中包含以下层次和组件（从下到上）：
>
> 存储层（底部）:
> - 原始文件存储 (raw/) - 文件夹图标
> - SQLite元数据 - 数据库图标
> - ChromaDB向量索引 - 圆柱体数据库图标
>
> 核心处理层（中间）:
> - 内容解析器 - 齿轮图标，标注 "PDF Parser / Web Scraper / Video Transcriber"
> - 向量索引引擎 - 齿轮图标，标注 "ChromaDB"
> - LLM调度器 - 齿轮图标，标注 "LLM Router (Python)"
>
> 用户交互层（顶部）:
> - 浏览器插件 - 浏览器图标
> - 命令行工具 - 终端图标
> - 桌面客户端（可选） - 桌面图标
>
> 层级之间用箭头连接，展示数据流向（从输入到存储，从检索到输出）
>
> 右侧标注数据流：
> - 输入: 内容 → 解析 → 向量化 → 索引
> - 输出: 查询 → 语义检索 → LLM加工 → 初稿
>
> 整体布局清晰，组件间距均匀，字体易读。
> ```

### 2.2 技术选型

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 内容解析 | PyMuPDF, Playwright, Whisper | PDF/网页/视频内容提取 |
| 向量数据库 | ChromaDB | 本地向量语义检索 |
| LLM推理服务 | Ollama（本地）/ 云端API | 模型部署与调用 |
| LLM调度器 | Python + litellm | 统一接口，支持多模型 |
| 模型来源 | ModelScope / Huggingface | 本地模型下载 |
| 元数据存储 | SQLite | 轻量级本地数据库 |

### 2.3 LLM调度器设计

```
┌─────────────────────────────────────────────────────────┐
│                    用户请求                              │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Python LLM 调度层 (litellm)                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  统一接口: chat_completion(model, messages)      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Ollama  │    │  OpenAI  │    │  Claude  │
   │  (本地)  │    │   API    │    │   API    │
   └──────────┘    └──────────┘    └──────────┘
```

**配置示例 (config.yaml)**：
```yaml
llm:
  provider: "ollama"  # 或 "openai", "anthropic", "deepseek"
  model: "qwen2.5:14b"  # 本地模型
  api_base: "http://localhost:11434"  # Ollama地址

  # 云端配置（切换时使用）
  cloud:
    openai:
      model: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"
    anthropic:
      model: "claude-sonnet-4-20250514"
      api_key: "${ANTHROPIC_API_KEY}"
```

---

## 3. 数据存储结构

### 3.1 目录结构

```
D:\Users\作业\科研实践\
├── memory/                 # 记忆存储根目录
│   ├── raw/               # 原始文件
│   │   ├── papers/        # arXiv论文PDF
│   │   │   ├── transformer.pdf
│   │   │   └── ...
│   │   ├── web/           # 网页内容
│   │   │   └── wiki-ai.html
│   │   └── video/         # 视频字幕/文稿
│   │       └── bilibili-video-xxx.srt
│   ├── chroma/            # ChromaDB向量索引
│   │   ├── collections/
│   │   └── data/
│   └── meta.sqlite        # 元数据数据库
├── config.yaml            # 系统配置
└── logs/                  # 操作日志
```

### 3.2 元数据模型 (SQLite)

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    source_type TEXT,        -- 'pdf', 'web', 'video'
    source_path TEXT,        -- 原始文件路径
    title TEXT,              -- 标题
    tags TEXT,               -- 逗号分隔标签
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    memory_id TEXT,
    content TEXT,            -- 文本内容块
    chunk_index INTEGER,     -- 在原文档中的位置
    page_ref TEXT,           -- 页码/段落引用
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE TABLE citations (
    id TEXT PRIMARY KEY,
    chunk_id TEXT,
    generated_text TEXT,     -- LLM生成的内容
    created_at TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
```

### 3.3 向量索引 (ChromaDB)

每个chunk存储：
- `id`: chunk ID
- `embedding`: 768/1024/1536维向量
- `document`: 原始文本
- `metadata`: {memory_id, page_ref, source_type, tags}

---

## 4. 数据流设计

### 4.1 输入流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 用户输入 │───▶│ 内容解析 │───▶│ 文本分块 │───▶│ 向量化  │
└─────────┘    └─────────┘    └─────────┘    └────┬────┘
                                                  │
                    ┌─────────────────────────────▼────┐
                    │          ChromaDB 索引存储         │
                    └───────────────────────────────────┘
```

### 4.2 输出流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 用户查询 │───▶│语义检索 │───▶│ LLM加工 │───▶│ 带溯源  │
└─────────┘    └────┬────┘    └─────────┘    │  初稿   │
                    │                          └────▲────┘
                    │                               │
                    └───────────────────────────────┘
                    （检索结果作为上下文传入LLM）
```

### 4.3 溯源机制

每个生成片段强制绑定原始来源：

```markdown
## 研究背景

Transformer架构采用自注意力机制实现了序列建模的突破。
【来源: papers/transformer.pdf, p.1-2】
【来源: web/wiki-attention.html, paragraph 5】

根据 Vaswani 等人在 2017 年的研究...
【来源: papers/transformer.pdf, Abstract】
```

---

## 5. 可靠性评估体系

### 5.1 准确性评估

| 指标 | 定义 | 评估方法 |
|------|------|----------|
| 幻觉率 | 生成内容中错误/虚构信息的比例 | 人工抽检 + 与原文比对 |
| 引用正确性 | 溯源标注与原文的一致性 | 随机抽样验证 |

**实现方式**：
1. 每个生成片段必须包含【来源】标注
2. 输出后自动进行原文交叉验证
3. 用户可点击来源直接跳转到原始文档

### 5.2 可追溯性评估

| 指标 | 定义 | 评估方法 |
|------|------|----------|
| 溯源覆盖率 | 可溯源片段 / 总片段数 | 自动统计 |
| 血缘完整度 | 输入→输出全链路是否完整记录 | 血缘图可视化 |

### 5.3 血缘记录机制

血缘（Lineage）是追踪记忆从**原始输入**到**最终输出**完整链路的机制。每条血缘记录包含以下关键信息：

#### 5.3.1 血缘节点类型

| 节点类型 | 说明 | 示例 |
|----------|------|------|
| `source` | 原始输入源 | `transformer.pdf`, `wiki-attention.html` |
| `chunk` | 分块后的文本片段 | `chunk_001`, `chunk_002` |
| `embedding` | 向量化后的索引记录 | `emb_001` (关联ChromaDB ID) |
| `memory` | 存储的记忆单元 | `mem_abc123` |
| `retrieval` | 检索结果记录 | 检索到的相关片段及相似度分数 |
| `citation` | 引用/溯源记录 | `cit_xyz789` (关联到chunk) |
| `draft` | LLM生成的初稿 | 生成的完整文本及来源标注 |

#### 5.3.2 血缘边关系

```
source ──分块──▶ chunk ──向量化──▶ embedding ──存储──▶ memory
                                                        │
                                                        ▼
                                                     retrieval
                                                        │
                                                        ▼
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                       citation    citation     citation
                                          │             │             │
                                          ▼             ▼             ▼
                                        chunk        chunk        chunk
                                          │             │             │
                                          └─────────────┼─────────────┘
                                                        ▼
                                                       draft
```

每条边记录：
- `from_node`: 源节点ID
- `to_node`: 目标节点ID
- `relation_type`: 关系类型（`parsed_from`, `embedded_from`, `retrieved_from`, `cited_from`, `generated_from`）
- `metadata`: 附加信息（如分块位置、相似度分数、页码引用等）
- `timestamp`: 记录时间

#### 5.3.3 血缘记录表结构 (SQLite)

```sql
-- 血缘节点表：记录所有血缘相关的节点
CREATE TABLE lineage_nodes (
    node_id TEXT PRIMARY KEY,           -- 节点唯一标识 (如: src_001, chunk_002)
    node_type TEXT NOT NULL,            -- 节点类型: source|chunk|embedding|memory|retrieval|citation|draft
    memory_id TEXT,                      -- 关联的记忆ID (对chunk/embedding/memory节点)
    source_path TEXT,                    -- 原始文件路径 (对source节点)
    content_hash TEXT,                   -- 内容哈希，用于验证完整性
    metadata TEXT,                       -- JSON格式的元数据 (页码、位置、标签等)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 血缘边表：记录节点之间的关系
CREATE TABLE lineage_edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    relation_type TEXT NOT NULL,        -- parsed_from|embedded_from|stored_in|retrieved_from|cited_from|generated_from
    metadata TEXT,                       -- JSON格式: {position, page_ref, similarity_score, ...}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_node) REFERENCES lineage_nodes(node_id),
    FOREIGN KEY (to_node) REFERENCES lineage_nodes(node_id)
);

-- 血缘路径表：记录完整的输入-输出链路
CREATE TABLE lineage_paths (
    path_id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL,             -- 最终输出的draft节点ID
    source_nodes TEXT NOT NULL,          -- JSON数组: 所有相关的source节点ID
    intermediate_path TEXT NOT NULL,     -- JSON数组: 完整的中间节点链路
    coverage_ratio REAL,                 -- 溯源覆盖率 (0.0-1.0)
    completeness_score REAL,             -- 血缘完整度评分
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (draft_id) REFERENCES lineage_nodes(node_id)
);

-- 索引以加速血缘查询
CREATE INDEX idx_edges_from ON lineage_edges(from_node);
CREATE INDEX idx_edges_to ON lineage_edges(to_node);
CREATE INDEX idx_edges_relation ON lineage_edges(relation_type);
CREATE INDEX idx_nodes_type ON lineage_nodes(node_type);
CREATE INDEX idx_nodes_memory ON lineage_nodes(memory_id);
```

#### 5.3.4 血缘追踪流程

**输入阶段（Ingestion）**：
```
1. 用户输入PDF/网页/视频
   └── 创建 source 节点，记录原始文件路径和哈希

2. 内容解析器解析文档
   └── 创建 chunk 节点，edge: source ─parsed_from──▶ chunk
   └── metadata: {page_start, page_end, chunk_index, char_offset}

3. 向量化引擎处理
   └── 创建 embedding 节点，edge: chunk ─embedded_from──▶ embedding
   └── metadata: {embedding_model, dimensions, vector_id}

4. 存入ChromaDB
   └── 创建 memory 节点，edge: embedding ─stored_in──▶ memory
   └── metadata: {chroma_collection, chroma_id}
```

**检索阶段（Retrieval）**：
```
1. 用户查询 "transformer架构优势"
   └── 创建 retrieval 节点
   └── edge: memory ─retrieved_from──▶ retrieval
   └── metadata: {query, top_k, similarity_scores[]}

2. 选择相关片段
   └── edge: retrieval ─cites──▶ citation
   └── metadata: {chunk_id, relevance_score, position_in_context}
```

**生成阶段（Generation）**：
```
1. LLM生成初稿
   └── 创建 draft 节点
   └── edge: citation ─cited_from──▶ chunk
   └── edge: draft ─generated_from──▶ retrieval
   └── metadata: {model, prompt_template, generation_time}

2. 记录完整路径
   └── 创建 lineage_path 记录
   └── 追踪: source[] → chunk[] → memory → retrieval → draft
```

#### 5.3.5 血缘查询接口

```python
# 查询某条记忆的完整血缘
def get_memory_lineage(memory_id: str) -> LineagePath:
    """获取记忆从输入到所有相关输出的完整链路"""

# 查询某条输出的来源
def get_output_sources(draft_id: str) -> List[SourceNode]:
    """追踪输出内容的所有原始来源"""

# 验证血缘完整性
def validate_lineage_completeness(path_id: str) -> CompletenessReport:
    """检查血缘链路是否有断点，返回完整性报告"""

# 溯源查询
def trace_citation_to_source(citation_id: str) -> SourceInfo:
    """将引用追溯到原始输入源"""
```

#### 5.3.6 血缘完整性验证

| 检查项 | 说明 | 失败处理 |
|--------|------|----------|
| 节点存在性 | 所有引用的节点ID都存在 | 标记为孤岛节点 |
| 边连续性 | 每条边的from/to节点都有效 | 标记断点 |
| 内容哈希 | 验证内容未被篡改 | 警告+记录差异 |
| 循环检测 | 检查是否有非法循环引用 | 拒绝创建 |
| 时间戳顺序 | 确保时间戳递增 | 标记异常 |

**血缘完整度评分公式**：
```
completeness_score = (有效边数 / 理论边数) × 内容哈希验证率 × 时间顺序正确率
```

### 5.4 记忆血缘图详解

> **图片说明**：需要制作一张记忆血缘图，示例流程。
>
> **图片生成 Prompt**：
>
> ```
> 设计一张记忆血缘图（Memory Lineage Graph），展示知识从输入到输出的完整链路。
> 风格：简洁的信息流图解（Information Flow Diagram），使用蓝绿色调。
>
> 横向流程（从左到右）：
>
> [输入层]
> ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
> │ PDF论文     │  │ Wikipedia   │  │ B站视频     │
> │ transformer │  │ AI词条      │  │ 科普视频    │
> │ .pdf        │  │ .html       │  │ .srt        │
> └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
>
> [处理层]
>         │              │              │
>         ▼              ▼              ▼
>    ┌─────────┐   ┌─────────┐   ┌─────────┐
>    │文本分块  │   │文本分块  │   │字幕分块  │
>    └────┬────┘   └────┬────┘   └────┬────┘
>         │              │              │
>         ▼              ▼              ▼
>    ┌─────────┐   ┌─────────┐   ┌─────────┐
>    │向量化    │   │向量化    │   │向量化    │
>    │(768维)  │   │(768维)  │   │(768维)  │
>    └────┬────┘   └────┬────┘   └────┬────┘
>
> [存储层]
>         │              │              │
>         └──────────────┼──────────────┘
>                        ▼
>               ┌─────────────────┐
>               │   ChromaDB      │
>               │   语义索引      │
>               └────────┬────────┘
>
> [检索层]
>                        │
>                        ▼
>               ┌─────────────────┐
>               │   用户查询       │
>               │  "transformer   │
>               │   研究背景"     │
>               └────────┬────────┘
>
> [输出层]
>                        │
>                        ▼
>               ┌─────────────────┐
>               │   LLM生成       │
>               │   带溯源初稿    │
>               └────────┬────────┘
>
> 用虚线框标注每个阶段的"血缘记录"：输入ID → chunk ID → 溯源ID → 输出ID
>
> 底部添加图例说明：
> - 实线箭头: 数据流向
> - 虚线: 血缘追踪
> - 数字标识每一步的元数据
> ```

#### 血缘图节点详解

**层级一：输入源节点（Source Nodes）**
```
属性：
- node_id: src_{hash}  (如: src_a1b2c3d4)
- node_type: "source"
- metadata: {
    source_type: "pdf" | "web" | "video",
    file_path: "raw/papers/transformer.pdf",
    file_hash: "sha256:...",
    file_size: 2048576,
    title: "Attention is All You Need",
    author: "Vaswani et al.",
    created_at: "2026-05-11T10:30:00Z"
  }

示例节点：
┌────────────────────────────────────────┐
│ src_a1b2c3d4                           │
│ type: source                           │
│ ──────────────────────────────────── │
│ path: raw/papers/transformer.pdf      │
│ type: pdf                             │
│ title: Attention is All You Need      │
│ pages: 12                              │
│ hash: sha256:e3b0c44298fc1c...        │
└────────────────────────────────────────┘
```

**层级二：分块节点（Chunk Nodes）**
```
属性：
- node_id: chunk_{memory_id}_{index}  (如: chunk_mem123_001)
- node_type: "chunk"
- metadata: {
    chunk_index: 1,
    page_start: 1,
    page_end: 2,
    char_start: 0,
    char_end: 2048,
    content_preview: "Transformer架构采用自注意力...",
    content_hash: "md5:..."
  }

边关系：
- source ─parsed_from──▶ chunk  (1:N，每个source产生多个chunk)
- chunk ─embedded_from──▶ embedding  (1:1)
```

**层级三：向量嵌入节点（Embedding Nodes）**
```
属性：
- node_id: emb_{chroma_id}  (如: emb_7f8a9b0c)
- node_type: "embedding"
- metadata: {
    embedding_model: "text2vec-base-chinese",
    dimensions: 768,
    chroma_collection: "memories",
    chroma_id: "落地ChromaDB中的ID"
  }

边关系：
- embedding ─stored_in──▶ memory  (1:1)
```

**层级四：记忆节点（Memory Nodes）**
```
属性：
- node_id: mem_{uuid}  (如: mem_123e4567)
- node_type: "memory"
- metadata: {
    title: "Transformer自注意力机制",
    tags: ["深度学习", "NLP", "Transformer"],
    source_count: 2,      // 引用了多少个source
    chunk_count: 3,        // 包含多少个chunk
    indexed_at: "2026-05-11T10:35:00Z"
  }

边关系：
- memory ─retrieved_from──▶ retrieval  (N:M)
```

**层级五：检索结果节点（Retrieval Nodes）**
```
属性：
- node_id: ret_{query_hash}_{timestamp}  (如: ret_abc123_20260511)
- node_type: "retrieval"
- metadata: {
    query: "transformer架构的核心优势",
    top_k: 5,
    results: [
      {memory_id: "mem_123", score: 0.92, chunk_id: "chunk_001"},
      {memory_id: "mem_456", score: 0.87, chunk_id: "chunk_015"},
      ...
    ],
    retrieval_time: "2026-05-11T11:00:00Z"
  }

边关系：
- memory ─retrieved_from──▶ retrieval
- retrieval ─cites──▶ citation  (1:N)
```

**层级六：引用节点（Citation Nodes）**
```
属性：
- node_id: cit_{uuid}
- node_type: "citation"
- metadata: {
    chunk_id: "chunk_001",
    position_in_context: 0,    // 在生成上下文中的位置
    relevance_score: 0.92,
    page_ref: "p.1-3",
    excerpt: "Transformer架构通过自注意力机制..."
  }

边关系：
- citation ─cited_from──▶ chunk  (N:1)
- citation ─part_of──▶ draft  (N:1)
```

**层级七：初稿节点（Draft Nodes）**
```
属性：
- node_id: draft_{uuid}
- node_type: "draft"
- metadata: {
    title: "Transformer研究背景",
    content: "完整生成的文本...",
    model: "qwen2.5:14b",
    generation_time_ms: 3200,
    token_count: 1536,
    citations_used: ["cit_001", "cit_002", "cit_003"]
  }

边关系：
- draft ─generated_from──▶ retrieval  (N:1)
- draft ─includes──▶ citation  (1:N)
```

#### 血缘图完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           完整血缘链路示例                                     │
└─────────────────────────────────────────────────────────────────────────────┘

输入阶段:
────────────────────────────────────────────────────────────────────────────
                                                                         │
    ┌──────────────┐                                                      │
    │ src_pdf_001  │ ──parsed_from──▶ ┌──────────────┐                    │
    │ transformer   │                  │ chunk_001_01 │                    │
    │ .pdf          │ ──parsed_from──▶ │ chunk_001_02 │                    │
    └──────────────┘                  └──────┬───────┘                    │
                                              │                             │
                                              ▼                             │
                                       ┌──────────────┐                     │
    ┌──────────────┐                   │ emb_001_01   │                     │
    │ src_wiki_002 │ ──parsed_from──▶ │ emb_001_02   │                     │
    │ attention    │                   └──────┬───────┘                     │
    └──────────────┘                          │                             │
                                              ▼                             │
                                       ┌──────────────┐                     │
                                       │ mem_001      │                      │
    ┌──────────────┐                   │ mem_002      │                      │
    │ src_video_003│                   └──────┬───────┘                     │
    │ bilibili     │                          │                             │
    └──────────────┘                          │                             │
                                              ▼                             │
──────────────────────────────────────────────────────────────────────────────
                                        存储层 (ChromaDB + SQLite)
──────────────────────────────────────────────────────────────────────────────
                                              │
                                              ▼
检索阶段:                                 ┌──────────────┐
                                              │ ret_query_01 │
    ┌──────────────┐                          │ "transformer  │
    │ 用户查询      │ ──────────────────────▶ │  研究背景"   │
    │ "transformer │                          └──────┬───────┘
    │  研究背景"   │                                   │
    └──────────────┘                                   │
                                              ┌───────┴────────┐
                                              ▼                ▼
                                       ┌────────────┐  ┌────────────┐
                                       │ cit_001   │  │ cit_002   │
                                       │ 来自mem_001│  │ 来自mem_002│
                                       └─────┬──────┘  └─────┬──────┘
                                             │               │
──────────────────────────────────────────────────────────────────────────────
                                              │               │
生成阶段:                                      ▼               ▼
                                       ┌────────────────────────────────────┐
    ┌──────────────┐                    │          draft_001                  │
    │ LLM (qwen)   │ ◀──generated_from──│     Transformer研究背景             │
    │              │                    │     [来源: src_pdf_001, p.1-3]     │
    └──────┬───────┘                    │     [来源: src_wiki_002, paragraph 5│
           │                            └────────────────────────────────────┘
           │ generated_from
           ▼
    ┌──────────────┐
    │ lineage_path │
    │ _001         │
    └──────────────┘
```

#### 血缘图可视化展示

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        记忆血缘追踪界面                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  当前查看: draft_001 "Transformer研究背景"                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         血缘链路可视化                                 │   │
│  │                                                                       │   │
│  │   [📄 PDF] ──▶ [📦 Chunk] ──▶ [🧠 Memory] ──▶ [🔍 Retrieval]        │   │
│  │      │                                                   │           │   │
│  │      │                                                   ▼           │   │
│  │      └──────────────────────▶ [📎 Citation] ◀── [✍️ Draft]          │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ──────────────────────────────────────────────────────────────────────    │
│  详细节点信息:                                                               │
│                                                                             │
│  draft_001                                                                  │
│  ├── model: qwen2.5:14b                                                    │
│  ├── generated_at: 2026-05-11 11:00:00                                    │
│  ├── citations_used: [cit_001, cit_002, cit_003]                           │
│  │                                                                          │
│  ├── cit_001 ──cites──▶ chunk_001_01                                       │
│  │   ├── source: transformer.pdf, p.1-3                                   │
│  │   ├── relevance: 0.92                                                  │
│  │   └── excerpt: "Transformer架构通过自注意力机制..."                      │
│  │                                                                          │
│  ├── cit_002 ──cites──▶ chunk_002_05                                       │
│  │   ├── source: wiki-attention.html, paragraph 5                         │
│  │   ├── relevance: 0.87                                                  │
│  │   └── excerpt: "注意力机制允许模型..."                                  │
│  │                                                                          │
│  └── lineage_completeness: 100%                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**血缘图示例**：
> **图片说明**：需要制作一张记忆血缘图，示例流程。
> **图片说明**：需要制作一张记忆血缘图，示例流程。
>
> **图片生成 Prompt**：
>
> ```
> 设计一张记忆血缘图（Memory Lineage Graph），展示知识从输入到输出的完整链路。
> 风格：简洁的信息流图解（Information Flow Diagram），使用蓝绿色调。
>
> 横向流程（从左到右）：
>
> [输入层]
> ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
> │ PDF论文     │  │ Wikipedia   │  │ B站视频     │
> │ transformer │  │ AI词条      │  │ 科普视频    │
> │ .pdf        │  │ .html       │  │ .srt        │
> └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
>
> [处理层]
>         │              │              │
>         ▼              ▼              ▼
>    ┌─────────┐   ┌─────────┐   ┌─────────┐
>    │文本分块  │   │文本分块  │   │字幕分块  │
>    └────┬────┘   └────┬────┘   └────┬────┘
>         │              │              │
>         ▼              ▼              ▼
>    ┌─────────┐   ┌─────────┐   ┌─────────┐
>    │向量化    │   │向量化    │   │向量化    │
>    │(768维)  │   │(768维)  │   │(768维)  │
>    └────┬────┘   └────┬────┘   └────┬────┘
>
> [存储层]
>         │              │              │
>         └──────────────┼──────────────┘
>                        ▼
>               ┌─────────────────┐
>               │   ChromaDB      │
>               │   语义索引      │
>               └────────┬────────┘
>
> [检索层]
>                        │
>                        ▼
>               ┌─────────────────┐
>               │   用户查询       │
>               │  "transformer   │
>               │   研究背景"     │
>               └────────┬────────┘
>
> [输出层]
>                        │
>                        ▼
>               ┌─────────────────┐
>               │   LLM生成       │
>               │   带溯源初稿    │
>               └────────┬────────┘
>
> 用虚线框标注每个阶段的"血缘记录"：输入ID → chunk ID → 溯源ID → 输出ID
>
> 底部添加图例说明：
> - 实线箭头: 数据流向
> - 虚线: 血缘追踪
> - 数字标识每一步的元数据
> ```

---

## 6. 演示案例

### 6.1 演示数据

| 类型 | 数据 | 获取方式 |
|------|------|----------|
| arXiv PDF | Attention is All You Need (1706.03762) | 自动下载 |
| Wikipedia | "Artificial Intelligence" 词条 | 公开抓取 |
| B站视频 | AI科普视频（技术向） | 字幕抓取 |

### 6.2 演示流程

#### Step 1: 内容输入与解析

```
输入: transformer.pdf (arXiv)
处理: PyMuPDF解析 → 文本分块 → 向量化
输出: 存储至 ChromaDB，索引创建成功
```

#### Step 2: 语义检索

```
查询: "transformer架构的核心优势和局限性"
检索: ChromaDB语义相似度搜索
结果: 返回Top-5相关记忆片段（附相似度分数）
```

#### Step 3: 初稿生成

```
输入: 检索结果 + 写作指令
处理: LLM上下文注入 → 结构化生成
输出: 带完整溯源的研究背景初稿
```

#### Step 4: 评估结果

```
准确性: 幻觉检验通过率 95%
可追溯性: 溯源覆盖率 100%（每段均有来源标注）
```

### 6.3 预期演示结果示例

> **图片说明**：需要制作一张演示结果截图的示例图。
>
> **图片生成 Prompt**：
>
> ```
> 创建一个Markdown渲染风格的截图示例，展示AI记忆系统生成的带溯源初稿。
> 风格：现代Markdown编辑器界面截图（类似Typora或Obsidian）
>
> 内容布局：
>
> ┌──────────────────────────────────────────────────────────────┐
> │ 📝 研究背景初稿                           [模型: qwen2.5:14b] │
> ├──────────────────────────────────────────────────────────────┤
> │                                                              │
> │ # Transformer 研究背景                                      │
> │                                                              │
> │ ## 架构优势                                                  │
> │                                                              │
> │ Transformer 架构通过**自注意力机制**实现了序列建模的突破，    │
> │ 相比传统RNN具有以下优势：                                    │
> │                                                              │
> │ - **并行计算**: 摆脱了序列依赖，支持完整并行训练              │
> │ - **长距离依赖**: 注意力机制直接建模任意位置间的关联          │
> │ - **可解释性**: 注意力权重可视化提供了一定的解释能力          │
> │                                                              │
> │ > 【来源: papers/transformer.pdf, p.1-3】                    │
> │                                                              │
> │ ## 主要应用                                                  │
> │                                                              │
> │ 1. 自然语言处理（GPT、BERT系列）                             │
> │ 2. 计算机视觉（ViT、DETR）                                  │
> │ 3. 多模态模型（CLIP、GPT-4V）                                │
> │                                                              │
> │ > 【来源: web/wiki-transformer.html, Applications】          │
> │                                                              │
> │ ## 时间线                                                     │
> │                                                              │
> │ - 2017: Vaswani等提出Transformer架构                        │
> │ - 2018: BERT刷新NLP基准                                      │
> │ - 2020: GPT-3展示大规模语言模型潜力                          │
> │                                                              │
> │ > 【来源: web/wiki-transformer.html, History】               │
> │                                                              │
> ├──────────────────────────────────────────────────────────────┤
> │ [📋 记忆血缘] [📊 评估报告] [💾 导出] [📤 分享]              │
> └──────────────────────────────────────────────────────────────┘
>
> 界面底部显示：
> - 准确性: ✓ 95%  |  可追溯性: ✓ 100%  |  生成耗时: 3.2s
>
> 整体设计现代简洁，使用深色主题，配色专业。
> ```

---

## 7. 实现计划

### 7.1 阶段划分

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| Phase 1 | 核心框架：PDF解析 + ChromaDB索引 + 基础检索 | P0 |
| Phase 2 | LLM调度器：统一接口 + 本地/云端切换 | P0 |
| Phase 3 | 网页解析 + B站视频字幕获取 | P1 |
| Phase 4 | 溯源机制 + 初稿生成 | P1 |
| Phase 5 | 评估体系 + 界面优化 | P2 |

### 7.2 后续步骤

1. 编写详细实现计划（writing-plans）
2. 搭建开发环境
3. 实现核心组件
4. 演示验证

---

## 8. 附录

### 8.1 术语表

| 术语 | 定义 |
|------|------|
| 记忆（Memory） | 系统中存储的任意知识单元（文档/片段/标签） |
| 向量语义检索 | 基于语义相似度的文档检索方式 |
| 血缘（Lineage） | 记忆从输入到输出的完整追踪链路 |
| 初稿（Draft） | LLM基于记忆生成的原始文本 |

### 8.2 参考资料

- ChromaDB 文档: https://docs.trychroma.com
- litellm 文档: https://docs.litellm.ai
- Ollama 文档: https://github.com/ollama/ollama
- PyMuPDF 文档: https://pymupdf.readthedocs.io
