# 📚 VisRAG - 可视化 RAG 调试平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-0.3.0-green.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/Streamlit-Latest-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

一个基于 **LangChain** 和 **Streamlit** 构建的可视化 RAG (Retrieval-Augmented Generation) 知识库调试平台，支持多种检索策略和文档切分方式，帮助开发者快速测试和优化 RAG 系统效果。

## ✨ 功能特性

### 📄 多格式文档支持
- **PDF** - 自动解析 PDF 文档内容
- **TXT** - 纯文本文档
- **Markdown** - Markdown 格式文档

### ✂️ 灵活的文档切分策略
| 切分方式 | 描述 |
|---------|------|
| **递归字符切分** (推荐) | 智能识别段落、句子等自然边界进行分割 |
| **固定大小切分** | 按固定字符数进行分割 |

支持自定义 Chunk Size 和 Overlap 参数。

### 🔍 三种检索模式
- **Vector (向量检索)** - 基于语义相似度，使用 `sentence-transformers/all-MiniLM-L6-v2` Embedding 模型
- **BM25 (关键字检索)** - 经典 BM25 算法，适合精确关键词匹配场景
- **Hybrid (混合检索)** - 综合 Vector 和 BM25 结果，通过 EnsembleRetriever 加权融合

### 🤖 智能对话模式
- **无知识库模式**: 直接使用 LLM 进行普通对话
- **RAG 模式**: 严格基于知识库内容回答，对于知识库中没有的信息会明确拒绝回答

### 🎨 现代化 UI
- 响应式设计，支持宽屏布局
- 精美的卡片式界面和渐变配色
- 实时 Chunk 预览和元数据查看
- 对话历史中展示检索上下文

## 🛠️ 技术栈

| 组件 | 技术 |
|-----|------|
| 前端框架 | Streamlit |
| LLM 框架 | LangChain 0.3 |
| 向量数据库 | ChromaDB |
| Embedding | HuggingFace Sentence-Transformers |
| 默认 LLM | 智谱 AI GLM-4-Flash (兼容 OpenAI API) |
| 关键字检索 | BM25 (rank_bm25) |

## 📦 安装

### 1. 克隆项目
```bash
git clone https://github.com/your-username/RAG_project.git
cd RAG_project
```

### 2. 创建虚拟环境 (推荐)
```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置 API Key (可选)
可以通过环境变量设置智谱 AI 的 API Key:
```bash
# Windows
set ZHIPU_API_KEY=your-api-key

# Linux/Mac
export ZHIPU_API_KEY=your-api-key
```
也可以在运行时直接在 UI 中输入。

## 🚀 快速开始

### 启动应用
```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`

### 使用流程
1. **上传文档** - 在侧边栏上传 PDF、TXT 或 MD 文件
2. **配置切分参数** - 选择切分方式，调整 Chunk Size 和 Overlap
3. **构建知识库** - 点击「构建/追加知识库」按钮
4. **预览 Chunks** - 在右侧面板查看切分结果和元数据
5. **选择检索模式** - 在对话页面选择 Vector / BM25 / Hybrid
6. **开始对话** - 输入问题，查看 RAG 检索结果和 LLM 回答

## 📁 项目结构

```
RAG_project/
├── app.py              # Streamlit 主应用 (UI 界面)
├── rag_engine.py       # RAG 引擎核心 (检索、LLM 调用)
├── utils.py            # 工具函数 (文件加载、保存)
├── requirements.txt    # Python 依赖
├── chroma_db/          # ChromaDB 持久化目录 (自动生成)
└── temp_uploads/       # 临时上传文件目录 (自动生成)
```

### 核心模块说明

| 文件 | 功能 |
|-----|------|
| `app.py` | Streamlit 前端，包含页面布局、样式、交互逻辑 |
| `rag_engine.py` | `RAGManager` 类，封装文档处理、向量存储、检索器创建、RAG 对话等核心功能 |
| `utils.py` | 文件上传保存、多格式文档加载器 |

## ⚙️ 配置说明

### LLM 配置
默认使用 **智谱 AI** 的 GLM-4-Flash 模型。如需使用其他兼容 OpenAI API 的服务，可修改:
- **Base URL**: API 端点地址
- **Model Name**: 模型名称
- **API Key**: 对应服务的密钥

### 检索参数
- 默认返回 Top 3 相关文档
- Hybrid 模式默认 Vector:BM25 权重为 0.5:0.5

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Made with ❤️ using LangChain & Streamlit**
