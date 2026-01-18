import streamlit as st
import pandas as pd
import os
from rag_engine import RAGManager
from utils import save_uploaded_file

import sys
# --- 页面配置 ---
st.set_page_config(
    page_title="RAG 知识库调试平台",
    page_icon="📚",
    layout="wide"
)

# --- 样式 ---
st.markdown("""
<style>
    /* 隐藏默认的分割线和边框 */
    .stApp > header {
        background-color: transparent;
    }
    
    /* 主容器样式 */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* 卡片样式 */
    .card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e8e8e8 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #b8b8b8 !important;
    }
    
    /* 按钮美化 */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    
    /* 输入框美化 */
    .stTextInput input, .stSelectbox select {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* 标签页美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f5f7fa;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Chunk 预览区域 */
    .chunk-preview {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        border-left: 4px solid #667eea;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* 文件列表项 */
    .file-item {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        background: #ffffff;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .file-item:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* 聊天消息美化 */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        margin: 0.5rem 0;
    }
    
    /* 隐藏调试信息 */
    .debug-info {
        display: none;
    }
    
    /* 统计数字标签 */
    .stat-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 初始化 Session State ---
if "rag" not in st.session_state:
    st.session_state.rag = RAGManager()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []

# --- 侧边栏 ---
with st.sidebar:
    st.title("⚙️ 配置面板")
    
    st.header("1. 文档上传")
    uploaded_files = st.file_uploader(
        "选择文档 (PDF, TXT, MD)", 
        accept_multiple_files=True,
        type=["pdf", "txt", "md"]
    )
    
    st.header("2. 切分参数")
    split_method = st.selectbox(
        "切分方式",
        options=["recursive", "fixed"],
        format_func=lambda x: "递归字符切分 (推荐)" if x == "recursive" else "固定大小切分",
        index=0,
        help="递归切分会按段落、句子智能分割；固定切分则按字符数硬切"
    )
    chunk_size = st.number_input("Chunk Size (字符数)", min_value=50, max_value=4000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap (重叠字符)", min_value=0, max_value=500, value=50, step=10)
    
    if st.button("🏗️ 构建/追加知识库", type="primary"):
        if not uploaded_files:
            st.warning("请先上传文件！")
        else:
            with st.spinner("正在处理文档..."):
                all_new_chunks = []
                temp_dir = "temp_uploads"
                for uploaded_file in uploaded_files:
                    # 保存文件
                    file_path = save_uploaded_file(uploaded_file, temp_dir)
                    # 处理 (传入切分方式)
                    chunks = st.session_state.rag.process_file(
                        file_path, 
                        chunk_size, 
                        chunk_overlap,
                        split_method=split_method
                    )
                    all_new_chunks.extend(chunks)
                
                st.session_state.latest_chunks = all_new_chunks
                st.success(f"成功处理 {len(uploaded_files)} 个文件，共生成 {len(all_new_chunks)} 个 Chunks！")
    
    st.header("3. LLM 设置 (默认智谱 AI)")
    
    # 自动检测环境变量
    env_key = os.environ.get("ZHIPU_API_KEY", "")
    api_key_placeholder = "已检测到 ZHIPU_API_KEY" if env_key else "请输入 API Key"
    
    api_key = st.text_input("API Key (为空则使用 ZHIPU_API_KEY)", type="password", placeholder=api_key_placeholder)
    base_url = st.text_input("Base URL", value="https://open.bigmodel.cn/api/paas/v4/")
    model_name = st.text_input("Model Name", value="glm-4-flash")

    st.divider()
    
    st.header("⚠️ 危险操作")
    if st.button("🗑️ 清空所有知识库", type="secondary"):
        st.session_state.rag.clear_database()
        st.session_state.latest_chunks = []
        # 强制刷新以更新界面状态
        st.success("知识库已清空！")
        st.rerun()

# --- 主界面 ---
st.title("📚 VisRAG - 可视化 RAG 调试平台")

tab1, tab2 = st.tabs(["📖 知识库管理 & 预览", "🤖 RAG 对话测试"])

# === Tab 1: 知识库管理 ===
with tab1:
    # 使用两列布局填充空间
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("""
        <div class="card">
            <div class="card-header">📊 已加载文档</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 获取当前数据库状态
        file_stats = st.session_state.rag.get_all_documents_metadata()
        
        if not file_stats:
            st.info("📭 当前知识库为空。请在侧边栏上传文档并点击构建。")
        else:
            # 统计信息
            total_chunks = sum(f['count'] for f in file_stats)
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div class="stat-badge">📁 {len(file_stats)} 个文件</div>
                <div class="stat-badge">📄 {total_chunks} 个 Chunks</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 显示文件列表
            for file_data in file_stats:
                with st.container():
                    col1, col2, col3 = st.columns([4, 2, 1])
                    with col1:
                        st.markdown(f"📄 **{os.path.basename(file_data['source'])}**")
                    with col2:
                        st.caption(f"{file_data['count']} chunks")
                    with col3:
                        if st.button("🗑️", key=f"del_{file_data['source']}", help="删除此文件"):
                            st.session_state.rag.delete_document(file_data['source'])
                            st.toast(f"已删除 {os.path.basename(file_data['source'])}")
                            st.rerun()
    
    with col_right:
        st.markdown("""
        <div class="card">
            <div class="card-header">🔍 Chunk 预览</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.latest_chunks:
            # 转换为 DataFrame 用于展示
            data = []
            for i, chunk in enumerate(st.session_state.latest_chunks):
                data.append({
                    "ID": i,
                    "来源": os.path.basename(chunk.metadata.get("source", "Unknown")),
                    "字符数": len(chunk.page_content),
                    "内容预览": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, height=300)
            
            # 详情查看
            st.markdown("---")
            selected_id = st.number_input("🔢 输入 Chunk ID 查看完整内容", min_value=0, max_value=len(data)-1, value=0, step=1)
            if 0 <= selected_id < len(data):
                with st.expander(f"📝 Chunk {selected_id} 完整内容", expanded=True):
                    st.markdown(f"<div class='chunk-preview'>{st.session_state.latest_chunks[selected_id].page_content}</div>", unsafe_allow_html=True)
                with st.expander("🏷️ 元数据"):
                    st.json(st.session_state.latest_chunks[selected_id].metadata)
        else:
            st.info("💡 构建知识库后，这里将显示切分后的 Chunk 预览。")
            st.markdown("""
            **使用说明**:
            1. 在左侧上传文档
            2. 选择切分方式和参数
            3. 点击「构建知识库」按钮
            """)

# === Tab 2: RAG 对话 ===
with tab2:
    col_config, col_chat = st.columns([1, 3])
    
    with col_config:
        st.markdown("**🔧 检索配置**")
        search_type = st.radio(
            "检索模式",
            ["Vector", "BM25", "Hybrid"],
            index=0,
            help="Vector: 语义相似度 | BM25: 关键字匹配 | Hybrid: 综合排序"
        )
        # 直接使用选择的值
        real_search_type = search_type
            
    with col_chat:
        # 显示历史消息
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "source_documents" in msg:
                    with st.expander("🔍 检索到的上下文 (历史记录)"):
                        for doc in msg["source_documents"]:
                            st.markdown(f"**来源**: `{os.path.basename(doc.metadata.get('source', 'unknown'))}`")
                            st.markdown(f"```\n{doc.page_content[:200]}...\n```")

        # 输入框
        if prompt := st.chat_input("请输入你的问题..."):
            # 1. 显示用户输入
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. 调用 RAG
            # 这里的 check 稍微宽容一点，如果没有输入 key 但是有 env key 也可以
            final_key = api_key or os.environ.get("ZHIPU_API_KEY")
            
            if not final_key:
                st.error("请在侧边栏填写 API Key，或设置 ZHIPU_API_KEY 环境变量！")
            else:
                with st.chat_message("assistant"):
                    with st.spinner("正在思考..."):
                        try:
                            result = st.session_state.rag.chat(
                                query=prompt,
                                api_key=api_key, # 传入原始输入即可，rag_engine 内部会再次 fallback
                                base_url=base_url,
                                model_name=model_name,
                                search_type=real_search_type
                            )
                            
                            answer = result.get("answer")
                            source_docs = result.get("source_documents", [])
                            
                            # 展示检索到的上下文
                            with st.expander("🔍 检索到的上下文", expanded=True):
                                if not source_docs:
                                    st.write("未检索到相关文档。")
                                for i, doc in enumerate(source_docs):
                                    st.markdown(f"**DOC {i+1}** - `{os.path.basename(doc.metadata.get('source', 'unknown'))}`")
                                    st.markdown(f"```\n{doc.page_content}...\n```")
                            
                            # 展示回答
                            st.markdown(answer)
                            
                            # 保存历史
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "source_documents": source_docs
                            })
                            
                        except Exception as e:
                            st.error("执行出错，详细日志请查看终端。")
                            st.exception(e)

