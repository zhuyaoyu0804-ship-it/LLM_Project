import os
import shutil
import time
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from utils import load_doc

class RAGManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # 初始化 Embedding，避免每次调用都重新加载
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore = None
        # 存储所有文档用于 BM25 检索
        self.stored_documents: List[Document] = []
        self._init_vectorstore()

    def _init_vectorstore(self):
        """
        初始化或加载现有的 ChromaDB
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def process_file(
        self, 
        file_path: str, 
        chunk_size: int, 
        chunk_overlap: int,
        split_method: str = "recursive"
    ) -> List[Document]:
        """
        加载文件，切分，并存入向量库。
        返回切分后的 chunks 以便预览。
        
        Args:
            split_method: "recursive" (递归字符切分) 或 "fixed" (固定大小切分)
        """
        # 1. 加载
        docs = load_doc(file_path)
        if not docs:
            return []

        # 2. 根据切分方式选择 Splitter
        if split_method == "fixed":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        else:  # 默认 recursive
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
        chunks = text_splitter.split_documents(docs)

        # 3. 存入向量库
        if chunks:
            self.vectorstore.add_documents(chunks)
            # 同时存储到内存列表，用于 BM25 检索
            self.stored_documents.extend(chunks)

        return chunks

    def get_all_documents_metadata(self) -> List[Dict]:
        """
        获取数据库中所有文档的 Metadata 信息，用于列表展示。
        注意：Chroma API 获取所有数据可能较慢，这里仅作简单实现。
        """
        # 这是一个比较重的操作，如果数据量大需优化
        # 直接利用 get() 获取所有 metadatas
        data = self.vectorstore.get()
        metadatas = data['metadatas']
        # 去重，按 source 归类
        unique_files = {}
        for idx, m in enumerate(metadatas):
            if not m: continue
            src = m.get('source', 'Unknown')
            if src not in unique_files:
                unique_files[src] = {'count': 0, 'source': src}
            unique_files[src]['count'] += 1
        
        return list(unique_files.values())

    def delete_document(self, source_path: str):
        """
        根据 source 删除文档
        """
        # Chroma 的 delete 方法支持 where 过滤
        self.vectorstore.delete(where={"source": source_path})

    def clear_database(self):
        """
        完全清空知识库
        """
        # 1. 删除内存中的对象
        # 2. 删除磁盘文件
        if self.vectorstore:
            # 尝试释放资源
            self.vectorstore = None
        
        # 清空 BM25 用的文档列表
        self.stored_documents = []
        
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                time.sleep(0.5) # 等待文件系统释放
            except Exception as e:
                print(f"Error deleting directory: {e}")
        
        # 3. 重新初始化
        self._init_vectorstore()

    def get_retriever(self, search_type="Vector", k=3):
        """
        获取检索器
        
        Args:
            search_type: 
                - "Vector": 向量相似度检索
                - "BM25": 关键字检索 (BM25 算法)
                - "Hybrid": 混合检索 (Vector + BM25 综合排序)
            k: 返回的文档数量
        """
        # 向量检索器
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
        
        if search_type == "Vector":
            return vector_retriever
        
        elif search_type == "BM25":
            # BM25 关键字检索
            if not self.stored_documents:
                print("WARNING: No documents in memory for BM25, falling back to Vector")
                return vector_retriever
            bm25_retriever = BM25Retriever.from_documents(self.stored_documents, k=k)
            return bm25_retriever
        
        elif search_type == "Hybrid":
            # 混合检索: 结合 Vector 和 BM25
            if not self.stored_documents:
                print("WARNING: No documents in memory for Hybrid, falling back to Vector")
                return vector_retriever
            bm25_retriever = BM25Retriever.from_documents(self.stored_documents, k=k)
            # EnsembleRetriever 将两个检索器的结果融合
            # weights 控制两者权重，默认各占 50%
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            return ensemble_retriever
        
        else:
            return vector_retriever

    def chat(self, query: str, api_key: str, base_url: str, model_name: str = "glm-4-flash", search_type: str = "Vector") -> Dict[str, Any]:
        """
        RAG 对话核心方法
        
        智能响应逻辑:
        - 如果知识库为空: 使用普通 LLM 对话模式
        - 如果知识库有文档: 使用 RAG 模式，且对于知识库中没有的内容会拒绝回答
        """
        # 优先使用传入的 api_key，如果为空则尝试环境变量 ZHIPU_API_KEY
        final_api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        
        if not final_api_key:
            return {"error": "请提供 API Key (或设置 ZHIPU_API_KEY 环境变量)"}

        # 1. 准备 LLM
        print(f"DEBUG: initializing LLM with model={model_name}, base_url={base_url}")
        try:
            llm = ChatOpenAI(
                openai_api_key=final_api_key,
                openai_api_base=base_url,
                model_name=model_name,
                temperature=0.1
            )
        except Exception as e:
            print(f"DEBUG: LLM Init Failed: {e}")
            raise e

        # 2. 检查知识库是否为空 (同时检查内存列表和向量库)
        # stored_documents 是内存列表，重启后会清空
        # vectorstore 是持久化的，需要检查两者
        try:
            vectorstore_data = self.vectorstore.get()
            vectorstore_count = len(vectorstore_data.get('ids', []))
        except:
            vectorstore_count = 0
        
        has_documents = len(self.stored_documents) > 0 or vectorstore_count > 0
        print(f"DEBUG: stored_documents={len(self.stored_documents)}, vectorstore={vectorstore_count}, has_documents={has_documents}")
        
        from langchain_core.output_parsers import StrOutputParser
        
        if not has_documents:
            # ========== 模式 A: 普通对话 (无知识库) ==========
            print("DEBUG: Using normal chat mode (no knowledge base)")
            
            normal_prompt = ChatPromptTemplate.from_template("""
你是一个友好的 AI 助手。请用中文回答用户的问题。

用户问题: {input}
""")
            
            normal_chain = normal_prompt | llm | StrOutputParser()
            
            try:
                response = normal_chain.invoke({"input": query})
                return {
                    "answer": response,
                    "source_documents": [],
                    "mode": "normal_chat"
                }
            except Exception as e:
                print(f"DEBUG: Normal chat failed: {e}")
                raise e
        
        else:
            # ========== 模式 B: RAG 模式 (有知识库) ==========
            print("DEBUG: Using RAG mode with strict answering policy")
            
            # 准备 Retriever
            retriever = self.get_retriever(search_type=search_type)
            retrieved_docs = retriever.invoke(query)
            print(f"DEBUG: Retrieved {len(retrieved_docs)} docs")
            
            # 严格的 RAG Prompt - 要求模型只根据上下文回答，否则拒绝
            rag_prompt = ChatPromptTemplate.from_template("""
你是一个基于知识库的问答助手。请严格遵守以下规则:

1. 只能根据下方提供的「上下文」内容来回答问题
2. 如果上下文中没有相关信息，必须回复："抱歉，知识库中没有找到相关内容，无法回答此问题。"
3. 不要编造或推测任何上下文中没有的信息
4. 用中文回答

<上下文>
{context}
</上下文>

用户问题: {input}

请根据上述规则回答:""")

            def format_docs(docs):
                if not docs:
                    return "（无相关内容）"
                return "\n\n---\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]}
                | rag_prompt 
                | llm
                | StrOutputParser()
            )
            
            try:
                response = rag_chain.invoke({
                    "input": query,
                    "context": retrieved_docs
                })
                print("DEBUG: RAG chain invoke success")
                
                return {
                    "answer": response,
                    "source_documents": retrieved_docs,
                    "mode": "rag"
                }
            except Exception as e:
                print(f"DEBUG: RAG chain invoke failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
