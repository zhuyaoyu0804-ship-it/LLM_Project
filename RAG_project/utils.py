import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """
    保存 Streamlit 上传的文件到指定目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_doc(file_path: str) -> List[Document]:
    """
    根据文件扩展名加载文档。
    支持: .pdf, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path) # 需要 unstructured 库吗？或者直接用 TextLoader
        # 为了简化依赖，对于 .md 我们也可以暂时用 TextLoader，除非需要高级解析
        # 考虑到 Unstructured 依赖较重，我们先尝试用 TextLoader 读取 MD
        loader = TextLoader(file_path, encoding="utf-8") 
    else:
        return []
        
    return loader.load()
