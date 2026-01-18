import os
# 注意这里导入的是 ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# =================配置区域=================
# 填入智谱的 API Key
os.environ["ZHIPUAI_API_KEY"] = "你的智谱API_KEY"

# =================1. 初始化模型 (使用 ChatOpenAI)=================
llm = ChatOpenAI(
    # 【关键点 1】智谱的 OpenAI 兼容接口地址
    base_url="https://open.bigmodel.cn/api/paas/v4/", 
    
    # 【关键点 2】使用智谱的 API Key
    api_key=os.environ["ZHIPU_API_KEY"],
    
    # 【关键点 3】指定智谱的模型名称
    model="glm-4-flash",  
    
    temperature=0.5,
    streaming=True,
)

# =================2. 下面的代码与之前完全一致=================
# 构建 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个使用 OpenAI 兼容协议的 GLM-4 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# 记忆存储
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# =================3. 运行测试=================
if __name__ == "__main__":
    # =================配置区域=================
    # 填入智谱的 API Key
    os.environ["ZHIPUAI_API_KEY"] = "你的智谱API_KEY"

    # =================1. 初始化模型 (使用 ChatOpenAI)=================
    llm = ChatOpenAI(
        # 【关键点 1】智谱的 OpenAI 兼容接口地址
        base_url="https://open.bigmodel.cn/api/paas/v4/", 
        
        # 【关键点 2】使用智谱的 API Key
        api_key=os.environ["ZHIPU_API_KEY"],
        
        # 【关键点 3】指定智谱的模型名称
        model="glm-4-flash",  
        
        temperature=0.5,
        streaming=True,
    )

    # =================2. 下面的代码与之前完全一致=================
    # 构建 Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个使用 OpenAI 兼容协议的 GLM-4 助手。请根据用户的问题提供准确的回答。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    # 记忆存储
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversation_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # =================3. 运行测试=================
    session_id = "user_123" # 定义一个固定的会话ID，保证有记忆
    
    print("--- 智谱 GLM-4 对话系统启动 ---")
    print("提示：输入 'q' 或 'exit' 并回车即可退出程序\n")

    while True:
        # 1. 获取用户输入
        # strip() 去除首尾空格，防止用户只敲了空格导致空发
        user_input = input("User: ").strip()
        
        # 2. 判断是否退出
        if not user_input: # 如果直接回车，不处理
            continue
        if user_input.lower() in ["q", "exit", "quit"]:
            print("AI: 再见！")
            break
            
        # 3. 调用 AI 并流式输出
        print("AI: ", end="")
        try:
            for chunk in conversation_chain.stream(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            ):
                print(chunk.content, end="", flush=True)
            print("\n") # 只有在完整回复结束后才换行
            
        except Exception as e:
            print(f"\n[系统错误]: {e}")
            print("请检查网络或 API Key 设置。")
            break