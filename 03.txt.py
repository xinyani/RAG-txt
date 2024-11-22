#!/usr/bin/python
# -*- coding:utf-8 -*-
import bs4
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain.document_loaders import TextLoader

# from modelscope import AutoModelForCausalLM, AutoTokenizer
persist_directory = 'chroma_langchain_db_txt_2'
# llm = AutoModelForCausalLM.from_pretrained(
#         r"D:\modelscope\hub\maple77\Qwen2-0.5B",
#         device_map="auto"
#     )


# 返回本地模型的嵌入。在存储嵌入和查询时都需要用到此嵌入函数。
def get_embedding():
    # nomic-embed-text: 一个高性能开放嵌入模型，具有较大的标记上下文窗口。
    # 安装：ollama pull nomic-embed-text:latest
    # 这个模型只有274M，但实际做嵌入和检索时，感觉比llama3这样的大模型还要好。
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


# 返回本地大模型
def get_llm():
    # temperature：用于控制生成语言模型中生成文本的随机性和创造性。
    # 当temperature值较低时，模型倾向于选择概率较高的词，生成的文本更加保守和可预测，但可能缺乏多样性和创造性。
    # 当temperature值较高时，模型选择的词更加多样化，可能会生成更加创新和意想不到的文本，但也可能引入语法错误或不相关的内容。
    # 当需要模型生成明确、唯一的答案时，例如解释某个概念，较低的temperature值更为合适；如果目标是为了产生创意或完成故事，较高的temperature值可能更有助于生成多样化和有趣的文本。
    return ChatOllama(model="llama3.2", temperature=0, verbose=True)


###这段代码只在第一次运行执行，生成RAG知识向量库，后续可以直接调用
# loader = TextLoader('book2.txt',encoding='utf-8')
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# vectordb = Chroma.from_documents(documents=texts, embedding=get_embedding(),persist_directory=persist_directory)
# vectordb.persist()


# # 对文本矢量化()

def get_retriever():
    # 使用本地矢量数据库创建矢量数据库实例
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=get_embedding())
    # 处理基于向量数据库的查询回答任务
    return vectorstore.as_retriever()


# 简单的问答
def ask(query):
    # 定义提示词
    # {context}将由retriever调用矢量数据库的相似结果填充
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the question."
        " If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    # {input}将由query填充
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = get_llm()

    retriever = get_retriever()

    # 将检索器纳入问答链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]


# 带有历史记录的聊天方法
# 显然，chat_history可以让模型更能“理解”上下文，做出更加妥帖的回答。
def chat(query, chat_history):
    # 构建检索器
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = get_llm()

    retriever = get_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 将检索器纳入问答链
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 调用链，返回结果
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return response["answer"]


# 测试ask方法
query1 = ("Please introduce the Three Gorges Group Research Institute to me in one paragraph.")

r = ask(query1)
print(r)

# # 测试chat方法
chat_history = ["The Three Gorges Science and Technology Research Institute is a research and innovation platform directly under the Three Gorges Group, established in June 2019, dedicated to promoting the group's technological innovation in core areas such as large-scale hydropower, new energy, and information technology. The research institute takes scientific research, application of technological achievements, and industrial promotion as its core tasks, aiming to provide technical support and innovation momentum for the high-quality development of the group. It is an important supporting unit for achieving technological leadership and strategic upgrading."]
ai_msg_1 = chat(query1, chat_history)
print(ai_msg_1)
chat_history.extend(
    [
        HumanMessage(content=query1),
        AIMessage(content=ai_msg_1),
    ]
)

# without rag
# I don't have information on the "Three Gorges Group Research Institute".
# However, I can tell you about the Three Gorges Corporation,
# a Chinese state-owned conglomerate that is involved in various industries such as energy,
# infrastructure, and real estate.
# The corporation was established in 2005 and is headquartered in Yichang, Hubei Province, China.


# with rag
# The Three Gorges Group Research Institute is a unified scientific research and innovation platform
# directly affiliated with the Three Gorges Group. Established in June 2019, it serves as a strategic
# unit focused on driving scientific and technological innovation across various core areas, including
# large hydropower, new energy, and information technology. The institute aims to provide a new engine
# for the high-quality development of the group by conducting research, applying scientific achievements,
# and promoting industrial applications.


