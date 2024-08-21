from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader


@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="llama3.1", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
"أنت مساعد مخصص للإجابة على الأسئلة. استخدم المعلومات المسترجعة التالية للإجابة على السؤال. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف. يمكنك فقط التحدث باللغة العربية."
            ),
            ("human", "السؤال: {question}\nالسياق: {context}\nالإجابة:"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Get list of PDF and DOCX files
    file_paths = [
        os.path.join("docs/", f)
        for f in os.listdir("docs/")
        if f.endswith(".pdf") or f.endswith(".docx")
    ]

    # Load documents based on their file extension
    docs = []
    for file in file_paths:
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(file).load())
        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(file).load())

    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    # Chain
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(
            model="llama3.1",
        ),
    )
    print("DEBUG", vectorstore)
    retriever = vectorstore.as_retriever()
    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
