from dotenv import load_dotenv

load_dotenv()

from langchain.agents import tool
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

import langchain.tools.render as tool_render
import langchain.agents.format_scratchpad.log as log
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
from langchain_community.vectorstores import FAISS

import os


def main():
    print("Hello, LangChain with Groq!")
    pdf_file = "react-paper.pdf"

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents)
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=os.environ["GROQ_API_KEY"],
        # other params...
    )
    # print(os.environ["GROQ_API_KEY"])
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # # print(result)

    # vectorstore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"],
    #     embedding=embeddings,
    # )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_documents_chain,
    )

    query = "Give me the gist of ReAct in 3 sentences"
    result = retrival_chain.invoke({"input": query})
    print(result["answer"])


if __name__ == "__main__":
    main()
