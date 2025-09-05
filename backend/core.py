from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

from langchain_groq import ChatGroq
import os


def run_llm(query: str) -> str:
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    docsearch = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    chat = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_document_chain
    )
    result = qa.invoke(input={"input": query})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    query = "What is LangChain?"
    result = run_llm(query)
    print(result["answer"])
