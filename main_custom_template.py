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
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import os


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():

    embeddings = PineconeEmbeddings(model="multilingual-e5-large")

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
    print(os.environ["GROQ_API_KEY"])
    query = "What is the main topic of the blog post?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings,
    )

    template = """ Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks you sir for asking!" at the end of the answer.

    {context}

    Question: {question}
    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)


if __name__ == "__main__":
    main()
