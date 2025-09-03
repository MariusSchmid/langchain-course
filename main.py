from dotenv import load_dotenv

load_dotenv()

# from langchain.agents import tool
# from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# import langchain.tools.render as tool_render
# import langchain.agents.format_scratchpad.log as log
# from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain.schema import AgentAction, AgentFinish
# from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain

import os


def main():

    # embeddings = PineconeEmbeddings(model="multilingual-e5-large")

    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        # other params...
    )
    print(os.environ["GROQ_API_KEY"])
    query = "What is the main topic of the blog post?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result)


if __name__ == "__main__":
    main()
