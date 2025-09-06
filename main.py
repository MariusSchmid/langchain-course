from dotenv import load_dotenv

load_dotenv()


from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.agents import create_tool_calling_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from langchain_tavily import TavilySearch
from schemas import AgentResponse


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant that helps people find information.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearch()]
    llm = ChatGroq(
        temperature=0,
        model="llama-3.3-70b-versatile",
        verbose=True,
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res = agent_executor.invoke(
        {
            "input": "What is the weather in dubay right now? compare with san francisco, output in celsius. Which tool did you u"
        }
    )
    print(res)

    pass


if __name__ == "__main__":
    main()
