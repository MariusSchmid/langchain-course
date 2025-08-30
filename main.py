from dotenv import load_dotenv

load_dotenv()


from langchain_tavily import TavilySearch
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent


# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


tools = [TavilySearch()]
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(input={"input": "What is LangSmith?"})
    print(result)
    pass
    # llm = ChatGroq(temperature=0, model="llama3-8b-8192")
    # llm = ChatOllama(model="gpt-oss", temperature=0)


if __name__ == "__main__":
    main()
