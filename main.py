from dotenv import load_dotenv
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()


def main():
    print("Hello, LangChain Course!")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # tools = [PythonREPLTool()]
    # agent = create_react_agent(
    #     ChatGroq(temperature=0, model="llama-3.3-70b-versatile"),
    #     tools,
    #     prompt=prompt,
    #     # verbose=True,
    # )
    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True
    # )
    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    csv_agent = create_csv_agent(
        ChatGroq(temperature=0, model="llama-3.3-70b-versatile"),
        "episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    csv_agent.invoke(
        input={"input": "which wrote the most episodes? how many episodes?"}
    )
    # csv_agent.invoke(input={"input": "How many episodes are there?"})


if __name__ == "__main__":
    main()
