from dotenv import load_dotenv

load_dotenv()


from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
# llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
llm = ChatGroq(temperature=0, model="meta-llama/llama-4-maverick-17b-128e-instruct")


output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(llm, tools, prompt=react_prompt_with_format_instructions)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# same as output_parser.parse(result["output"])
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(input={"input": "What is Langsmith?"})
    print(result.answer)
    print(result.sources)
    # print(agent_response.answer)
    # print(agent_response.sources)
    pass


if __name__ == "__main__":
    main()
