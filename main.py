from ctypes import Union
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
from callbacks import AgentCallbackHandler


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the input text"""
    text = text.strip("'\n").strip('"')
    return len(text)


def main():
    # print("Hello ReAct LAngChain!")
    tools = [get_text_length]

    template = hub.pull("hwchase17/react").template

    # print(template)
    prompt = PromptTemplate.from_template(template).partial(
        tools=tool_render.render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
        agent_scratchpad="",
    )

    llm = ChatGroq(
        temperature=0,
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        stop=["\Observation", "Observation:"],
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: log.format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of  'DOG' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = [t for t in tools if t.name == tool_name][0]
            tool_input = agent_step.tool_input
            observation = tool_to_use.run(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(f"Final Answer: {agent_step.return_values['output']}")


if __name__ == "__main__":
    main()
