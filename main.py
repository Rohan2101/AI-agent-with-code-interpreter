from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from numpy.f2py.crackfortran import verbose
from langchain.tools import Tool

load_dotenv()

def main():
    print("Starting.....")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to excute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer. 
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatGroq(
            temperature = 0,
            groq_api_key="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",
            model_name="llama-3.1-70b-versatile",
        ),
        tools=tools,
    )
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
    agent_executor.invoke(
        input={
            "input":"""Generate and save in current working directory 15 QRcodes that point to
            www.udemy.com/course/langchain, you have the qrcode package installed already."""
        }
    )
    csv_agent = create_csv_agent(
        llm=ChatGroq(
            temperature=0,
            groq_api_key="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",
            model_name="llama-3.1-70b-versatile",
        ),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    # csv_agent.invoke(
    #     input={"input":"how many columns are there in the file episode_info.csv"}
    # )
    #
    # csv_agent.invoke(
    #     input={"input":"Which writer wrote the most number of episodes? How many episodes did he write?"}
    # )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return agent_executor.invoke({"input":original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""Useful when you need to transform natural language to python and execute the python code,
            returning the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""Useful when you need to answer questions over episode_info.csv file,
            takes an input the entire question and returns the answer after running pandas calculations"""
        )
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatGroq(
            temperature=0,
            groq_api_key="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",
            model_name="llama-3.1-70b-versatile",
        ),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent,tools=tools,verbose=True)
    # print(grand_agent_executor.invoke(
    #     {"input":"which season has the most number of episodes?"}
    # ))


    print(
        grand_agent_executor.invoke(
            {"input":"Generate and save in current working directory 15 QRcodes that point to www.udemy.com"}
        )
    )
if __name__=="__main__":
    main()