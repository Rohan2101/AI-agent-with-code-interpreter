from dotenv import load_dotenv
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

if __name__=="__main__":
    prompt = ChatPromptTemplate(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(),multiply]

    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-70b-versatile",
    )

    agent = create_tool_calling_agent(llm,tools,prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools)

    res = agent_executor.invoke(
        {"input": "what is the weather in dubai right now? compare it with San Fransisco, output should in in celsious",}
    )
    print(res)
