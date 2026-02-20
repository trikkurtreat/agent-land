from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import Annotated, TypedDict
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from rich import print as rprint

model = init_chat_model(
    model = "claude-haiku-4-5-20251001",
    temperature = 0.1
)

@tool("add", description = "returns the sum of two numbers a and b")
def add(a : int, b : int) -> int:
    return a+b

@tool("subtract", description = "returns the difference of two numbers a and b")
def subtract(a : int, b : int) -> int:
    return a-b

@tool("multiply", description = "returns the product of two numbers a and b")
def multiply(a : int, b : int) -> int:
    return a*b

@tool("divide", description = "returns the division of two numbers a and b")
def divide(a : int, b : int) -> float:
    return a/b


tools = [add, subtract, multiply, divide]
tools_by_name = {tool.name : tool for tool in tools}

model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages : Annotated[list[AnyMessage], operator.add]
    llm_calls : int


def llm_call(state : dict):
    inputs = [SystemMessage(content = "You are a helpful math agenet that helps perform arithmetic on numbers.")] + state["messages"]
    outputs = model_with_tools.invoke(inputs)
    return {
        "messages" : [outputs], 
        "llm_calls" : state.get("llm_calls", 0) + 1
    }

def tool_node(state : dict):
    '''this node handles tool calls'''
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content = observation, tool_call_id = tool_call["id"]))
    return {"messages":result}

def should_continue(state : MessagesState) -> Literal["tool_node", "__end__"]:
    '''decide if we should continue or end based on weather the llm made a tool call'''
    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return "__end__"
    
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call",llm_call)
agent_builder.add_node("tool_node",tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

from IPython.display import Image, display
png_data = agent.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("graph has been saved.")

messages = [HumanMessage(content="please add three and 4, then add 3 and 3, then find the product of these 2")]
messages = agent.invoke({"messages":messages})
for m in messages["messages"]:
    print("\n\n")
    rprint(m)