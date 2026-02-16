from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from rich import print as rprint

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. 
If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


@dataclass
class Context:
    '''custom runtime context schema'''
    user_name : str

@tool
def get_weather_for_location(city : str) -> str:
    '''returns the weather for the given location'''
    return f"it is always sunny in {city}!"

@tool
def get_user_location(runtime : ToolRuntime[Context]) -> str:
    '''retrieve the user's location using runtime context'''
    if runtime.context.user_name == "Aditya":
        return "Toronto"
    else:
        return "Buffalo"
    
@dataclass
class ResponseFormat:
    '''response schema for agent'''
    # A punny response (always required)
    punny_response : str
    # additional weather info if available
    additional_info : str | None = None


model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model
    , system_prompt = SYSTEM_PROMPT
    , tools = [get_weather_for_location, get_user_location]
    , context_schema = Context
    , response_format = ToolStrategy(ResponseFormat)
    , checkpointer = checkpointer
)

config = {"configurable" : {"thread_id" : "123"}}

## call the model using the same thread twice so that saved convo memory in the checkpointer is reused

# response = agent.invoke({"messages" : [{"role" : "user", "content" : "whats the weather here?"}]}
#                         , config = config
#                         , context = Context(user_name = "Aditya"))

# print(response["structured_response"])


# response = agent.invoke({"messages" : [{"role" : "user", "content" : "awesome! what did you mean by under the weather?"}]}
#                         , config = config
#                         , context = Context(user_name = "Aditya"))

# print(response["structured_response"])


## visualize the state graph changes as the different tool calls happen
inputs = {"messages" : [{"role" : "user", "content" : "whats the weather here?"}]}
for chunk in agent.stream(inputs, config = config, context = Context(user_name="Aditya"), stream_mode="updates"):
    rprint(chunk)
    print("\n\n")