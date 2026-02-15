from langchain.agents import create_agent
from rich import print as rprint

def get_weather(city : str) -> str:
    '''function that returns weather in a given city'''
    return f"its always sunny in {city}"

agent = create_agent(
    model = "claude-haiku-4-5-20251001"
    , tools = [get_weather]
    , system_prompt = "you are a helpful assistant"
)

inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}

# # prints all the updates to the state graph in real time
# for chunk in agent.stream(inputs, stream_mode="updates"):
#     rprint(chunk)

# calls agent and prints the result (this shows the conversation between the agent, the tool and back)
result = agent.invoke(inputs)
rprint(result)
print("\n\n")
result['messages'][-1].pretty_print()