from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model = "claude-haiku-4-5-20251001",
    temperature = 0.5
)


def way1():
    # invoke the model with simple text prompt
    inputs = "hey what is up?"

    response = model.invoke(inputs)

    print("\n",response,"\n")
    print(response.content)


def way2():
    # invoke the model using a conversation
    inputs = {'messages':[
        {'role':'user', 'content':'Hey whats up?'}
    ]}

    response = model.invoke(inputs['messages'])

    print("\n",response,"\n")
    print(response.content)

def way3():
    # invoke using list of message objects
    inputs = {'messages':[
        SystemMessage("you are a very sarcastic bot, always be very sarcastic"),
        HumanMessage("what are you doing right now?")
    ]}

    response = model.invoke(inputs["messages"])

    print("\n",response,"\n")    
    print(response.content)

def way4():
    # streaming outputs
    inputs = {'messages':[
        SystemMessage("you are a very sarcastic bot, always be very sarcastic"),
        HumanMessage("what are you doing right now?")
    ]}

    for chunk in model.stream(inputs["messages"]):
        print(chunk.text, end = '', flush = True)


if __name__ == "__main__":
    # print("\n\nWAY 1\n")
    # way1()
    # print("\n\n")

    # print("\n\nWAY 2\n")
    # way2()
    # print("\n\n")

    # print("\n\nWAY 3\n")
    # way3()
    # print("\n\n")

    print("\n\nWAY 4\n")
    way4()
    print("\n\n")