import json
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
import pandas as pd
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List
from datetime import date


# read the transaction data from csv
def read_transactions(file_path = "./transactions.csv"):
    df = pd.read_csv(file_path)
    return df

# write dict with lists ({date:[], thing:[], amount:[], category:[]}) into a csv 
def add_transactions(data, file_path = "./transactions.csv"):
    df = read_transactions(file_path=file_path)
    new_df = pd.DataFrame(data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(file_path, index = False)


class TransactionsSchema(BaseModel):
    '''schema for the llm to understand the required inputs for insert_new_transactions'''
    date: List[str] = Field(description="Dates of transactions (YYYY-MM-DD)")
    thing: List[str] = Field(description="Names of items/services/shops of transactions")
    amount: List[float] = Field(description="cost of each transaction")
    category: List[str] = Field(description="spending categories one of [grocery, food, rent, utils, internet, transit, misc]")


@tool("get_category_totals", description="returns a dictionary with each category and its total spend, from the transaction list")
def get_category_totals():
    df = read_transactions()
    category_totals = df.groupby("category")["amount"].sum().to_dict()
    return category_totals

@tool("insert_new_transactions", args_schema=TransactionsSchema, description='''inserts new transactions into transaction database''')
def insert_new_transactions(date, thing, amount, category) -> str:
    new_ts = {
                "date":date,
                "thing":thing,
                "amount":amount,
                "category":category
                }
    add_transactions(new_ts)
    return "Transactions have been added successfully"

def get_agent():
    model = init_chat_model(
    model = "claude-haiku-4-5-20251001",
    temperature = "0.5"
    )
    agent = create_agent(
        model = model, 
        tools = [get_category_totals, insert_new_transactions],
        system_prompt = '''you are a helpful chatbot that helps maintain the users transactions in a database.
                            help to add transactions into the database using the tools available. call the tools 
                            by passing data in the required format that is specified.''' + f" todays date : {str(date.today())}"
    )
    return agent

def read_and_add_transactions():
    df = read_transactions()
    print(df.head())
    add_transactions({
                    "date":["2026-03-02"],
                    "thing":["rogers"],
                    "amount":[79.2],
                    "category":["internet"]})
    df = read_transactions()
    print(df.head())

if __name__ == '__main__':
    #read_and_add_transactions()
    #get_category_totals()
    getbud = get_agent()
    while(True):
        #saywhat = "add this transaction to the database, I spend 30.56 dollars on milk day before yesterday"
        saywhat = input("\n describe the transaction/s master \n")
        inputs = {"messages":[{"role":"user", "content":saywhat}]}
        response = getbud.invoke(inputs)
        print(response["messages"][-1].content)