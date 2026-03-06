from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.messages import SystemMessage, HumanMessage
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
import os
from rich import print as rprint

dir = "./10K/"
files = os.listdir(dir)
# print(files) 

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = PGVector(
    embeddings = embeddings,
    collection_name = '10k',
    connection = "postgresql+psycopg://adi_admin:adi_admin@localhost:5432/getbud"
)


if False:
    for f in files:
        loader = PyPDFLoader(dir+f)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
        all_splits = splitter.split_documents(documents)
        ids = vector_store.add_documents(all_splits)
        print(f"{len(ids)} documents added to vector store")


def fetch_documents(query, k = 4):
    fetched = vector_store.similarity_search(query, k = k)
    formatted_fetched = "\n\n\n".join(
        f"metadata : {d.metadata} \n\n content : {d.page_content}" for d in fetched 
        )
    return formatted_fetched

model = init_chat_model(model = "claude-haiku-4-5-20251001")

while True:
    input_query = input("basic-rag at your command : ")
    fetched_docs = fetch_documents(input_query, k = 4)
    print(fetched_docs)
    messages = [
        SystemMessage("You are a financial 10K chatbot. Answer the questions using the context and metadata provided and give citeations at the end of your answer."),
        SystemMessage(fetched_docs),
        HumanMessage(input_query)
    ]
    response = model.invoke(messages)
    print("\n\nRESPONSE\n\n")
    rprint(response.content)
    
