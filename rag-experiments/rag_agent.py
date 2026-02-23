from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from rich import print as rprint
import bs4

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()
print("length of document : ", len(docs[0].page_content))

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
all_splits = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

vector_store = PGVector(
    embeddings=embeddings,
    collection_name='mycoll2',
    connection='postgresql+psycopg://adi_admin:adi_admin@localhost:5432/getbud'
)

# _ = vector_store.add_documents(all_splits)

@tool("retrieve_documents", description="retrieves the most relevant 2 docuements from the vector store given a query", response_format='content_and_artifact')
def retrieve_documents(query : str):
    results = vector_store.similarity_search(query, k = 2)
    serialized = "\n\n".join(
        (f"\n source : {doc.metadata} \n content : {doc.page_content} \n")
        for doc in results
    )
    return serialized, results

model = init_chat_model(model = 'claude-haiku-4-5-20251001', temperature = '0.5')
tools = [retrieve_documents]

agent = create_agent(model, tools, system_prompt='You are a helpful agent. Use the tool to fetch documents to help you answer the users query.')

inputs = {'messages':[HumanMessage('what are some limitations of llm-based agents?')]}

response = agent.invoke(inputs)

rprint(response)