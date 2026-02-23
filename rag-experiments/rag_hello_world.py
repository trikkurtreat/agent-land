from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
#from sklearn.metrics.pairwise import cosine_similarity

# test_documents = [
#     Document(page_content="Hi this is page 1", metadata = {"source":"my document", "page":1}),
#     Document(page_content="HI this is another document 2", metadata={"source":"my document 2", "page":2})
# ]

filepath = "./test_docs/nke-10k-2023.pdf"
loader = PyPDFLoader(filepath)

documents = loader.load()

print("Number of documents : ", len(documents))
# print(documents[0].page_content[:200])
# print(documents[0].metadata)

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                          chunk_overlap = 200,
                                          add_start_index = True)

all_splits = splitter.split_documents(documents)

print("Total number of splits : ", len(all_splits))

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

# q1 = "There are seven kids in my customer engagement class"
# q2 = "there are several Kid objects in the class called CustomerEngagement"
# q3 = "there are several kids studying in the customer engagement class"

# v1 = embeddings.embed_query(q1)
# v2 = embeddings.embed_query(q2)
# v3 = embeddings.embed_query(q3)

# print(len(v3))

# cosine_sim_mat = cosine_similarity([v1, v2, v3], [v1,v2,v3])
# print(cosine_sim_mat)

vector_store = PGVector(
    embeddings = embeddings,
    collection_name = "mydocs",
    connection = "postgresql+psycopg://adi_admin:adi_admin@localhost:5432/getbud"
)

## uncomment below line to delete collection
#vector_store.delete_collection()

## uncomment below line to add embeddings into vector store (caution : could duplicate)
#ids = vector_store.add_documents(documents = all_splits)


### 1
# print("\n\nHow many distribution centres does nike have in the US?\n")
# results = vector_store.similarity_search("How many distribution centres does nike have in the US?")
# print(results[0])


### 2
# print("\n\nHow many distribution centres does nike have in the US?\n")
# results = vector_store.similarity_search_with_score("How many distribution centres does nike have in the US?")
# print("\n\nresult 1")
# print(results[0])
# print("\n\nresult 2")
# print(results[1])


### 3
print("\n\nwhen was nike incorporated?\n")
emb = embeddings.embed_query("when was nike incorporated?")
results = vector_store.similarity_search_by_vector(emb)
print(results[0])




