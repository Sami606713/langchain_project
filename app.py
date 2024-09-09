# ====================================Import Packages===============================#
import os
from utils import load_documents, create_vector_store
from langchain_cohere import CohereEmbeddings,ChatCohere
from dotenv import load_dotenv
load_dotenv()
# ================================================================================#

# Steps for doing the project
# 1- Load the documents
# 2- Convert the text into chunks
# 3- Create Embedding
# 4- Save  the embedding in vector db
# 5- Make the retriever

#====================================Load Documents================================#
file_path="Doc/OS.pdf"
vector_store_path="db"


# doc_chunks=load_documents(file_path=file_path)
# print("========Total Chunks========\n",len(doc_chunks))
# print("========First Chunk========\n",doc_chunks[0])


#====================================Create Vector Store================================#
try:
    print("=====Load the embedding model=======")
    embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])
    print("=====Creating the vector store=======")
    # temp chunk
    doc_chunks="i am samiullah"
    vector_store=create_vector_store(embedding=embedding,doc_chunks=doc_chunks,
                                     vectcor_store_path=vector_store_path)
    
    # Create the retriever
    retriever=vector_store.as_retriever(search_type="similarity",
                                            search_kwargs={"k":3})
except Exception as e:
    print(e)

#====================================Chat================================#
query="What is the operating system?"
context=retriever.invoke(input=query)

print("========Context========\n",context[0].page_content)