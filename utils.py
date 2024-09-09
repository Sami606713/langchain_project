# ====================================Import Packages===============================#
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv
import os
# ================================================================================#

# Load Document
def load_documents(file_path:str):
    """
    This fun is responsible for loading the documents
    """
    # load the documents
    print("======Laoding Docuemts======")
    loader= PyPDFLoader(file_path)
    documents=loader.load()
    
    # Split the text into chunks
    print("======Splitting the text into chunks======")
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    doc_chunks=splitter.split_documents(documents)

    return doc_chunks

def create_vector_store(embedding,doc_chunks,vectcor_store_path:str):
    """
    This fun is responsible for creating the vector store
    """
    if not os.path.exists(vectcor_store_path):
        print("Creating Folder.....")
        os.makedirs(vectcor_store_path)
        vector_store=FAISS.from_documents(embedding=embedding,documents=doc_chunks)  
        
        print("===save the embedding in vector db===")
        vector_store.save_local(folder_path=vectcor_store_path)
        print("=======Embedding Save Successfully====")

        return vector_store
    else:
        print("Loading the embedding.....")
        vector_store=FAISS.load_local(folder_path=vectcor_store_path,embeddings=embedding,
                                      allow_dangerous_deserialization=True)
        return vector_store