# ====================================Import Packages===============================#
import os
from utils import (load_documents, create_vector_store,
                   create_rag_chain,get_response,
                   load_previous_chat,stream_data,save_notes)
from langchain_cohere import CohereEmbeddings,ChatCohere
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
# ================================================================================#

# Steps for doing the project
# 1- Load the documents
# 2- Convert the text into chunks
# 3- Create Embedding
# 4- Save  the embedding in vector db
# 5- Make the retriever
# =====================================Set the title and Page configration=============================#
if "chat_history" not  in st.session_state:
    st.session_state['chat_history']=[]
# Set the page configuration
st.set_page_config(
    page_title="Document Question Answering",  # Title of the web page
    page_icon="🎥",  # Icon for the web page (you can use any emoji)
    layout="centered",  # Layout of the page (either 'centered' or 'wide')
    initial_sidebar_state="auto"  # Initial state of the sidebar ('auto', 'expanded', 'collapsed')
)

# Set the page title
st.title("Documents Quesion Answering")

#====================================Load Documents================================#
folder_path="Doc"
vector_store_path="db"

print("=====Loading Files======")
doc_chunks=load_documents(folder_path=folder_path)
print("========Total Chunks========\n",len(doc_chunks))
print("========First Chunk========\n",doc_chunks[0])


#====================================Create Vector Store================================#
try:
    print("=====Load the embedding model=======")
    embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])
    print("=====Creating the vector store=======")
    # temp chunk
    # doc_chunks="i am samiullah"
    vector_store=create_vector_store(embedding=embedding,doc_chunks=doc_chunks,
                                     vectcor_store_path=vector_store_path)
    
    # Create the retriever
    retriever=vector_store.as_retriever(search_type="similarity",
                                            search_kwargs={"k":3})
except Exception as e:
    print(e)

#====================================Chat================================#
# query="What is the operating system?"
# context=retriever.invoke(input=query)

# print("========Context========\n",context[0].page_content)

# Load the llm
llm=ChatCohere(cohere_api_key=os.environ['cohere_api'])

# create a rag chain
rag_chain=create_rag_chain(llm=llm,retriever=retriever)

# Get response
prompt=st.chat_input(placeholder="Enter you question? ")

with st.container(border=False):
    load_previous_chat()

if prompt:
    st.write(f'💬 You: {prompt}') 
    save_notes("Doc/Note_os.docx",f"Q: {prompt+'\n'}")
    response = get_response(rag_chain, prompt)
    st.write_stream(stream_data(response))

    # Save notes
    save_notes("Doc/Note_os.docx",f"Ans: {response+'\n'}")