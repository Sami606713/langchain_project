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
import time
import os
# ================================================================================#

# Load Document
def load_documents(folder_path:str):
    """
    This fun is responsible for loading the documents
    """
    # load the documents
    print("======Laoding Docuemts======")
    files= os.listdir(folder_path)
    documents=[]
    for file in files:
        if file.endswith(".pdf"):
            print(f"========={file}========")
            curr_doc=os.path.join(folder_path,file)
             # load the document
            print(f"Loading Document===={file}")
            text_loader=PyPDFLoader(file_path=curr_doc)
            docs=text_loader.load()

            # Adding meta data to the document
            print("======Adding Meta data")
            for doc in docs:
                doc.metadata={"source":file,"type":"PDF","name":file.split()[0]}
                documents.append(doc)

    
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


def create_rag_chain(llm,retriever):
    try:
        # make a sytem message
        system_message = (
            "You are given a chat history and a new user question. "
            "Your task is to review both the chat history and the new question. "
            "If the new question references context from the chat history, reformulate it "
            "to ensure it is standalone and comprehensible. If the new question does not "
            "need any modification and is clear on its own, return it as is. "
            "Do not answer the question—focus solely on reformulating it for clarity if necessary."
        )

        # cousomize the prompt
        coustomize_system_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_message),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # set the history aware retriever
        history_aware_retriever=create_history_aware_retriever(
            llm=llm,retriever=retriever,prompt=coustomize_system_prompt
        )

        # make a question_answer prompt
        q_a_prompt = (
            "You are an expert assistant for Opreating System Question Anwering. "
            "Please use the following context to formulate your response to the user’s query. "
            "If the provided context does not allow you to answer the question, respond with 'I don't know.' "
            "if question is not related to math and calculus subject answer should be return in this format.\n"
            "Defination:\n define with in 3 lines.\nExplanation:\n Explain them in bullet point format.\nExample\n"
            "Give the answer in word doc format."
            "Note that answer is not too long keep the answer consise and clear and make its easier for preparation."
            "{context}"
        )


        # coustoize the Q&A prompt
        final_qa_prompt=ChatPromptTemplate(
            [
                ("system",q_a_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # create a document stuff
        question_answer_chain=create_stuff_documents_chain(llm,final_qa_prompt)

        # make the rag chain
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        return rag_chain
    except Exception as e:
        return str(e)


def get_response(rag_chain,query):
    """
    This fun is responsible for continuing the chat
    
    Args:
        rag_chain (object): The rag chain object
        query (str): The user query
    return:
        response (str): The response from the chatbot
    """
    try:
        st.session_state['chat_history'].append(HumanMessage(content=query))
        response=rag_chain.invoke({
            "input":query,'chat_history':st.session_state['chat_history']
        })
        st.session_state['chat_history'].append(SystemMessage(content=response['answer']))

        return response['answer']
            
    except Exception as e:
        print(str(e))


def load_previous_chat():
    for i in range(len(st.session_state['chat_history'])):
        if i%2==0:
            st.write(f'💬 You: {st.session_state['chat_history'][i].content}') 
        else:
            st.write(f'🤖 Bot: {st.session_state['chat_history'][i].content}')  

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def save_notes(file,respone):
    """
    This fun is responsible for saving the notes
    """
    with open(file,"a+") as f:
        f.write(f"{respone}")