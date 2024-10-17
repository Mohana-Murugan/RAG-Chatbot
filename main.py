import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

import warnings
import uuid
import time

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
groq_api_key = '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'
qdrant_url = "https://44a82cdb-7d22-46a7-b2f3-c948f4ff16ec.europe-west3-0.gcp.cloud.qdrant.io:6333"
qdrant_api_key = "S8AkQRPHd57btMMsDt_5dlM9tSOUemKeZcHX1LtX6jIm4uI8W35INA"

warnings.filterwarnings("ignore")

# Function to load text from a PDF file
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split the document and generate embeddings
def process_pdfs(uploaded_files):
    docs = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        text = load_pdf(uploaded_file)
        docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(docs)

    # Generate embeddings
    points = get_embedding(text_chunks)

    # Store embeddings in Qdrant
    vectorstore = Qdrant.from_documents(
        documents=text_chunks,
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="Chatbot-HR",
        force_recreate=True
    )

    return vectorstore

# Function to generate embeddings for text chunks
def get_embedding(text_chunks, model_name='all-MiniLM-L6-v2'):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    points = []
    
    embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in text_chunks])

    for idx, chunk in enumerate(text_chunks):
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": embeddings[idx],
            "payload": {"text": chunk.page_content, "source": chunk.metadata["source"]}
        })

    return points

# Chatbot interaction function
def chat_with_bot(vectorstore, query):
    retriever = vectorstore.as_retriever()

    # Define the LLM and tools for the agent
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    
    pdf_tool = create_retriever_tool(
        retriever=retriever,
        name="pdf_search",
        description="Search for information about Company policy, HR, and leave"
    )

    # Define the prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say "answer is not available in the context."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    {agent_scratchpad}
    """)

    # Define the agent and executor
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    agent = create_openai_tools_agent(
        llm=llm,
        tools=[pdf_tool],
        prompt=prompt
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=[pdf_tool], verbose=False)

    # Run the query and return the result
    response = agent_executor.invoke({
        "input": query,
        "context": "",  # Populate if required
        "agent_scratchpad": ""  # Populate if required
    })

    return response['output']


# Streamlit UI
st.title("Let's Chat with your data 🤖")

# File uploader to allow users to upload PDFs
uploaded_files = st.file_uploader("Upload your data as pdf files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write("Processing your data...")
    
    # Process PDFs and create vector store
    vectorstore = process_pdfs(uploaded_files)
    
    st.write("Your data has been processed and stored in Qdrant DB. You can now interact with the chatbot.")

    # Text input for user query
    query = st.text_input("May I have your Query?")

    # Button to get chatbot response
    if st.button("Get Answer") and query:
        st.write("Getting answer from the chatbot...")
        
        start_overall = time.time()
        
        try:
            # Chatbot interaction
            response = chat_with_bot(vectorstore, query)
            response_time_overall = time.time() - start_overall
            
            # Display chatbot response
            st.write(response)
            st.markdown(f"<p style='color:blue;'>Response Time: {response_time_overall:.2f} seconds</p>", unsafe_allow_html=True)
        
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")