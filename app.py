from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PDFMinerLoader
import boto3
import os
import streamlit as st
import tempfile

# AWS environment setup
os.environ["AWS_PROFILE"] = "test-user-profile"

# Bedrock client initialization
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-instant-v1"

# LangChain Bedrock configuration
llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.7}
)

# Streamlit App UI
st.set_page_config(
    page_title="Pseudo Chatbot",
    page_icon="📄💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📄💬 Pseudo Chatbot")
st.markdown("""
Upload a document and ask questions within its scope.
""")

# Sidebar configuration
st.sidebar.header("Configure Chatbot")
temperature = st.sidebar.slider("🌡️ Creativity Level (Temperature):", 0.0, 1.0, 0.7, step=0.1)

# Document Upload
uploaded_file = st.file_uploader("📂 Upload a document (PDF or TXT)", type=["pdf", "txt"])

def process_document(uploaded_file):
    """Loads and processes the document into a retrievable format."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith(".pdf") else ".txt") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PDFMinerLoader(temp_file_path)
    else:
        loader = TextLoader(temp_file_path)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    os.remove(temp_file_path)  # Clean up after processing
    return vectorstore

if uploaded_file:
    st.success("Document uploaded successfully! Processing...")
    vectorstore = process_document(uploaded_file)
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    # Initialize chat history as empty
    chat_history = []

    # User input
    query = st.text_area("🔍 Ask a question about the document:")
    if st.button("💬 Get Response"):
        if query.strip():
            with st.spinner("Retrieving answer..."):
                # Pass the query along with chat history as input
                response = qa_chain.run({"question": query, "chat_history": chat_history})
                st.success("Here is the response:")
                st.write(response)

                # Update chat history with the current query and response
                chat_history.append((query, response))
        else:
            st.error("Please enter a question before submitting.")

