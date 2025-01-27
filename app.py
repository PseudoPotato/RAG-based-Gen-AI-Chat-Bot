from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

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
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)

# Chatbot logic
def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})
    return response


# Streamlit App UI
st.set_page_config(
    page_title="Pseudo Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title("💬 Pseudo Chatbot")
st.markdown(
    """
    Welcome to the Pseudo Chatbot app!  
    Select a language, type your question, and get an intelligent response.  
    """
)

# Sidebar for Input Options
st.sidebar.header("Configure Chatbot")
language = st.sidebar.selectbox("🌐 Choose Language:", ["English", "Spanish"])
max_length = st.sidebar.slider("🔢 Response Length (Max Tokens):", 50, 2000, 500, step=50)
temperature = st.sidebar.slider("🌡️ Creativity Level (Temperature):", 0.0, 1.0, 0.7, step=0.1)

# User Input Section
st.header("🤔 Ask Your Question")
freeform_text = st.text_area("Type your question below:", max_chars=200, placeholder="What's on your mind?")

# Submit Button
if st.button("💬 Get Response"):
    if freeform_text.strip():
        with st.spinner("Generating response..."):
            # Update model parameters dynamically
            llm.model_kwargs["max_tokens_to_sample"] = max_length
            llm.model_kwargs["temperature"] = temperature

            # Get chatbot response
            response = my_chatbot(language, freeform_text)
            st.success("Here is the response:")
            st.write(response['text'])
    else:
        st.error("Please enter a question before submitting.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Vedant")
