import PyPDF2
from huggingface_hub import InferenceClient
import streamlit as st
from streamlit_chat import message
from langchain.memory import ConversationBufferWindowMemory
import random

aa=["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","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"]
bb=random.choice(aa)
# Open and read the PDF file
with open('am_2.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    all_text = "".join(page.extract_text() for page in reader.pages)

# System message setup
data = f': I am ETChat, an advanced AI Technology of ETC. Tariff link: https://www.etconline.in/images/PAN%20India%20Tariff%202024.pdf . {all_text}.'
system_content = (
    "You are a helpful AI assistant. You will be given knowledge and need to answer questions strictly based on that document. "
    "If any question is asked out of the text, just answer: 'I apologize for the inconvenience, but I only have information about ETC. "
    "For further details, please visit our website at https://www.etconline.in/'. "
    "Don't reveal that you are answering based on the document provided. Here is your data: " + data
)

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=bb)  # Replace with your Hugging Face API key

# Streamlit UI for chatbot with title and logo aligned
col1, col2 = st.columns([0.8, 0.2])  # Adjust width: 80% text, 20% logo
with col1:
    st.markdown("<h4 style='text-align: left; font-family: Lora, serif;'>ETChat</h4>", unsafe_allow_html=True)  # Left-aligned title with Georgia font

with col2:
    st.image("ETC.png", width=100)  # Right-aligned logo (replace with actual image path)

# Initialize session state for storing chat history and memory
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=100, return_messages=True)

# Display chat history
for i, chat in enumerate(st.session_state['chat_history']):
    message(chat['content'], is_user=chat['role'] == 'user', key=f"message_{i}")

# User input for question
user_prompt = st.chat_input("Ask anything about me...")

if user_prompt:
    # Store user's question
    st.session_state['chat_history'].append({"role": "user", "content": user_prompt})
    message(user_prompt, is_user=True, key=f"user_message_{len(st.session_state['chat_history'])}")

    # Generate response using Hugging Face API
    assistant_response = client.text_generation(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # Ensure model availability on HF
        prompt=f"{system_content}\nUser: {user_prompt}\nAssistant:",
        max_new_tokens=500
    )
    assistant_response1 = assistant_response.split("</think>")
    assistant_response2 = assistant_response1[1] if len(assistant_response1) > 1 else assistant_response

    # Store and display response
    st.session_state['chat_history'].append({"role": "assistant", "content": assistant_response2})
    message(assistant_response2, key=f"assistant_message_{len(st.session_state['chat_history'])}")
