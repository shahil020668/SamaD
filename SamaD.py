from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task = "text-generation"

)
model = ChatHuggingFace(llm=llm)

st.title("💬 Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


user_input = st.chat_input("Ask anything")

if user_input:
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )
    result = model.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=result.content))


for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
        