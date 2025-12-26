from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2',
    task = "text-generation"

)
model = ChatHuggingFace(llm=llm)

st.title("💬 SamaD")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant. You must always respond in English.")]


user_input = st.chat_input("Ask anything")

if user_input:
    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )
    result = model.invoke(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=result.content))

count = 0

for msg in st.session_state.chat_history:
    if count != 0:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)
    count = count + 1
        