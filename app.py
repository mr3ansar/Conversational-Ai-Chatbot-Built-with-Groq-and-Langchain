import os
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
# there was this problem with the env api key not loading so Ai told be to use this
from pydantic import SecretStr

#loading Api key

ENV_GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()

#Streamlit Page Config
st.set_page_config(
    page_title="Groq API with Langchain",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Conversational AI Chatbot with Groq API and Langchain")
st.caption("This is a simple chatbot built using the Groq API and Langchain. It maintains conversational context and provides responses based on user input.")

#Sidebar Controls

with st.sidebar:
    st.header("⚙ Controls")
    # API Key Input
    api_key_input = st.text_input(
        "Groq API Key (optional)",
        type="password"
    )

    GROQ_API_KEY = api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY
    model_name = st.selectbox(
    "Select Groq Model",
    options = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "groq/compound-mini",
        "moonshotai/kimi-k2-instruct-0905"
    ],
    index=1
    )
# Temperature Slider for Creativity
    temperature = st.slider(
        "Response Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1
    )
    max_tokens = st.slider(
        "Max Tokens (Reply Length)",
        min_value=64,
        max_value=1024,
        value=256,
        step=64
    )
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be clear, correct and concise."

    TONE_PRESETS = {
        "Default": DEFAULT_SYSTEM_PROMPT,
        "Customer Care": """You are emotionally intelligent and act like a professional customer support agent.
⭐Acknowledge user concerns.
⭐Show empathy before giving solutions.
⭐Be calm, respectful, and solution-focused.""",
        "Formal": "You are a highly professional and formal AI assistant. Respond seriously and respectfully.",
        "Friendly": "You are a friendly and warm AI assistant. Use casual, engaging language.",
        "Funny": "You are a humorous AI assistant. Add light jokes where appropriate.",
        "Professional": "You are an expert consultant AI. Provide structured and precise answers."
    }
# Max Tokens Slider for controlling response length
    
    
# Initialize system prompt and last tone in session state if not already present
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if "tone" not in st.session_state:
        st.session_state.tone = "Default"
    # Reset System Prompt Button
    if st.button("🔄 Reset System Prompt"):
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.session_state.tone = "Default"
        st.rerun()
    # Response Tone Selector
    tone = st.selectbox(
        "Select Response Tone:",
        list(TONE_PRESETS.keys()),
        key="tone"
    )
 
    # Update prompt if tone changes
    st.session_state.system_prompt = TONE_PRESETS[st.session_state.tone]

    
    # SYSTEM PROMPT TEXT AREA
    system_prompt = st.text_area(
        "System Prompt (Rules for the bot)",
        key="system_prompt",
        height=140
    )
    
    

    #typing effect toggle
    typing_effect = st.checkbox("Enable typing effect", value=True)
    st.divider()
    # Reset chat history button
    if st.button(" 🧹 Clear Chat "):
        st.session_state.pop("history_store",None)
        st.session_state.pop("download_cache", None)
        st.rerun()

## API Key Guard, to check if the key is present before allowing the user to interact with the chatbot. This prevents errors and guides the user to set up their environment correctly.
# if not GROQ_API_KEY:
#     st.secrets.write(st.secrets)
#     st.error("🔑 Groq API Key is missing. Add it in your .env or paste it in the sidebar")
#     st.stop()

# Initialize chat history in session state if not already present
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

SESSION_ID = "default_session"
# this function takes a session_id as input and returns the chat history for that session. If the session_id is not present in the history_store, it creates a new InMemoryChatMessageHistory object and stores it in the history_store with the session_id as the key with the value of the new InMemoryChatMessageHistory object which is used to store the messages of the chat history in memory for that particular session. 

def get_history(session_id:str)-> InMemoryChatMessageHistory: # this function takes a session_id as input and returns the chat history for that session. If the session_id is not present in the history_store, it creates a new InMemoryChatMessageHistory object and stores it in the history_store with the session_id as the key with the value of the new InMemoryChatMessageHistory object which is used to store the messages of the chat history in memory for that particular session. 
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]
# Building LLM
llm = ChatGroq(
    api_key = SecretStr(GROQ_API_KEY),
    model = model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

# Making the Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}"),
    ]
)

# Creating a Chain of prompt , llm and StrOutputParser
#this will take the userinput then show the prompt with the system prompt and chat history then pass it down to llm to generate the response and parse the output using the StrOutputParser to get the string response in the chatbot
chain = prompt | llm | StrOutputParser()
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history" # this is the key that will be used to pass the chat history to the prompt when rendering it. The RunnableWithMessageHistory class is a wrapper that allows us to easily manage the chat history and pass it to the chain when invoking it. It takes care of retrieving the chat history for the current session, rendering the prompt with the history, and updating the history with the new messages after generating a response.
)

history_obj = get_history(SESSION_ID) 

for msg in history_obj.messages:
    role = getattr(msg,"type","")
    if role == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
# User Input + Model Response
user_input = st.chat_input("Type your message....")
if user_input:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response_text = chat_with_history.invoke(
                {"input": user_input,"system_prompt": system_prompt},
                config = {"configurable":{"session_id": SESSION_ID}},
            )
        except Exception as e:
            st.error(f"Model Error: {e}")
            response_text = ""

        if typing_effect and response_text:
            typed = ""
            for ch in response_text:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)
        else:
            placeholder.write(response_text)

# Download chat as JSON

st.divider()



export_data = []

for m in get_history(SESSION_ID).messages:
    role = getattr(m, "type","")
    if role == "human":
        export_data.append({"role":"user","text": m.content})
    else:
        export_data.append({"role":"assistant","text":m.content})

history_messages = get_history(SESSION_ID).messages

if history_messages:
    st.subheader("⬇️ Download Chat History")
    st.download_button(
        label="⬇️ Download chat_history.json",
        data=json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
)

## Downlaod Chat as Txt

def convert_chat_to_text(messages):
    chat_text = ""
    for msg in messages:
        role = getattr(msg, "type", "")
        
        if role == "human":
            chat_text += f"User: {msg.content}\n\n"
        else:
            chat_text += f"Assistant: {msg.content}\n\n"
    
    return chat_text

history_messages = get_history(SESSION_ID).messages

if history_messages:
    chat_text = convert_chat_to_text(history_messages)

    st.download_button(
        label="⬇️ Export Chat as TXT",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"

    )











