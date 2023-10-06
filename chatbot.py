# import databutton as db
import streamlit as st
import os
# from embedchain import _stream_query_response
# from string import Template
from langchain.chains import ConversationChain,LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# with openai
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(temperature=0.2)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Serve as a virtual lawyer aiding under-trial prisoners in India. Offer precise legal information, guide through processes, and suggest resources based on user input. Curate relevant details through interactive questions. Avoid disclaimers like 'I am not a lawyer' in responses. Keep responses concise."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{query}")
    ]
)

memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
) 

st.subheader("LawGPT 🤖 by `Aapka Adhikar`")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    import time
    import openai

    @st.cache_resource
    def botadd(prompt):
        response = conversation({"query":prompt})
        answer = response['text']
        return answer

    if "btn_state" not in st.session_state:
        st.session_state.btn_state = False

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = botadd(prompt=prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response:
            full_response += chunk + ""
            time.sleep(0.03)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )