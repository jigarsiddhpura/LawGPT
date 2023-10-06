# import databutton as db
import streamlit as st
import os
# from embedchain import _stream_query_response
from string import Template
from embedchain.config import CustomAppConfig, QueryConfig, AppConfig,ChatConfig
from dotenv import load_dotenv
load_dotenv()

bot_context = Template("""
    $context : "You are an AI system that assists under-trial prisoners with common legal queries and guides them through legal processes. You should provide accurate information on legal matters, offer insights into legal procedures, and recommend relevant resources based on the specific input provided by the users."
    $history : chat_history
    $query : query
""")

chat_config = ChatConfig(stream = True,temperature=0,number_documents=3,model='gpt-3.5-turbo',template=bot_context)

st.subheader("`LawGPT` 🤖")

# with openai
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def list_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

# Remove the IF block if using from secrets
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    import time
    import openai
    from embedchain import App

    @st.cache_resource
    def botadd(FOLDER_PATH='.\\data'):
        
        bot = App()
        if os.path.isdir(FOLDER_PATH):
            files = list_files(FOLDER_PATH)
        else:
            st.error("Invalid folder path. Please enter a valid path.")

        if files :
            # Embed Online Resources
            for file_path in files:
                bot.add("pdf_file", file_path)
            st.success(f"All {len(files)} files connected to the bot")

        return bot

    if "btn_state" not in st.session_state:
        st.session_state.btn_state = False

    prompt = st.text_input(
        "Enter path of file folder: ",
        placeholder=".\\data",
    )
    btn = st.button("Initialize Bot")

    if btn or st.session_state.btn_state:
        st.session_state.btn_state = True
        bot = botadd(prompt)
        st.success("Bot Ready ☑️! ")

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
                assistant_response = bot.chat(prompt,config=chat_config)

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response:
                full_response += chunk + ""
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

            if st.button("Reset the bot"):
                bot.reset()
    else:
        st.info("Initiate a bot first!")