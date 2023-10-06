from langchain.chains import ConversationChain,LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

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

def bot(prompt):
    response = conversation({"query":prompt})
    answer = response['text']
    return answer