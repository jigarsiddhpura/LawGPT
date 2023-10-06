from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig 
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr 
from accelerate import init_empty_weights, infer_auto_device_map

def chat(chat_history, user_input):

    template = f"""Act as a chatbot that solves legal queries and answer the below query of an undertrial prisoner in a calm manner.
    Query : {user_input} """

    bot_response = qa_chain({"query": template})
    bot_response = bot_response['result']
    response = ""
    for letter in ''.join(bot_response):
        response += letter + ""
        yield chat_history + [(user_input, response)]

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"   
tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir='./model_config')      # 12 sec 

base_model = AutoModelForSeq2SeqLM.from_pretrained(
checkpoint,
device_map="auto",
offload_folder="offload",
offload_state_dict = True,
torch_dtype = torch.float32)   # 30 sec

# device_map = infer_auto_device_map(base_model)

# config = AutoConfig.from_pretrained('./config.json')
# model_state_dict = torch.load('./pytorch_model.bin')
# base_model = AutoModelForSeq2SeqLM.from_config(config)
# base_model.load_state_dict(model_state_dict)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

db = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)

pipe = pipeline(
    'text2text-generation',
    tokenizer= tokenizer,
    model = base_model,
    max_length = 512,
    do_sample = True,
    temperature = 0.8,
    top_p= 0.95
)
local_llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":2}),
        return_source_documents=True,
        )


with gr.Blocks() as gradioUI:
    
    gr.Image('lawgptlogo.png')
    
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        input_query = gr.TextArea(label='Input',show_copy_button=True)

    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            clear_input_btn = gr.Button("Clear Input", variant='secondary')
        with gr.Column():
            clear_chat_btn = gr.Button("Clear Chat", variant='stop')

    submit_btn.click(chat, [chatbot, input_query], chatbot)
    submit_btn.click(lambda: gr.Textbox(value=""), None, input_query, queue=False)
    clear_input_btn.click(lambda: None, None, input_query, queue=False)
    clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

gradioUI.queue().launch(share=True)
