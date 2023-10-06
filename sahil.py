from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from typing_extensions import Concatenate

import os
load_dotenv()

from fastapi import FastAPI, UploadFile, File
import io

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/upload")
async def help(question : str, file: UploadFile = File(...)):
    pdf_content = await file.read()

    # Create a PDFReader object
    pdf_reader = PdfReader(io.BytesIO(pdf_content))

# read text from pdf
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    raw_text

# We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    len(texts)

# Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

    print(document_search)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    question = question
    docs = document_search.similarity_search(question)
    ans = chain.run(input_documents=docs, question=question)
    return {'Answer: ' : ans}