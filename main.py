from fastapi import FastAPI
import uvicorn
from chat_functions import bot

app = FastAPI()

@app.get('/')
async def root():
    return "Welcome to Aapka Aadhikar"

@app.post("/predict/")
async def predict(query : str = None):
    if(not query or query==None):
        return {"success":False,"error":"Provide a valid input"}
    
    bot_response = bot(query)
    return {"success":True,"text":query,"response":bot_response}

if __name__=="__main__":
    uvicorn.run('main:app',port=8080)