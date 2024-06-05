import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# Load the model globally to avoid reloading it for each request
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
model.eval()  # Set model to evaluation mode

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=2000)  # Adjust the number of workers as needed

class AnswerModel(BaseModel):
    model_answer: str
    user_answer: str

async def async_sentence_similarity(model_answer, user_answer):
    loop = asyncio.get_event_loop()
    similarity = await loop.run_in_executor(executor, sentence_similarity, model_answer, user_answer)
    return similarity

def sentence_similarity(model_answer, user_answer):
    sentence1 = torch.tensor([model.encode([model_answer])[0]]).to(model.device)
    sentence2 = torch.tensor([model.encode([user_answer])[0]]).to(model.device)
    cos_sim1 = util.pytorch_cos_sim(sentence1, sentence2)
    return cos_sim1.item()

@app.get("/answer/")
async def get_similarity(answer: AnswerModel):
    similarity = await async_sentence_similarity(answer.model_answer, answer.user_answer)
    return {"similarity": str(similarity)}
