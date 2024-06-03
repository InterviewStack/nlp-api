from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class answer_model(BaseModel):
    model_answer: str
    user_answer: str

app = FastAPI()

@app.get("/answer/")
async def get_similarity(answer: answer_model):
    similarity = sentence_similarity(answer.model_answer, answer.user_answer)
    return {"similarity": str(similarity)}

def sentence_similarity(model_answer, user_answer):
    sentence1 = [model_answer]
    sentence2 = [user_answer]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    cos_sim1 = cosine_similarity(embedding1, embedding2)
    return cos_sim1[0][0]