# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from my_neural_network.model import Sequential
from my_neural_network.layers import Linear
from my_neural_network.activations import ReLU, Softmax
import joblib  # для загрузки сохранённых весов

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем предобученную модель (пример для классификации текста)
# В реальности нужно преобразовать текст в вектор и подать на вход.
model = Sequential([
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
    Softmax()
])
# Загружаем веса (например, из файла)
# model.load_weights('mnist_weights.pkl')

class Request(BaseModel):
    message: str
    history: list = []

class Response(BaseModel):
    reply: str

@app.post("/generate", response_model=Response)
def generate(req: Request):
    # Здесь нужно преобразовать текст в вектор (например, через bag-of-words)
    # Для примера просто возвращаем эхо
    reply = f"Вы сказали: {req.message}. (Ответ от самописной нейросети)"
    return Response(reply=reply)

@app.get("/health")
def health():
    return {"status": "ok"}
