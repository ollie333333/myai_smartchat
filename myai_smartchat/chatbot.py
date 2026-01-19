# ~/myai_package/myai_smartchat/myai_smartchat/chatbot.py

import json
import numpy as np
import pickle
from ddgs import DDGS

class WebSmartChatbot:
    def __init__(self, input_size=10, hidden_size=20, output_size=10, lr=0.01):
        # Neural network parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        # Training data
        self.questions = []
        self.answers = []

    # --- Activation functions ---
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # --- Convert text to vector ---
    def text_to_vector(self, text):
        vec = np.zeros((self.input_size,))
        for i, c in enumerate(text):
            vec[i % self.input_size] += ord(c)
        return vec

    # --- Load training data ---
    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.questions = [q['question'] for q in data]
        self.answers = [q['answer'] for q in data]

    # --- Train neural network ---
    def train(self, filename, epochs=500):
        self.load_data(filename)
        questions = self.questions
        answers = self.answers

        # Convert answers to simple numeric vectors
        Y = np.zeros((len(answers), self.output_size))
        for i, a in enumerate(answers):
            Y[i, 0] = hash(a) % 100 / 100

        for epoch in range(epochs):
            X = np.array([self.text_to_vector(q) for q in questions])
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)

            m = len(questions)
            dZ2 = A2 - Y
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.sigmoid_deriv(A1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if (epoch + 1) % 100 == 0:
                loss = np.mean((A2 - Y) ** 2)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

    # --- Predict ---
    def predict(self, question):
        # 1️⃣ Try math evaluation
        try:
            answer = str(eval(question))
            return answer
        except:
            pass

        # 2️⃣ Neural network fallback
        X = np.array([self.text_to_vector(question)])
        A1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        A2 = self.softmax(np.dot(A1, self.W2) + self.b2)
        idx = np.argmax(A2)
        return f"I'm not sure, maybe: {idx}"

    # --- DuckDuckGo search ---
    def duckduckgo_search(self, query, max_results=5):
        results = []
        with DDGS() as ddgs:
            gen = ddgs.text(
                query=query,            # ✅ updated for ddgs package
                region='us-en',
                safesearch='moderate',
                timelimit='y',
            )
            for i, r in enumerate(gen):
                results.append(f"{r['title']}: {r['href']}")
                if i + 1 >= max_results:
                    break
        if not results:
            results.append("No results found.")
        return results

    # --- Save / Load model ---
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
