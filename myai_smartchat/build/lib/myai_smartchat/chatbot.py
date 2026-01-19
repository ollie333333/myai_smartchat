import numpy as np
import pickle
import re
import random
from duckduckgo_search import ddg_answers
import json

class WebSmartChatbot:
    """
    Neural-network-based chatbot that can search DuckDuckGo if it doesn't know the answer.
    """

    def __init__(self, lr=0.01, hidden_size=16):
        self.patterns = {}
        self.word_index = {}
        self.index_word = {}
        self.input_size = 0
        self.hidden_size = hidden_size
        self.output_size = 0
        self.lr = lr
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    # ---------------------------
    # Text processing
    # ---------------------------
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text

    def build_vocab(self, questions):
        words = set()
        for q in questions:
            words.update(self.clean_text(q).split())
        words = sorted(list(words))
        self.word_index = {w:i for i,w in enumerate(words)}
        self.index_word = {i:w for w,i in self.word_index.items()}
        self.input_size = len(words)
        return words

    def text_to_vector(self, text):
        vec = np.zeros((self.input_size,))
        for word in self.clean_text(text).split():
            if word in self.word_index:
                vec[self.word_index[word]] = 1
        return vec

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, data_file, epochs=1000):
        with open(data_file, "r") as f:
            data = json.load(f)

        self.patterns = data
        questions = list(data.keys())
        answers = list(data.values())
        self.output_size = len(answers)
        self.build_vocab(questions)

        # Encode outputs
        Y = np.zeros((len(questions), self.output_size))
        for i in range(len(questions)):
            Y[i][i] = 1

        # Shuffle to avoid bias
        combined = list(zip(questions, answers))
        random.shuffle(combined)
        questions, answers = zip(*combined)

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        # Training loop
        for epoch in range(epochs):
            X = np.array([self.text_to_vector(q) for q in questions])
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.softmax(Z2)

            m = len(questions)
            dZ2 = A2 - Y
            dW2 = np.dot(A1.T, dZ2)/m
            db2 = np.sum(dZ2, axis=0, keepdims=True)/m

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.sigmoid_deriv(A1)
            dW1 = np.dot(X.T, dZ1)/m
            db1 = np.sum(dZ1, axis=0, keepdims=True)/m

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if (epoch+1) % 100 == 0:
                loss = -np.sum(Y*np.log(A2 + 1e-8))/m
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

    # ---------------------------
    # Prediction
    # ---------------------------
    def predict(self, user_input):
        # Step 1: Keyword match
        user_input_clean = self.clean_text(user_input)
        for key in self.patterns:
            if key in user_input_clean:
                return random.choice(self.patterns[key])

        # Step 2: Neural network prediction
        if self.W1 is not None:
            x = self.text_to_vector(user_input).reshape(1, -1)
            A1 = self.sigmoid(np.dot(x, self.W1) + self.b1)
            A2 = self.softmax(np.dot(A1, self.W2) + self.b2)
            idx = np.argmax(A2)
            key = list(self.patterns.keys())[idx]
            return random.choice(self.patterns[key])

        # Step 3: DuckDuckGo fallback
        try:
            results = ddg_answers(user_input, related=True)
            if results:
                return results[0]
        except Exception as e:
            return f"Error searching DuckDuckGo: {e}"

        return "Sorry, I couldn't find an answer."

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, path="websmart.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="websmart.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ---------------------------
    # Helpers
    # ---------------------------
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
        return x*(1-x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
