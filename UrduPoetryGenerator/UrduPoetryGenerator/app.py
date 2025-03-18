import os
import torch
import torch.nn as nn
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # To embed HTML
from collections import Counter

# Load and Display Bootstrap Landing Page
def load_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

html_content = load_html("landingPage.html")

# Display the landing page before main app
st.set_page_config(layout="wide")  # Optional: Full-width layout
components.html(html_content, height=700, scrolling=True)

# Custom CSS for Styling
st.markdown(
    """
    <style>
        body {
            background-color: #5A3E36; /* Dark Brown */
            font-family: 'Arial', sans-serif;
            color: black;
        }
        .stApp {
            background-color: #5A3E36;
        }
        .title {
            text-align: center;
            color: white; /* Title in white for contrast */
            font-size: 36px;
            font-weight: bold;
        }
        .subheader, .text4 {
            color: black; /* Ensure both subheader and text4 are black */
            font-size: 24px;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            border: 2px solid #5A4B2E;
            background-color: #FAF0E6;
            color: black;
            font-size: 18px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #8B7765;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #5A4B2E;
        }
        .stSpinner {
            color: black !important;
            font-size: 20px;
            font-weight: bold;
        }
        .generated-poem  {
            font-size: 32px; /* Large Font */
            font-weight: bold;
            font-family: 'Georgia', serif;
            color: #FAEBD7; /* Light antique white */
            text-align: center;
            line-height: 1.8;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 20px;
            border-radius: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.markdown('<h1 class="title">✨ AI Urdu Poetry Generator ✨</h1>', unsafe_allow_html=True)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
@st.cache_data
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    if "poem" not in data.columns:
        raise ValueError("CSV file must contain a 'poem' column.")
    return data["poem"].tolist()

csv_path = "poems_dataset.csv"
poems = load_data(csv_path)

# Build Vocabulary
@st.cache_data
def build_vocab(poems):
    all_words = []
    for poem in poems:
        all_words.extend(poem.split())
    vocab = sorted(set(all_words))
    word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx) + 1  # +1 for padding
    return vocab_size, word_to_idx, idx_to_word

vocab_size, word_to_idx, idx_to_word = build_vocab(poems)

# Define Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits

# Load Model Efficiently
@st.cache_resource
def load_model():
    embed_dim = 100
    hidden_dim = 150
    model = LSTMModel(vocab_size, embed_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Generate Poem Function
def generate_poem(seed_text, next_words=50):
    model.eval()
    generated_text = seed_text

    for _ in range(next_words):
        token_list = [word_to_idx.get(word, 0) for word in generated_text.split()]
        token_list = token_list[-19:]
        pad_len = 19 - len(token_list)
        token_list = [0] * pad_len + token_list
        input_seq = torch.tensor(token_list, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_seq)
            predicted_idx = torch.argmax(output, dim=1).item()

        if predicted_idx == 0:
            break

        predicted_word = idx_to_word.get(predicted_idx, "")
        generated_text += " " + predicted_word
    return generated_text

# User Input
st.markdown('<h2 class="subheader">✍️ Enter a Starting Line:</h2>', unsafe_allow_html=True)
seed_text = st.text_input("", "dil ke virane mein" , label_visibility="hidden")

st.markdown('<h2 class="subheader">📏 Choose Number of Words:</h2>', unsafe_allow_html=True)
next_words = st.slider("", min_value=10, max_value=200, value=50)

if st.button("Generate Poem"):
    with st.spinner('<p class="text4">📝 Generating your beautiful Urdu poem...</p>'):
        result = generate_poem(seed_text, next_words)
    
    # Display the generated poem with large font and input line included
    st.markdown('<h2 class="subheader">📜 Generated Poem:</h2>', unsafe_allow_html=True)
    formatted_poem = f"<div class='generated-poem'>{result}</div>"
    st.markdown(formatted_poem, unsafe_allow_html=True)
