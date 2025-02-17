import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from collections import Counter

# Load dataset
data_path = 'Roman-Urdu-Poetry (1).csv'  # Dataset file path
df = pd.read_csv(data_path)
poetry_texts = df['Poetry'].dropna().tolist()

# Combine all poetry into one text
all_text = " ".join(poetry_texts)

# Tokenize at word level and count word frequencies
words = all_text.split()
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)

# Add padding index to vocab
vocab_size = len(vocab) + 1  # Adding 1 for padding index
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # Index starts from 1
idx2word = {idx: word for word, idx in word2idx.items()}

# Convert text to sequence of integers
def text_to_int(text):
    return [word2idx[word] for word in text.split() if word in word2idx]

# Convert sequence of integers back to text
def int_to_text(indices):
    return ' '.join([idx2word[idx] for idx in indices])

# Define the RNN model
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(WordRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = output.contiguous().view(-1, output.shape[2])
        logits = self.fc(output)
        return logits, hidden

# Load trained model
model_path = 'model.pth'
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

# Initialize model with parameters from checkpoint
vocab_size = checkpoint['vocab_size']
embed_size = checkpoint['embed_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']

# Create model instance and load state dict
model = WordRNN(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Function to generate poetry
def generate_poetry(seed_text, model, word2idx, idx2word, seq_length=10, num_words=20, temperature=0.8, words_per_line=5):
    words = seed_text.split()
    state_h, state_c = None, None
    poetry_lines = []
    current_line = [seed_text]

    for _ in range(num_words):
        input_seq = text_to_int(' '.join(words[-seq_length:]))
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits, (state_h, state_c) = model(input_tensor, (state_h, state_c) if state_h is not None else None)
            logits = logits[-1] / temperature
            probs = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
            next_word_idx = np.random.choice(len(probs), p=probs)

        next_word = idx2word.get(next_word_idx, "<UNK>")
        words.append(next_word)
        current_line.append(next_word)

        if len(current_line) >= words_per_line:
            poetry_lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        poetry_lines.append(" ".join(current_line))

    return "\n".join(poetry_lines)

# Streamlit UI for poetry generation
st.set_page_config(page_title="Roman Urdu Poetry Generator", page_icon="✍️", layout="centered")
st.write("📝 **Roman Urdu Poetry Generator**")
st.write("Enter a word in Roman Urdu, and let AI generate poetry for you!")

# Input for seed word
seed_text = st.text_input("Enter a word:", "ishq").strip()

# Generate poetry on button click
if st.button("Generate Poetry"):
    if seed_text:
        generated_poetry = generate_poetry(seed_text, model, word2idx, idx2word)
        st.text(generated_poetry)
    else:
        st.warning("Please enter a word.")
