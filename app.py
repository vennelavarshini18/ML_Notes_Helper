import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json

# Load model and tokenizer
model = tf.keras.models.load_model("lstm_next_word.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

# Generate next words
def generate_text(seed_text, next_words=5):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted = np.argmax(predicted_probs)
        next_word = tokenizer.index_word.get(predicted, '')
        result += ' ' + next_word
    return result

# Load or initialize notes
def load_notes():
    if os.path.exists("notes.json"):
        with open("notes.json", "r") as f:
            return json.load(f)
    return {}

def save_notes(notes):
    with open("notes.json", "w") as f:
        json.dump(notes, f, indent=2)

notes = load_notes()

st.title("🧠 ML Notes Assistant")

menu = st.sidebar.selectbox("Menu", ["Create Note", "View/Edit Notes"])

if menu == "Create Note":
    topic = st.text_input("Enter Note Title")
    seed = st.text_area("Start typing your notes...", height=150)
    next_words = st.slider("How many words to predict?", 1, 20, 5)

    if st.button("Suggest Next Words"):
        completed = generate_text(seed, next_words)
        st.success("Suggested:")
        st.write(completed)

    if st.button("Save Note"):
        if topic and seed:
            notes[topic] = seed
            save_notes(notes)
            st.success("Note saved successfully!")
        else:
            st.warning("Title and content cannot be empty!")

elif menu == "View/Edit Notes":
    if notes:
        selected = st.selectbox("Select a note", list(notes.keys()))
        edited = st.text_area("Edit your note:", value=notes[selected], height=200)

        if st.button("Update Note"):
            notes[selected] = edited
            save_notes(notes)
            st.success("Note updated!")

        if st.button("Rename Note"):
            new_name = st.text_input("New title:")
            if new_name and new_name != selected:
                notes[new_name] = notes.pop(selected)
                save_notes(notes)
                st.success("Renamed successfully!")

        if st.button("Delete Note"):
            del notes[selected]
            save_notes(notes)
            st.warning("Note deleted.")
    else:
        st.info("No notes saved yet.")

