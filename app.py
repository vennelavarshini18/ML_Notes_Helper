import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fpdf import FPDF
import os
import json

model = tf.keras.models.load_model("lstm_next_word.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

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

def load_notes():
    if os.path.exists("notes.json"):
        with open("notes.json", "r") as f:
            return json.load(f)
    return {}

def save_notes(notes):
    with open("notes.json", "w") as f:
        json.dump(notes, f, indent=2)

def export_to_txt(title, content):
    with open(f"{title}.txt", "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n{content}")

def export_to_pdf(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{title}\n\n{content}")
    pdf.output(f"{title}.pdf")

notes = load_notes()

st.markdown(
    "<h1 style='text-align: center; color: #FAD02E;'>🧠 ML Notes Assistant</h1>",
    unsafe_allow_html=True
)

menu = st.sidebar.selectbox("📁 Menu", ["Create Note", "View/Edit Notes"])

if menu == "Create Note":
    with st.container():
        st.subheader("✍️ Create a New Note")

        topic = st.text_input("📌 Note Title", placeholder="e.g., Introduction to Neural Networks")
        seed = st.text_area("📝 Start typing your notes...", height=180, placeholder="Begin your note here...")
        next_words = st.slider("🔮 How many words to predict?", 1, 20, 5)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✨ Suggest Next Words"):
                completed = generate_text(seed, next_words)
                st.success("Here’s a suggestion:")
                st.markdown(f"<div style='background-color:#252525;padding:10px;border-radius:10px;color:white'><b>{completed}</b></div>", unsafe_allow_html=True)

        with col2:
            if st.button("💾 Save Note"):
                if topic and seed:
                    notes[topic] = seed
                    save_notes(notes)
                    st.success("✅ Note saved successfully!")
                else:
                    st.warning("⚠️ Please enter both a title and content!")

elif menu == "View/Edit Notes":
    st.subheader("🗂️ View or Edit Your Notes")
    if notes:
        search = st.text_input("🔍 Search Notes").lower()
        filtered_titles = [k for k in notes if search in k.lower()]
        if filtered_titles:
            selected = st.selectbox("📑 Select a note", filtered_titles)
            edited = st.text_area("✏️ Edit Note:", value=notes[selected], height=200)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 Update Note"):
                    notes[selected] = edited
                    save_notes(notes)
                    st.success("✅ Note updated!")

            with col2:
                new_name = st.text_input("📝 Rename Note", placeholder="Enter new title...")
                if st.button("🔁 Rename"):
                    if new_name and new_name != selected:
                        notes[new_name] = notes.pop(selected)
                        save_notes(notes)
                        st.success("✅ Renamed successfully!")

            with col3:
                if st.button("🗑️ Delete Note"):
                    del notes[selected]
                    save_notes(notes)
                    st.warning("🗑️ Note deleted.")

            st.markdown("---")
            col4, col5 = st.columns(2)
            with col4:
                if st.button("📄 Export to .txt"):
                    export_to_txt(selected, notes[selected])
                    st.success(f"Exported {selected}.txt")

            with col5:
                if st.button("📄 Export to .pdf"):
                    export_to_pdf(selected, notes[selected])
                    st.success(f"Exported {selected}.pdf")
        else:
            st.info("🔎 No matching notes found.")
    else:
        st.info("📭 No notes available. Create one first!")
