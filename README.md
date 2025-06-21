# 🧠 ML Notes Assistant

A Streamlit application that helps you write and save notes, and even suggests the next few words using an Bidirectional LSTM language model!

## 🚀 Features

- **Create Notes** with a custom title and body.
- **Intelligent Word Prediction**: Suggest the next 1–20 words based on your current note, leveraging a trained LSTM model.
- **Save & Load**: Persist notes locally using `notes.json`.
- **Edit Capabilities**:
  - Update existing notes
  - Rename notes
  - Delete notes

---

## 🧩 Usage

### Create a Note
1. Select **Create Note** from the sidebar.
2. Enter a **Title** and start typing in the **Note Body**.
3. Pick how many words you want the model to predict (1–20).
4. Click **Suggest Next Words** to view the AI-generated continuation.
5. Save your note using the **Save Note** button.

### View / Edit Notes
1. Go to **View/Edit Notes** in the sidebar.
2. Select a note to:
   - **Update** its content
   - **Rename** via a new title
   - **Delete** permanently
   - **Export** note to '.pdf' or '.txt'
---

## 🧠 How It Works

- **Text Processing**: Uses Keras’s `Tokenizer` and `pad_sequences` to prepare input for the LSTM.
- **Inference**: Predicts the next word index by calling `model.predict()` and mapping it back to text.
- **Storage**: Notes are stored locally in `notes.json`.

---
