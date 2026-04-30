import os
import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
from ebooklib import epub
import ebooklib
import pydub
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract text from EPUB
def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_content().decode('utf-8')
    return text

# Function to extract text from MOBI and DJVU
def extract_text_from_mobi_djvu(path):
    return extract_text_from_pdf(path)

# Function to handle audio files
def process_audio(audio_path):
    audio = pydub.AudioSegment.from_file(audio_path)
    return np.array(audio.get_array_of_samples())

# Function to handle image files
def process_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to handle video files
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# Simple neural network model for demonstration
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to process a single file and extract features
def process_file(file_path):
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
        return text, 'text'
    elif file_path.endswith('.epub'):
        text = extract_text_from_epub(file_path)
        return text, 'text'
    elif file_path.endswith('.djvu') or file_path.endswith('.mobi'):
        text = extract_text_from_mobi_djvu(file_path)
        return text, 'text'
    elif file_path.endswith(('.wav', '.mp3')):
        audio_data = process_audio(file_path)
        return audio_data, 'audio'
    elif file_path.endswith(('.jpg', '.png')):
        image_data = process_image(file_path)
        return image_data, 'image'
    elif file_path.endswith('.mp4'):
        video_data = process_video(file_path)
        return video_data, 'video'
    return None, None

# Training function
def train_model(data, labels):
    input_shape = data.shape[1:] if data.ndim > 1 else (1,)
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels, epochs=5, batch_size=32)
    return model

# GUI for selecting training data folder
def select_training_data():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def main():
    # Paths for model and vectorizer
    model_path = "trained_model.keras"
    vectorizer_path = "vectorizer.pkl"
    context_path = "contexts.pkl"
    context_embeddings_path = "context_embeddings.pkl"

    # Load sentence transformer model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Check if trained model exists
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(context_path) and os.path.exists(context_embeddings_path):
        print("Loading existing model, vectorizer, contexts, and context embeddings...")
        model = keras.models.load_model(model_path)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        with open(context_path, 'rb') as file:
            contexts = pickle.load(file)
        with open(context_embeddings_path, 'rb') as file:
            context_embeddings = pickle.load(file)
    else:
        print("Training new model...")
        # Get training data from user
        training_data_path = select_training_data()

        if not training_data_path:
            print("No folder selected. Exiting.")
            return

        # Process each file in the training data folder
        data = []
        labels = []
        texts = []
        contexts = []

        # Use multiprocessing to process files in parallel
        files_to_process = []
        for root, _, files in os.walk(training_data_path):
            for file in files:
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)

        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(process_file, files_to_process))

        for processed_data, data_type in results:
            if data_type == 'text':
                texts.append(processed_data)
                contexts.append(processed_data[100000])  # Save the first 1000 characters as context
            elif processed_data is not None:
                data.append(processed_data)
                labels.append(0) 
        vectorizer = TfidfVectorizer(max_features=1000)
        if texts:
            text_features = vectorizer.fit_transform(texts).toarray()
            data.extend(text_features)
            labels.extend([0] * len(text_features))  # Placeholder labels, replace with actual labels

        # Convert lists to numpy arrays
        if len(data) > 0:
            data = np.array(data)
            labels = np.array(labels)

            # Train the model
            print("Starting model training...")
            model = train_model(data, labels)
            print("Model training complete.")

            # Compute context embeddings
            context_embeddings = sentence_model.encode(contexts)

            # Save the trained model, vectorizer, contexts, and context embeddings
            print("Saving model, vectorizer, contexts, and context embeddings...")
            model.save(model_path)
            with open(vectorizer_path, 'wb') as file:
                pickle.dump(vectorizer, file)
            with open(context_path, 'wb') as file:
                pickle.dump(contexts, file)
            with open(context_embeddings_path, 'wb') as file:
                pickle.dump(context_embeddings, file)
            print("Model, vectorizer, contexts, and context embeddings saved.")

    # Load the question-answering pipeline from Hugging Face
    qa_pipeline = pipeline("question-answering")

    # User interaction for new input
    while True:
        user_input = input("Enter a question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Find the most relevant context using semantic similarity
        question_embedding = sentence_model.encode(user_input)
        similarities = util.dot_score(question_embedding, context_embeddings).numpy()
        best_context_index = np.argmax(similarities)
        best_context = contexts[best_context_index]

        # Answer the user's question using the QA pipeline
        answer = qa_pipeline(question=user_input, context=best_context)
        print(f"Answer: {answer['answer']}")

if __name__ == "__main__":
    main()
