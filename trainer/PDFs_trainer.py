import os
import glob
import pickle
import logging
import time
import fitz  
import pytesseract
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import nltk
import spacy
from nltk.corpus import stopwords

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the necessary NLTK data is downloaded
nltk.download('stopwords')

# Load spaCy models
spacy_models = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm'),
    'fr': spacy.load('fr_core_news_sm'),
    'zh': spacy.load('zh_core_web_sm')
}

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path} with PyMuPDF: {e}. Using OCR.")
        try:
            text = pytesseract.image_to_string(pdf_path)
        except Exception as ocr_error:
            logging.error(f"Error extracting text from {pdf_path} with OCR: {ocr_error}")
            text = ""
    return text

def preprocess_text(text):
    # Detect language
    try:
        lang = detect(text)
    except:
        lang = 'en'  # Default to English if detection fails

    if lang not in spacy_models:
        lang = 'en'  # Default to English if the language is not supported

    nlp = spacy_models[lang]
    doc = nlp(text.lower())

    # Remove stop words and non-alphanumeric tokens
    stop_words = set(stopwords.words(lang if lang in stopwords.fileids() else 'english'))
    words = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

    preprocessed_text = ' '.join(words)
    return preprocessed_text

def load_data_from_folder(folder_path):
    documents = []
    labels = []
    for file_path in glob.glob(os.path.join(folder_path, '*.pdf')):
        text = extract_text_from_pdf(file_path)
        preprocessed_text = preprocess_text(text)
        documents.append(preprocessed_text)
        labels.append(os.path.basename(file_path).split('.')[0])  # Assuming filenames are labels
    return documents, labels

def train_and_save_model(X, y, model_name='model.csv'):
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(),
        'mlp': MLPClassifier(max_iter=1000),
        'decision_tree': DecisionTreeClassifier()
    }

    best_classifier = None
    best_score = -1

    min_class_count = min([y.count(label) for label in set(y)])
    n_splits = min(5, min_class_count)

    for classifier_name, classifier in classifiers.items():
        logging.info(f"Training {classifier_name}")
        model = make_pipeline(
            TfidfVectorizer(),
            classifier
        )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv)
        avg_score = scores.mean()

        logging.info(f"Average cross-validation score for {classifier_name}: {avg_score}")

        if avg_score > best_score:
            best_score = avg_score
            best_classifier = classifier

    logging.info(f"Best classifier: {best_classifier}")

    best_model = make_pipeline(
        TfidfVectorizer(),
        best_classifier
    )
    best_model.fit(X, y)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    versioned_model_name = f"{model_name.split('.')[0]}_{timestamp}.csv"
    with open(versioned_model_name, 'wb') as model_file:
        pickle.dump(best_model, model_file)

def load_model(model_name='model.csv'):
    with open(model_name, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

folder_path = r"E:\BAI\book"

X, y = load_data_from_folder(folder_path)

train_and_save_model(X, y)

loaded_model = load_model()

# Now you can use the loaded_model for predictions
