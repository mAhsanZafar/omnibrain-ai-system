import argparse
import logging
import os
import re

import joblib
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

class TrainingDataExtractor:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def scrape_chatgpt(self, query):
        url = f"https://chatgpt.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        response = self.session.get(url, headers=headers)
        logging.info("Fetching URL: %s", url)
        logging.info("Response Status Code: %s", response.status_code)

        results = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            search_results = soup.find_all("div", class_="search-result")
            logging.info("Number of search results found: %s", len(search_results))

            for result in search_results:
                title_element = result.find("h3", class_="title")
                title = title_element.text.strip() if title_element else "Title not found"
                logging.info("Extracted Title: %s", title)

                link_element = result.find("a", class_="result-link", href=True)
                link = link_element['href'] if link_element else "Link not found"
                logging.info("Extracted Link: %s", link)

                results.append({"title": title, "link": link})

        return results

def load_queries(queries_path):
    if not os.path.exists(queries_path):
        logging.warning("Queries file not found: %s", queries_path)
        return []
    with open(queries_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return re.findall(r'"([^"]+)"', content)


def build_dataset(queries, extractor):
    documents = []
    labels = []
    for query in queries:
        logging.info("Processing query: %s", query)
        results = extractor.scrape_chatgpt(query)
        label = query.replace('+', ' ')
        for result in results:
            documents.append(f"{result['title']} {result['link']}")
            labels.append(label)
    return documents, labels


def train_and_save_model(documents, labels, output_path):
    if not documents:
        logging.warning("No documents available to train.")
        return
    if len(set(labels)) < 2:
        logging.warning("Not enough classes to train a classifier.")
        return
    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)
    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info("Training completed with accuracy: %.4f", accuracy)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info("Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_queries_file = os.path.join(repo_root, "Data source", "GS.json")
    default_output_model = os.path.join(repo_root, "pre-trained model", "google_model.pkl")

    parser = argparse.ArgumentParser(description="Train a Google search model and save google_model.pkl")
    parser.add_argument("--queries-file", default=default_queries_file, help="Path to GS.json query list")
    parser.add_argument("--output-model", default=default_output_model, help="Path to save the trained model")
    args = parser.parse_args()

    extractor = TrainingDataExtractor()
    queries = load_queries(args.queries_file)
    documents, labels = build_dataset(queries, extractor)
    train_and_save_model(documents, labels, args.output_model)
