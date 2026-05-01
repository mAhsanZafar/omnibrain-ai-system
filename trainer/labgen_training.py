import argparse
import logging
import os

import joblib
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def search_library_genesis(query):
    url = f'http://libgen.rs/search.php?req={query}&lg_topic=libgen&open=0&view=simple&res=25&phrase=1&column=def'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    book_links = []

    for row in soup.find_all('tr', {'valign': 'top'}):
        link = row.find_all('a', href=True)[0]['href']
        if link.startswith('book/index.php?md5='):
            book_links.append(link)

    return book_links

def fetch_book_details(book_link):
    url = f'http://libgen.rs/{book_link}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Safely extract title
    title_tag = soup.find('h1')
    title = title_tag.text.strip() if title_tag else 'Unknown Title'

    # Safely extract author
    author_tag = soup.find('h2')
    author = author_tag.text.strip() if author_tag else 'Unknown Author'

    # Safely extract content
    content_tag = soup.find('pre')
    content = content_tag.text.strip() if content_tag else 'No Content'

    return {'title': title, 'author': author, 'content': content}

def build_dataset(queries):
    documents = []
    labels = []
    for query in queries:
        logging.info("Searching LibGen for: %s", query)
        book_links = search_library_genesis(query)
        if not book_links:
            logging.warning("No books found for query: %s", query)
            continue
        for link in book_links:
            book_data = fetch_book_details(link)
            text = f"{book_data['title']} {book_data['author']} {book_data['content']}"
            documents.append(text)
            labels.append(query)
            logging.info("Learned from book: %s", book_data['title'])
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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_output_model = os.path.join(repo_root, "pre-trained model", "libgen_model.pkl")

    parser = argparse.ArgumentParser(description="Train a LibGen model and save libgen_model.pkl")
    parser.add_argument("--output-model", default=default_output_model, help="Path to save the trained model")
    parser.add_argument("--queries", nargs="*", help="Queries to search (omit to use interactive mode)")
    args = parser.parse_args()

    if args.queries:
        queries = args.queries
    else:
        queries = []
        while True:
            query = input("Enter a topic to learn (or type 'exit' to stop): ").strip()
            if query.lower() == 'exit':
                break
            if query:
                queries.append(query)

    documents, labels = build_dataset(queries)
    train_and_save_model(documents, labels, args.output_model)
    logging.info("Learning process completed.")


if __name__ == '__main__':
    main()
