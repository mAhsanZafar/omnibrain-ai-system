import requests
from bs4 import BeautifulSoup
import os

class TrainingDataExtractor:
    def __init__(self):
        self.training_data = []

    def scrape_chatgpt(self, query):
        url = f"https://chatgpt.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        print(f"Fetching URL: {url}")
        print(f"Response Status Code: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            search_results = soup.find_all("div", class_="search-result")
            print(f"Number of search results found: {len(search_results)}")

            for result in search_results:
                title_element = result.find("h3", class_="title")
                title = title_element.text.strip() if title_element else "Title not found"
                print(f"Extracted Title: {title}")

                link_element = result.find("a", class_="result-link", href=True)
                link = link_element['href'] if link_element else "Link not found"
                print(f"Extracted Link: {link}")

                self.training_data.append({"title": title, "link": link})


    def save_training_data(self, filename):
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode, encoding='utf-8') as file:
            for data in self.training_data:
                file.write(f"Title: {data['title']}\n")
                file.write(f"Link: {data['link']}\n\n")
        print(f"Training data saved to {filename}")
        self.training_data = []  # Clear the training data after saving


if __name__ == "__main__":
    extractor = TrainingDataExtractor()
    GS = [
        "What+is+the+weather+like+today?"
    ]
    
    for query in GS:
        print(f"Processing query: {query}")
        extractor.scrape_chatgpt(query)
        extractor.save_training_data('training_data.csv')
        print("Query processed:", query)
