import requests
from bs4 import BeautifulSoup
import os

filename = 'training_data.csv'

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

def save_training_data(data):
    if os.path.exists(filename):
        mode = 'a'
    else:
        mode = 'w'
    
    with open(filename, mode) as file:
        file.write(f"{data['title']},{data['author']},{data['content']}\n")

def main():
    while True:
        query = input("Enter a topic to learn (or type 'exit' to stop): ")
        if query.lower() == 'exit':
            break

        book_links = search_library_genesis(query)
        if not book_links:
            print("No books found for the given query.")
            continue

        for link in book_links:
            book_data = fetch_book_details(link)
            save_training_data(book_data)
            print(f"Learned from book: {book_data['title']}")

    print("Learning process completed.")

if __name__ == '__main__':
    main()
