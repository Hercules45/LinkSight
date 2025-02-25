# LinkSight

**A lightweight web tool to scrape URLs and answer questions using only their content.** LinkSight lets users input URLs, ask questions, and get concise answers derived from scraped text—no external knowledge or APIs involved.

## Features

- **URL Scraping:** Extracts main content from web pages (e.g., Wikipedia, news sites) using BeautifulSoup.
- **Question Answering:** Responds to queries like "Tell me about this page" with the lead sentence, boosted for relevance.
- **Simple Interface:** Single-page Flask app with URL input, question field, and answer display.
- **Content Processing:** Uses `sentence-transformers` for similarity scoring, with a fallback to ensure summaries.
- **No External Dependencies:** Runs entirely on local Python packages—purely self-contained.

## Requirements

- **Python 3.12+** 

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Hercules45/LinkSight.git
   cd LinkSight
   cd LinkSight Project
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
   ```bash
   python app.py
   ```

## Usage
1. **Add URLs:** Enter a URL and click "Add URL."
2. **Ask a Question:** Type a question (e.g., "Tell me about this page") and click "Ask Question."
3. **View Answer:** Get a concise response based on the scraped content.

