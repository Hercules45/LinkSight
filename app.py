import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, render_template, jsonify
from bs4 import BeautifulSoup
import requests
import nltk
import numpy as np
import re
import time
import traceback

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

app = Flask(__name__)

model = None

def initialize_model():
    """Initialize the sentence transformer model with fallback options"""
    global model
    if model is not None:
        return model
    
    model_options = [
        'all-MiniLM-L6-v2',
        'paraphrase-MiniLM-L3-v2',
        'distilbert-base-nli-mean-tokens'
    ]
    
    last_exception = None
    for model_name in model_options:
        try:
            print(f"Trying to load model: {model_name}")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            print(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            last_exception = e
            print(f"Failed to load model {model_name}: {str(e)}")
            time.sleep(1)
    
    if model is None:
        try:
            print("Using scikit-learn TF-IDF as fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            class SimpleTfIdfModel:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer()
                    self.fitted = False
                def encode(self, texts, convert_to_tensor=False):
                    if isinstance(texts, str):
                        texts = [texts]
                    if not self.fitted:
                        self.vectorizer.fit(texts)
                        self.fitted = True
                    return self.vectorizer.transform(texts).toarray()
            model = SimpleTfIdfModel()
            return model
        except Exception as e:
            print(f"Failed to initialize fallback model: {str(e)}")
            if last_exception:
                raise last_exception
            raise e
    return model

def similarity_scores(query_embedding, sentence_embeddings):
    """Calculate similarity scores between query and sentences"""
    if not hasattr(model, 'util'):
        from sklearn.metrics.pairwise import cosine_similarity
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return cosine_similarity(query_embedding, sentence_embeddings)[0]
    else:
        from sentence_transformers import util
        return util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0].cpu().numpy()

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', '.', text)
    return text

def scrape_and_extract_text(url):
    """Fetches and extracts text from a URL, targeting main content across websites."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else "Untitled Page"
        
        content_areas = [
            soup.find('div', id='mw-content-text'),  # Wikipedia
            soup.find('article'),  # Blogs, news
            soup.find('main'),  # HTML5 standard
        ]
        content_div = next((area for area in content_areas if area), soup)
        
        for element in content_div.find_all(['script', 'style', 'header', 'footer', 'nav', 'aside',
                                            'div[class*="ad"]', 'div[class*="advertisement"]',
                                            'div[class*="nav"]', 'ul[class*="menu"]',
                                            'div[class*="footer"]', 'ol[class="references"]',
                                            'div[class="reflist"]', 'sup', 'span[class="citation"]',
                                            'span[class="reference"]', 'a[class="external text"]',
                                            'a[class="new"]', 'a[class="image"]', 'li[class="reference"]']):
            element.decompose()
        
        paragraphs = []
        for paragraph in content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=True):
            if paragraph.find_parent(class_=['reference', 'reflist', 'references']):
                continue
            text = paragraph.get_text().strip()
            if text and len(text) > 50:
                paragraphs.append(text)
        
        if not paragraphs:
            for paragraph in soup.find_all('p'):
                text = paragraph.get_text().strip()
                if text and len(text) > 50:
                    paragraphs.append(text)
        
        if not paragraphs:
            text = soup.get_text(separator=' ', strip=True)
            text = clean_text(text)
            print(f"Scraped content from {url} (first 1000 chars, full fallback): {text[:1000]}")
            return text, title
        
        text = "\n\n".join(paragraphs)
        text = clean_text(text)
        print(f"Scraped content from {url} (first 1000 chars): {text[:1000]}")
        return text, title
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

scraped_data = {}
url_metadata = {}

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape_url():
    """Handles URL ingestion."""
    try:
        data = request.get_json()
        url = data.get('url', '')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        if url in scraped_data:
            return jsonify({
                'success': True,
                'title': url_metadata.get(url, 'Untitled Page'),
                'message': 'URL already processed'
            }), 200
        
        text, title = scrape_and_extract_text(url)
        if text:
            scraped_data[url] = text
            url_metadata[url] = title
            return jsonify({
                'success': True,
                'title': title,
                'message': f'Successfully scraped {url}'
            }), 200
        else:
            return jsonify({'error': 'Could not extract content from the URL'}), 400
    except Exception as e:
        print(f"Error in scrape_url: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/answer', methods=['POST'])
def answer_question():
    """Handles question answering based on scraped content."""
    try:
        initialize_model()
        data = request.get_json()
        question = data.get('question', '')
        urls = data.get('urls', [])
        if not question or not urls or not any(url in scraped_data for url in urls):
            return jsonify({'error': 'Question and valid URLs required'}), 400
        
        relevant_texts = [scraped_data[url] for url in urls if url in scraped_data]
        combined_text = "\n\n".join(relevant_texts)
        
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(combined_text)
        except LookupError:
            print("NLTK punkt not found, downloading now...")
            nltk.download('punkt')
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(combined_text)
        
        filtered_sentences = []
        for i, s in enumerate(sentences):
            cleaned = re.sub(r'\[\d+\](?:\[\w+\])?\s*', '', s).strip()
            if (len(cleaned) > 50 or i < 5) and len(cleaned) > 20:
                filtered_sentences.append(cleaned)
        
        if not filtered_sentences:
            return jsonify({'answer': 'No meaningful content found after filtering.'}), 200
        
        question_embedding = model.encode(question, convert_to_tensor=True)
        sentence_embeddings = model.encode(filtered_sentences, convert_to_tensor=True)
        scores = similarity_scores(question_embedding, sentence_embeddings)
        
        # Correctly apply boost to early sentences
        boosted_scores = np.array([score + 0.3 if i < 5 else score for i, score in enumerate(scores)])
        
        all_sentences_with_scores = [(s, float(boosted_scores[i])) for i, s in enumerate(filtered_sentences)]
        print(f"All sentences with boosted scores for '{question}': {all_sentences_with_scores[:20]}")
        
        top_n = min(5, len(filtered_sentences))
        top_indices = np.argsort(-boosted_scores)[:top_n]
        top_sentences = [filtered_sentences[idx] for idx in top_indices]
        top_scores = [float(boosted_scores[idx]) for idx in top_indices]
        
        print(f"Top sentences for '{question}': {list(zip(top_sentences, top_scores))}")
        
        # Ensure lead sentence for summary queries
        if "tell me about" in question.lower():
            answer = filtered_sentences[0]  # Force lead sentence for "tell me about" queries
        elif max(top_scores) > 0.3:
            answer = top_sentences[0]
        else:
            answer = filtered_sentences[0]
        
        return jsonify({'answer': answer}), 200
    except Exception as e:
        traceback.print_exc()
        print(f"Error in answer_question: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        initialize_model()
    except Exception as e:
        print(f"Warning: Could not initialize model at startup: {e}")
        print("The model will be initialized when needed.")
    app.run(debug=False)