import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from duckduckgo_search import DDGS  # Import DuckDuckGo search
import logging  # Add logging
from urllib.parse import urlparse  # Import for URL validation
import requests  # Import requests for URL accessibility check

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


def search_duckduckgo(query):
    """Fetch web search results using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=1)]
            return "\n".join(results)
    except Exception as e:
        return None


def is_valid_url(url):
    """Validate the URL format."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def is_url_accessible(url):
    """Check if the URL is accessible."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"URL accessibility check failed for {url}: {e}")
        return False


def setup_rag_chain(url):
    """
    Loads data, creates embeddings, vector store, and the RAG chain for a given URL.
    Returns the retriever for later use.
    """
    try:
        if not is_valid_url(url):
            logging.error(f"Invalid URL provided: {url}")
            return None

        if not is_url_accessible(url):
            logging.error(f"URL is not accessible: {url}")
            return None

        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            logging.error(f"No documents were loaded from the URL: {url}")
            return None

        logging.info(f"Loaded {len(docs)} documents from the URL: {url}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_documents(split_docs, embeddings)

        return vector_store.as_retriever(search_kwargs={'k': 5})
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error for URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error in setup_rag_chain for URL {url}: {e}")
        return None