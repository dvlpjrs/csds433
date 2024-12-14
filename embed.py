from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

# Load environment variables
load_dotenv()

# Step 1: Use FireCrawlLoader to scrape and load documents
api_key = os.getenv("FIRECRAWL_API_KEY")
start_url = "https://sites.google.com/a/case.edu/hpcc"

# Initialize the FireCrawlLoader
loader = FireCrawlLoader(
    api_key=api_key,
    url=start_url,
    params={
        "limit": 500,  # total document in HPC 367
        "scrapeOptions": {
            "formats": ["markdown"],
        },
    },
)

# Load documents directly using FireCrawlLoader
documents = loader.load()

print(f"Loaded {len(documents)} documents from FireCrawlLoader.")

# Step 2: Split large documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Set chunk size (tokens or characters)
    chunk_overlap=100,  # Overlap between chunks for context preservation
)

# Split each document into smaller chunks
chunked_docs = []
for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    chunked_docs.extend(chunks)

print(f"Split documents into {len(chunked_docs)} chunks.")

# Step 3: Embed the chunks using OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create a FAISS vector store from the chunks
vector_store = FAISS.from_texts(chunked_docs, embeddings)

# Step 4: Save the FAISS vector store for later use
vector_store.save_local("faiss_index")

print("FAISS vector store created and saved successfully!")
