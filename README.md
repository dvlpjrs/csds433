# HPC Docs Chatbot

This repository demonstrates a Retrieval-Augmented Generation (RAG) pipeline for optimizing HPC documentation search using vector databases and GPT-based models. It includes data preprocessing, embedding generation, and indexing, allowing for efficient semantic retrieval and context-aware response generation.

## Prerequisites

- Python 3.10+
- OpenAI API key
- FireCrawl API key

## Setup

1. Install required packages

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:

```bash
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

3. Run the `embed.py` script to create the vector store:

```bash
python embed.py
```

4. Run the `serve.py` script to start the chatbot:

```bash
streamlit run serve.py
```
