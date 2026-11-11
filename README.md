
# KnowFlow AI – Enterprise Knowledge Assistant

KnowFlow AI is an AI-powered enterprise knowledge assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from business documents.

Users can upload PDF documents, ask natural-language questions, and receive relevant answers with source references.

## Problem Statement

Businesses often store important information in lengthy documents, making it difficult for employees to find specific answers quickly.

KnowFlow AI simplifies this process by letting users search and ask questions about their documents through an AI-powered interface.

## Key Features

- Upload business PDF documents.
- Ask questions in natural language.
- Retrieve relevant information from uploaded documents.
- Generate answers using a local language model.
- Display source document references and page numbers.

## Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2 (1B)
- PyPDF

## How It Works

1. Upload a business PDF.
2. The application processes the document.
3. Text is converted into embeddings.
4. Relevant information is retrieved using ChromaDB.
5. The language model generates an answer based on the retrieved content.
6. The application displays the answer with source references.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/sowmyapilla1997-collab/knowflow-ai-enterprise-knowledge-assistant.git
```

Navigate to the project folder:

```bash
cd knowflow-ai-enterprise-knowledge-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the Llama 3.2 model is available:

```bash
ollama pull llama3.2:1b
```

Run the application:

```bash
streamlit run src/app.py
```

## Project Purpose

This project demonstrates the practical application of Python, document processing, vector databases, embeddings, and Retrieval-Augmented Generation to solve a business knowledge retrieval problem.
