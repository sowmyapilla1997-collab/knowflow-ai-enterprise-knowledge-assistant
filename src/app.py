
import streamlit as st
import chromadb
import requests
from pypdf import PdfReader
from io import BytesIO
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def split_into_chunks(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


st.set_page_config(
    page_title="KnowFlow AI",
    page_icon="📚",
    layout="wide"
)

st.title("KnowFlow AI")
st.subheader("Enterprise Knowledge Assistant")
st.write("Find answers from your business documents.")

model = load_model()

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="company_documents",
    metadata={"hnsw:space": "cosine"}
)

st.divider()
st.header("1. Upload Business Documents")

uploaded_files = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Documents"):

    if not uploaded_files:
        st.warning("Please upload a PDF first.")

    else:
        existing_ids = collection.get()["ids"]

        if existing_ids:
            collection.delete(ids=existing_ids)

        total_chunks = 0

        for file in uploaded_files:

            reader = PdfReader(BytesIO(file.getvalue()))

            for page_number, page in enumerate(reader.pages, start=1):

                page_text = page.extract_text() or ""

                if not page_text.strip():
                    continue

                chunks = split_into_chunks(page_text)
                embeddings = model.encode(chunks)

                collection.upsert(
                    ids=[
                        f"{file.name}_page{page_number}_chunk{i}"
                        for i in range(len(chunks))
                    ],
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "source": file.name,
                            "page": page_number
                        }
                        for _ in chunks
                    ]
                )

                total_chunks += len(chunks)

        st.success(
            f"Processing complete! Total chunks: {total_chunks}"
        )

st.divider()
st.header("2. Ask Your Documents")

question = st.text_input("Enter your question")

if st.button("Get AI Answer"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please process your documents first.")

    else:

        question_embedding = model.encode([question])

        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=min(3, collection.count())
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""

        for i, document in enumerate(documents):
            context += (
                f"\nSource: {metadatas[i]['source']}, "
                f"Page: {metadatas[i]['page']}\n"
                f"Content: {document}\n"
            )

        prompt = f"""
Answer using ONLY the information in the documents.
If the answer is unavailable, say so clearly.

Documents:
{context}

Question: {question}
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )

            response.raise_for_status()

            answer = response.json()["response"]

            st.subheader("AI Answer")
            st.write(answer)

            st.subheader("Sources")

            unique_sources = {
                (metadata["source"], metadata["page"])
                for metadata in metadatas
            }

            for source, page in sorted(unique_sources):
                st.write(f"📄 {source} — Page {page}")

        except requests.exceptions.RequestException as error:
            st.error(f"Could not connect to Ollama: {error}")
