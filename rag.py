from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import requests
from bs4 import BeautifulSoup  # ✅ clean HTML

# ✅ LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Globals
llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9, max_tokens=500)
    
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def preprocess_urls(urls):
    print("Initialize_components")
    initialize_components()

    vector_store.reset_collection()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    print("Load the data")

    all_docs = []
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            # ✅ Clean HTML
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            doc = Document(page_content=text, metadata={"source": url})
            all_docs.append(doc)
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    print("Splitting the data")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' ', '.'],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(all_docs)

    print("Add docs to vector store")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    print("Done!")

# from langchain.chains.retrieval import RetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ✅ Define a modern ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "You are an expert assistant. Use the below context to answer accurately.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}"
    )

    # ✅ Build the retrieval chain manually (new 1.0 way)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    result = chain.invoke(query)

    # ✅ Extract URLs (sources)
    docs = retriever.invoke(query)
    sources = list({
        doc.metadata.get("source")
        for doc in docs if doc.metadata.get("source")
    })

    return result.content, sources


if __name__ == '__main__':
    urls = [
        'https://en.wikipedia.org/wiki/Apple_Inc.'
    ]
    preprocess_urls(urls)
    answer,sources =  generate_answer("Tell me about the founder of Apple")
    print(f"Answer : {answer}")
    print(f"Sources : {sources}")

    
