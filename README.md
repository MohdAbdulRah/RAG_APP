# RAG_APP — Web-Based Question Answering using LangChain, Groq & Streamlit

RAG_APP is an intelligent **Retrieval-Augmented Generation (RAG)** system with a **Streamlit UI**, allowing users to input **web URLs** and ask **natural language questions**. It scrapes, embeds, and retrieves relevant information from the web pages, generating accurate, context-aware answers with **source links** using **Groq LLMs**.

---
### Live Link :- https://ragapp-hqsvr5spy46a5gqdveq9x3.streamlit.app/
---

##  Features

-  Accepts multiple URLs as input directly from the web interface.
-  Cleans and processes webpage text using BeautifulSoup.
-  Embeds text using **HuggingFace Embeddings**.
-  Stores document vectors in **Chroma vector database**.
-  Uses **Groq (ChatGroq)** LLM for contextual responses.
-  Interactive **Streamlit** interface for real-time QA.
-  Displays **sources (URLs)** used for every answer.

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10+ |
| Frontend Framework | Streamlit |
| AI Framework | LangChain v1.0+ |
| Vector Database | Chroma |
| Embeddings | HuggingFace (`Alibaba-NLP/gte-base-en-v1.5`) |
| LLM | ChatGroq |
| Web Scraping | BeautifulSoup, Requests |
| Environment Variables | python-dotenv |

---

##  Project Structure

```
RAG_APP/
├── main.py                    # Streamlit frontend
├── rag.py                     # Core RAG pipeline logic
├── resources/
│   └── vectorstore/           # Persistent Chroma vector DB
├── requirements.txt           # Dependencies
├── .env                       # API key for Groq
└── README.md                  # Project documentation
```

---

## Installation

### 1️ Clone the repository
```bash
git clone https://github.com/MohdAbdulRah/RAG_APP.git
cd RAG_APP
```

### 2️ Create and activate a virtual environment
```bash
conda create -n langchain python=3.10 -y
conda activate langchain
```

### 3️ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️ Set up `.env` file
Create a file named `.env` in the project root and add your **Groq API key**:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶ Running the Streamlit App

To launch the web interface:
```bash
streamlit run main.py
```

The app will open in your browser (default: http://localhost:8501).

---

## How to Use

1️ **Enter URLs** — Paste one or multiple webpage URLs in the input field.

2️ **Process Data** — The app fetches, cleans, and embeds text into Chroma DB.

3️ **Ask a Question** — Type your query about the content of the provided URLs.

4️ **Get Answers** — The AI will retrieve relevant sections and respond, showing both **answers and source URLs**.

---

## Example Demo

**Input URLs:**
```
https://en.wikipedia.org/wiki/Apple_Inc.
```

**Question:**
> Who founded Apple?

**Output:**
```
Answer: Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
Sources: ['https://en.wikipedia.org/wiki/Apple_Inc.']
```

---

## Key Functions

### `preprocess_urls(urls)`
- Fetches, cleans, and splits content from web pages.
- Stores the embeddings persistently in Chroma.

### `generate_answer(query)`
- Retrieves the most relevant text from the vector database.
- Generates an answer using Groq LLM with references to source URLs.

---

## Concept Overview
**Retrieval-Augmented Generation (RAG)** combines **information retrieval** and **generative AI**. Instead of relying solely on pre-trained LLM knowledge, RAG retrieves up-to-date, factual information from external sources — in this case, webpages — and then uses an LLM to generate coherent, grounded answers.

---

## Streamlit UI Overview

| UI Section | Description |
|-------------|-------------|
| **URL Input Field** | Enter one or more URLs separated by commas |
| **Process Button** | Embeds and stores webpage content |
| **Question Box** | Input any natural language query |
| **Answer Display** | Shows AI-generated response and sources |

---

## Future Enhancements
- Support for PDF/document uploads
- Option to save and reload embeddings
- Add UI theme customization
- Option to switch between multiple LLMs (Groq, OpenAI, Gemini)

---

## Author
**Mohd Abdul Rahman**  
[GitHub](https://github.com/MohdAbdulRah)  
mohdabdulrahman510@gmail.com

---


