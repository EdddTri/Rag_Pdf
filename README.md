# RAG Chatbot with Groq, LangChain & FAISS

This project is a web-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about specific PDF documents. It's built with Python, Flask, LangChain, and powered by the lightning-fast Groq LPU™ Inference Engine for real-time responses.

The application allows you to chat with the content of the "Attention Is All You Need" and a general LLM paper, providing answers with source citations from the documents.

<img width="1213" height="731" alt="image" src="https://github.com/user-attachments/assets/45df4b5a-cd00-40c0-a699-632958c38dd0" />

---

## ✨ Features

-   **Fast & Responsive:** Uses the Groq API (Llama 3.1) for near-instant LLM responses.
-   **Citation-Friendly:** Cites sources directly from the provided PDF documents.
-   **Chat History:** Remembers the context of the current conversation.
-   **Local Vector Store:** Uses FAISS to store document embeddings locally for quick retrieval.
-   **Easy to Set Up:** Requires minimal setup with a simple Python and Flask stack.
-   **Modular Code:** Scripts are separated for building the index (`build_index.py`) and running the web application (`app.py`).

---

## 🚀 How It Works

The project follows a standard RAG pipeline:

1.  **Indexing (`build_index.py`):**
    * Loads PDF documents from the `/paper` directory.
    * Splits the documents into smaller, manageable chunks.
    * Uses Hugging Face's `all-MiniLM-L6-v2` model to create vector embeddings for each chunk.
    * Saves these embeddings into a local FAISS vector store located in the `/indexes` directory. This step only needs to be run once.

2.  **Inference (`app.py`):**
    * A user asks a question through the Flask web interface.
    * The app creates an embedding of the user's question.
    * FAISS performs a similarity search to retrieve the most relevant document chunks from the index.
    * The retrieved chunks (the context), the user's question, and the chat history are passed to a prompt template.
    * This complete prompt is sent to the Groq API, which generates a response based *only* on the provided context.
    * The final answer and its sources are displayed to the user.

---

## 🛠️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

You'll need a Groq API key to use the LLM.

1.  Create a file named `.env` in the root directory of the project.
2.  Sign up for a free account at [GroqCloud](https://console.groq.com/keys) to get your API key.
3.  Add your key to the `.env` file:

    ```env
    GROQ_API_KEY="your-groq-api-key-goes-here"
    ```

### 5. Build the Vector Index

Run the `build_index.py` script to process the PDFs and create the FAISS vector store.

```bash
python build_index.py
```

You should see a confirmation message: `Saved the FAISS index to the indexes/attention_faiss`.

### 6. Run the Application

Now you can start the Flask web server.

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000` in your web browser.

---

## 💬 Usage

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Type a question about the "Attention Is All You Need" paper or general LLM concepts into the input box.
3.  Press Enter or click "Ask".
4.  The model's answer will appear, along with the chat history and the sources it used to generate the response.

---

## 📂 Project Structure

```
.
├── 📄 app.py              # Main Flask application file
├── 📄 build_index.py        # Script to create the FAISS index
├── 📄 requirements.txt      # List of Python dependencies
├── 📄 .env                  # Environment variables (you create this)
├── 📁 paper/                # Contains the source PDF documents
│   ├── 📄 Attention.pdf
│   └── 📄 LLM.pdf
├── 📁 templates/            # HTML templates for the web UI
│   └── 📄 index.html
└── 📁 indexes/              # (Generated) Stores the FAISS vector index
    └── 📁 attention_faiss/
```

---

## 💻 Technologies Used

-   **Backend:** Python, Flask
-   **LLM Orchestration:** LangChain
-   **LLM Provider:** Groq (Llama-3.1-8B-Instant)
-   **Embeddings:** Hugging Face Sentence Transformers
-   **Vector Store:** FAISS (Facebook AI Similarity Search)
-   **Frontend:** HTML, CSS
````
