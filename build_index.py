# build_index.py
from pathlib import Path
from dotenv import load_dotenv

# Loaders, splitters, embeddings, vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os





#loading the Pdfs

pdf_path = ['paper/Attention.pdf','paper/LLM.pdf']


INDEX_DIR = "indexes/attention_faiss"

def load_docs(paths):
    docs = []
    for p in paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    return docs

def main():
    docs = load_docs(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    chunk = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    vecor_db = FAISS.from_documents(chunk , embedding)

    Path(INDEX_DIR).mkdir(parents= True , exist_ok= True)
    vecor_db.save_local(INDEX_DIR)
    print(f'Saved the FAISS index to the {INDEX_DIR}')

if __name__ == "__main__":
    main()

