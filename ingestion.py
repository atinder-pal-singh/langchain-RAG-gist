import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def main():
    print("ingestion...")
    file_path = "./Harry Potter and the Philosopher's Stone.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print("chunking...")
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_spliter.split_documents(docs)

    print("vectorization...")
    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_Name = os.environ["INDEX_NAME"])

    print("ingestion complete")

if __name__ == "__main__":
    main()