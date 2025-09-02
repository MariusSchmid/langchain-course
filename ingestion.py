import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    print("This is the ingestion module.")
    loader = TextLoader("./mediumblog1.txt")
    documents = loader.load()

    print("splitting the document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = PineconeEmbeddings(model="multilingual-e5-large")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    print("Ingestion complete.")
