from pymongo import MongoClient
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document

load_dotenv()

def test_connection():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME")
    coll_name = os.getenv("COLLECTION_NAME")
    index_name = os.getenv("VECTOR_INDEX_NAME")
    
    print(f"Testing Connection to: {coll_name}")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[coll_name]
    
    # Check count
    count = collection.count_documents({})
    print(f"Current document count in {coll_name}: {count}")
    
    # Try inserting a tiny test doc
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    doc = Document(page_content="Test document for vector search", metadata={"source": "test"})
    
    print("Attempting to insert test document with embeddings...")
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[doc],
        embedding=embeddings,
        collection=collection,
        index_name=index_name
    )
    
    new_count = collection.count_documents({})
    print(f"New document count: {new_count}")
    
    if new_count > count:
        print("Success! Document inserted.")
        # Check if embedding field exists in the new doc
        sample = collection.find_one({"metadata.source": "test"})
        if sample and "embedding" in sample:
            print(f"Embedding field found! Length: {len(sample['embedding'])}")
        else:
            print("Embedding field MISSING or named differently!")
            print(f"Keys found: {sample.keys() if sample else 'None'}")
    else:
        print("Failure. Count did not increase.")

if __name__ == "__main__":
    test_connection()
