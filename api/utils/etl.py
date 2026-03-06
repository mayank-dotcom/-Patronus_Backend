import pdfplumber
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def process_pdf(pdf_path, mongodb_uri, db_name, collection_name, index_name):
    """
    Extracts text and tables from PDF, chunks them, and stores in MongoDB Atlas Vector Search via OpenAI.
    Uses memory-efficient processing for large PDFs.
    """
    print(f"DEBUG: Starting PDF processing for: {pdf_path}")
    print(f"DEBUG: Using DB: {db_name}, Collection: {collection_name}, Index: {index_name}")

    client = MongoClient(mongodb_uri)
    try:
        client.admin.command('ping')
        print("DEBUG: MongoDB Atlas connection successful!")
    except Exception as e:
        print(f"DEBUG: MongoDB connection failed: {str(e)}")
        raise e
        
    collection = client[db_name][collection_name]
    
    # Initialize OpenAI Embeddings
    print("DEBUG: Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", # Uses 1536 dimensions
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    documents = []
    
    print("DEBUG: Opening PDF with pdfplumber (memory-efficient mode)...")
    with pdfplumber.open(pdf_path) as pdf:
        print(f"DEBUG: PDF opened. Total pages: {len(pdf.pages)}")
        
        # Process pages in smaller batches to save memory
        batch_size = 5  # Process 5 pages at a time
        total_pages = len(pdf.pages)
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            print(f"DEBUG: Processing batch {batch_start//batch_size + 1}: pages {batch_start+1} to {batch_end}")
            
            for page_num in range(batch_start + 1, batch_end + 1):
                page = pdf.pages[page_num - 1]
                
                # 1. Extract Text (with memory optimization)
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"WARNING: Failed to extract text from page {page_num}: {str(e)}")
                    text = ""
                
                # 2. Extract Tables (only if they exist)
                table_markdown = ""
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table:
                                cleaned_table = [["" if v is None else str(v) for v in row] for row in table]
                                if cleaned_table:
                                    header = " | ".join(cleaned_table[0])
                                    separator = " | ".join(["---"] * len(cleaned_table[0]))
                                    rows = [" | ".join(row) for row in cleaned_table[1:]]
                                    table_markdown += f"\n\nTable on page {page_num}:\n{header}\n{separator}\n" + "\n".join(rows) + "\n"
                except Exception as e:
                    print(f"WARNING: Failed to extract tables from page {page_num}: {str(e)}")

                full_content = f"Page {page_num}\n\n{text}\n\n{table_markdown}"
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "page_number": page_num,
                        "source": os.path.basename(pdf_path)
                    }
                )
                documents.append(doc)
                print(f"DEBUG: Processed page {page_num}, extracted {len(full_content)} chars.")
            
            # Force garbage collection after each batch
            import gc
            gc.collect()

    print(f"DEBUG: Total documents prepared: {len(documents)}")
    
    if not documents:
        raise Exception("No content extracted from PDF. Document list is empty.")

    # Store in MongoDB - insert in batches to save memory
    print(f"DEBUG: Starting MongoDB insertion into {collection_name} using OpenAI...")
    try:
        # Insert documents one by one to avoid memory spikes
        for i, doc in enumerate(documents, 1):
            vector_search = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name=index_name
            )
            vector_search.add_documents([doc])
            if i % 10 == 0:
                print(f"DEBUG: Inserted {i}/{len(documents)} documents")
            
        print("DEBUG: Successfully inserted all documents into MongoDB.")
    except Exception as e:
        print(f"DEBUG: Error during MongoDB insertion: {str(e)}")
        raise e
    
    return f"Processed {len(documents)} pages and inserted into {collection_name}."
