import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = 'addresses.csv'
INDEX_FILE = 'address_index_similarity.faiss' # Using a new index file
ID_MAP_FILE = 'address_ids.csv'

def index_data():
    """Reads addresses, creates normalized embeddings, and builds a FAISS index for Cosine Similarity."""
    if os.path.exists(INDEX_FILE):
        print("Similarity index already exists. Skipping indexing.")
        return

    print("--- Phase 1: Indexing Your Dataset for Similarity Search ---")
    
    # 1. Load a pre-trained model
    print(f"Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Read addresses from your CSV
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please make sure it's in the same directory.")
        return
        
    df = pd.read_csv(DATA_FILE)
    df.dropna(subset=['address'], inplace=True)
    addresses = df['address'].astype(str).tolist()
    print(f"Found {len(addresses)} addresses to index from {DATA_FILE}.")

    # 3. Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(addresses, show_progress_bar=True)
    
    # 4. Normalize embeddings for Cosine Similarity
    faiss.normalize_L2(embeddings)

    # 5. Build the FAISS index for Inner Product (Cosine Similarity)
    # <-- CHANGED: Using IndexFlatIP for cosine similarity
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)
    
    print(f"Saving FAISS similarity index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)

    # 6. Save the original IDs and addresses for mapping results back
    df[['ticket_id', 'address']].to_csv(ID_MAP_FILE, index=False)
    
    print("--- Indexing Complete ---")


class SemanticSearcher:
    """A class to handle loading the index and performing searches."""
    def __init__(self):
        print("--- Phase 2: Initializing Searcher ---")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_FILE)
        self.id_map_df = pd.read_csv(ID_MAP_FILE)

    def search(self, query: str, top_k: int = 5):
        """Performs a semantic search for a given query."""
        print(f"\nSearching for '{query}'...")
        
        # 1. Encode and normalize the query vector
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # 2. Search the FAISS index
        # For IndexFlatIP, the returned distances are actually cosine similarities
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # 3. Format and print results
        print("--- Top Results ---")
        for i, idx in enumerate(indices[0]):
            original_id = self.id_map_df.iloc[idx]['ticket_id']
            address = self.id_map_df.iloc[idx]['address']
            # <-- CHANGED: Convert similarity score to a 0-100 scale
            score_percent = similarities[0][i] * 100
            
            print(f"Score: {score_percent:.2f}% - Address: {address} (Ticket ID: {original_id})")


# --- Main Execution ---
if __name__ == "__main__":
    # The indexing step will run once if the index file doesn't exist.
    index_data()
    
    if os.path.exists(INDEX_FILE):
        searcher = SemanticSearcher()
        
        # --- Interactive Loop ---
        while True:
            query = input("\nEnter an address to search (or type 'exit' to quit): ")
            
            if query.lower() == 'exit':
                print("Exiting.")
                break
            
            if not query:
                continue
            
            searcher.search(query)
            
    else:
        print("\nCould not find index file. Please check for errors during the indexing phase.")