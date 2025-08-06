import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = 'addresses.csv'
INDEX_FILE = 'address_index.faiss'
ID_MAP_FILE = 'address_ids.csv'

def index_data():
    """Reads addresses from your CSV, creates embeddings, and builds a FAISS index."""
    if os.path.exists(INDEX_FILE):
        print("Index already exists. Skipping indexing.")
        return

    print("--- Phase 1: Indexing Dataset ---")
    

    print(f"Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)


    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please make sure it's in the same directory.")
        return
        
    df = pd.read_csv(DATA_FILE)
    df.dropna(subset=['address'], inplace=True)
    addresses = df['address'].astype(str).tolist()
    print(f"Found {len(addresses)} addresses to index from {DATA_FILE}.")


    print("Generating embeddings... (This may take a while for large datasets)")
    start_time = time.time()
    embeddings = model.encode(addresses, show_progress_bar=True, convert_to_tensor=True)
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")


    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings.cpu().numpy())
    
    print(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)


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
        start_time = time.time()
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")
        
        print("--- Top Results ---")
        for i, idx in enumerate(indices[0]):
            original_id = self.id_map_df.iloc[idx]['ticket_id']
            address = self.id_map_df.iloc[idx]['address']
            distance = distances[0][i]
            
            print(f"Score (L2 Distance): {distance:.4f} - Address: {address} (Ticket ID: {original_id})")



if __name__ == "__main__":

    index_data()
    
    if os.path.exists(INDEX_FILE):
        searcher = SemanticSearcher()
        

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