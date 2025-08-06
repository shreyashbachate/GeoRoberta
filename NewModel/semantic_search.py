import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = 'addresses.csv' # <-- CHANGED to use your file
INDEX_FILE = 'address_index.faiss'
ID_MAP_FILE = 'address_ids.csv'

def index_data():
    """Reads addresses from your CSV, creates embeddings, and builds a FAISS index."""
    if os.path.exists(INDEX_FILE):
        print("Index already exists. Skipping indexing.")
        return

    print("--- Phase 1: Indexing Your Dataset ---")
    
    # 1. Load a pre-trained model
    print(f"Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Read addresses from your CSV
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please make sure it's in the same directory.")
        return
        
    df = pd.read_csv(DATA_FILE)
    # Handle potential missing values in the address column
    df.dropna(subset=['address'], inplace=True)
    addresses = df['address'].astype(str).tolist()
    print(f"Found {len(addresses)} addresses to index from {DATA_FILE}.")

    # 3. Generate embeddings for all addresses
    print("Generating embeddings... (This may take a while for large datasets)")
    start_time = time.time()
    embeddings = model.encode(addresses, show_progress_bar=True, convert_to_tensor=True)
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

    # 4. Build and save the FAISS index
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    # FAISS requires numpy arrays on the CPU
    index.add(embeddings.cpu().numpy())
    
    print(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)

    # 5. Save the original IDs and addresses for mapping results back
    # <-- CHANGED to use your 'ticket_id' column
    df[['ticket_id', 'address']].to_csv(ID_MAP_FILE, index=False)
    
    print("--- Indexing Complete ---")


class SemanticSearcher:
    """A class to handle loading the index and performing searches."""
    def __init__(self):
        print("--- Phase 2: Initializing Searcher ---")
        # 1. Load the model
        print(f"Loading model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        
        # 2. Load the FAISS index
        print(f"Loading FAISS index from {INDEX_FILE}")
        self.index = faiss.read_index(INDEX_FILE)

        # 3. Load the ID and address mapping
        self.id_map_df = pd.read_csv(ID_MAP_FILE)

    def search(self, query: str, top_k: int = 5):
        """
        Performs a semantic search for a given query.
        """
        print(f"\nSearching for '{query}'...")
        start_time = time.time()
        
        # 1. Encode the query into a vector
        query_embedding = self.model.encode([query])

        # 2. Search the FAISS index
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        
        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")
        
        # 3. Format and print results
        print("--- Top Results ---")
        for i, idx in enumerate(indices[0]):
            # <-- CHANGED to use your 'ticket_id' column
            original_id = self.id_map_df.iloc[idx]['ticket_id']
            address = self.id_map_df.iloc[idx]['address']
            distance = distances[0][i]
            
            print(f"Score (L2 Distance): {distance:.4f} - Address: {address} (Ticket ID: {original_id})")


# --- Main Execution ---
if __name__ == "__main__":
    index_data()
    
    # Check if index was created before starting search
    if os.path.exists(INDEX_FILE):
        searcher = SemanticSearcher()
        
        # Example searches using data from your file
        searcher.search("12829 harper, Detroit")
        searcher.search("9100 van dyke avenue")
        searcher.search("russell street, 48207")
    else:
        print("\nCould not find index file. Please check for errors during the indexing phase.")