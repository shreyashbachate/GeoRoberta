import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Address Semantic Search",
    page_icon="🔎",
    layout="centered"
)

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE = 'addresses.csv'
INDEX_FILE = 'address_index_similarity.faiss'
ID_MAP_FILE = 'address_ids.csv'

# --- Caching Functions for Performance ---
# @st.cache_resource is used for non-serializable objects like models and indexes
@st.cache_resource
def load_model():
    """Loads the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index."""
    if not os.path.exists(INDEX_FILE):
        return None
    return faiss.read_index(INDEX_FILE)

# @st.cache_data is used for serializable data like dataframes
@st.cache_data
def load_id_map():
    """Loads the mapping from index to ticket_id and address."""
    if not os.path.exists(ID_MAP_FILE):
        return None
    return pd.read_csv(ID_MAP_FILE)

# --- Indexing Function (runs only if index file doesn't exist) ---
def index_data(model):
    """Reads addresses, creates embeddings, and builds the FAISS index."""
    with st.spinner('Building search index for the first time. This may take a few minutes...'):
        if not os.path.exists(DATA_FILE):
            st.error(f"FATAL ERROR: {DATA_FILE} not found. Please place it in the same directory.")
            st.stop()
        
        df = pd.read_csv(DATA_FILE)
        df.dropna(subset=['address'], inplace=True)
        addresses = df['address'].astype(str).tolist()
        
        st.info(f"Found {len(addresses)} addresses to index.")
        
        embeddings = model.encode(addresses, show_progress_bar=True)
        faiss.normalize_L2(embeddings)

        embedding_dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dimension)
        index.add(embeddings)
        
        faiss.write_index(index, INDEX_FILE)
        df[['ticket_id', 'address']].to_csv(ID_MAP_FILE, index=False)
    st.success("Search index built successfully!")
    st.balloons()

# --- Main App Logic ---
st.title("🔎 Address Semantic Search")
st.write("This app uses a deep learning model to find the most semantically similar addresses from your dataset, even if the wording isn't an exact match.")

# Load resources
model = load_model()
faiss_index = load_faiss_index()
id_map_df = load_id_map()

# Check if indexing is needed
if faiss_index is None or id_map_df is None:
    st.warning("Search index not found. Building it now.")
    index_data(model)
    # Reload resources after indexing
    faiss_index = load_faiss_index()
    id_map_df = load_id_map()

# --- Search Interface ---
with st.form(key='search_form'):
    query = st.text_input("Enter an address to search", placeholder="e.g., 9100 van dyke avenue")
    submit_button = st.form_submit_button(label='Search')

if submit_button and query:
    st.write(f"### Results for: *'{query}'*")
    
    # 1. Encode and normalize the query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # 2. Search the FAISS index
    with st.spinner("Searching..."):
        similarities, indices = faiss_index.search(query_embedding, 5) # Search for top 5

    # 3. Display results
    if len(indices[0]) == 0:
        st.warning("No results found.")
    else:
        for i, idx in enumerate(indices[0]):
            original_id = id_map_df.iloc[idx]['ticket_id']
            address = id_map_df.iloc[idx]['address']
            score_percent = similarities[0][i] * 100
            
            st.metric(label=f"Match #{i+1}", value=f"{score_percent:.2f}%")
            st.write(f"**Address:** {address}")
            st.write(f"**Ticket ID:** {original_id}")
            st.divider()

elif submit_button:
    st.warning("Please enter an address to search.")