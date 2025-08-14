import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle

st.set_page_config(
    page_title="Address Semantic Search",
    page_icon="🔎",
    layout="centered"
)


PICKLE_FILE = './PickleDeploy/searcher.pkl'


class SemanticSearcher:
    """A class to handle loading the index and performing searches."""
    def __init__(self):
        
        self.model = None
        self.index = None
        self.id_map_df = None

    def search(self, query: str, top_k: int = 5):
        """Performs a semantic search for a given query."""
        if not all([self.model, self.index, self.id_map_df is not None]):
            st.error("Searcher is not properly initialized.")
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            original_id = self.id_map_df.iloc[idx]['ticket_id']
            address = self.id_map_df.iloc[idx]['address']
            score_percent = similarities[0][i] * 100
            results.append({
                'score': f"{score_percent:.2f}%",
                'address': address,
                'ticket_id': original_id
            })
        return results


@st.cache_resource
def load_searcher_from_pickle():
    """Loads the entire SemanticSearcher object from the pickle file."""
    if not os.path.exists(PICKLE_FILE):
        st.error(f"FATAL ERROR: The required '{PICKLE_FILE}' was not found.")
        st.info("Please run the indexing script first to generate the pickle file.")
        st.stop() # Halts the app if the pickle file isn't there.
    
    with st.spinner(f"Loading search engine from '{PICKLE_FILE}'..."):
        with open(PICKLE_FILE, 'rb') as f:
            searcher = pickle.load(f)
    return searcher


st.title("🔎 Address Semantic Search")
st.write("This app uses a pre-loaded search engine to find the most semantically similar addresses from your dataset.")


searcher = load_searcher_from_pickle()


with st.form(key='search_form'):
    query = st.text_input("Enter an address to search", placeholder="e.g., 12829 harper, Detroit")
    submit_button = st.form_submit_button(label='Search')

if submit_button and query:
    st.write(f"### Results for: *'{query}'*")
    
    with st.spinner("Searching..."):
        results = searcher.search(query)

    if not results:
        st.warning("No results found.")
    else:
        for i, result in enumerate(results):
            st.metric(label=f"Match #{i+1}", value=result['score'])
            st.write(f"**Address:** {result['address']}")
            st.write(f"**Ticket ID:** {result['ticket_id']}")
            st.divider()

elif submit_button:
    st.warning("Please enter an address to search.")
