"""
ui/app.py — Streamlit Frontend Dashboard
=========================================
Run with: streamlit run ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.search import HybridRetriever
from agents.workflow import run_agent

st.set_page_config(page_title="Codebase Assistant", page_icon="🤖", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark gradient background for the whole app */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Hide default streamlit header */
    header {visibility: hidden;}
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* Hero title styling with glowing text */
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(0, 198, 255, 0.2); }
        to { text-shadow: 0 0 20px rgba(0, 198, 255, 0.6), 0 0 30px rgba(0, 114, 255, 0.4); }
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Primary Button Styling */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 3.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        width: 100%;
    }
    
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
        color: white;
    }
    
    /* Secondary Button Styling */
    .stButton>button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 3.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    /* Text Input Styling */
    .stTextInput>div>div>input {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #00c6ff;
        box-shadow: 0 0 0 1px #00c6ff;
    }
    
    /* Custom divider */
    hr {
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "home"

# Caching the retriever
@st.cache_resource
def get_retriever():
    return HybridRetriever()

# ==========================================
# PAGE 1: HOME
# ==========================================
if st.session_state.page == "home":
    st.markdown('<div class="hero-title">Codebase AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Instantly search, understand, and chat with any GitHub repository using advanced semantic retrieval and LLM reasoning.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center; margin-bottom: 2rem;'>Welcome to the Portal</h3>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", type="primary"):
            st.session_state.page = "ingest"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: INGEST
# ==========================================
elif st.session_state.page == "ingest":
    st.markdown('<div class="hero-title">Setup Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Provide a GitHub URL to index the codebase into our secure Vector Engine.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    repo_url = st.text_input("🔗 GitHub Repository URL", placeholder="https://github.com/docker/cli")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅️ Back", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
    with col_btn2:
        if st.button("Build Index ⚡", type="primary"):
            if not repo_url:
                st.error("Please enter a valid GitHub URL.")
            else:
                from ingestion.repo import RepoIngester
                from chunking.parser import ChunkingPipeline
                from retrieval.vector_store import VectorStoreManager
                from config import settings
                
                with st.status("Initializing AI Pipeline...", expanded=True) as status:
                    try:
                        st.write("📥 Cloning repository to secure environment...")
                        ingester = RepoIngester(repo_url=repo_url, local_path=settings.repo_local_path, force_reclone=True)
                        files = ingester.ingest()
                        
                        st.write(f"🧩 Parsing and building AST for {len(files)} files...")
                        pipeline = ChunkingPipeline()
                        chunks = pipeline.chunk_repository(files)
                        
                        st.write("🧠 Generating embeddings and building Vector DB...")
                        manager = VectorStoreManager()
                        stored = manager.ingest_chunks(chunks, clear_existing=True)
                        
                        status.update(label=f"Index Built Successfully! ({stored} chunks)", state="complete", expanded=False)
                        
                        # Clear cache for retriever
                        st.cache_resource.clear()
                        
                        # Move to Chat
                        st.session_state.page = "chat"
                        st.rerun()
                        
                    except Exception as e:
                        status.update(label="Pipeline Failed", state="error", expanded=True)
                        st.error(f"Error details: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: CHAT & SEARCH
# ==========================================
elif st.session_state.page == "chat":
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<h2 style='margin:0;'>🤖 Assistant Workspace</h2>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Change Repo", type="secondary"):
            st.session_state.page = "ingest"
            st.rerun()
            
    st.markdown("<hr>", unsafe_allow_html=True)
    
    retriever = get_retriever()
    
    mode = st.radio("Intelligence Mode", ("Hybrid Search (Fast)", "LangGraph Agent (Reasoning)"), horizontal=True)
    query = st.chat_input("Ask a question about the codebase...")
    
    if query:
        st.chat_message("user").write(query)
        
        if mode == "Hybrid Search (Fast)":
            with st.spinner("Searching semantic index..."):
                results = retriever.search(query, top_k=5)
                
                with st.chat_message("assistant"):
                    if not results:
                        st.warning("No results found in the current index.")
                    else:
                        st.markdown("Here are the most relevant code snippets I found:")
                        for i, r in enumerate(results, 1):
                            with st.expander(f"{i}. {r.symbol} — {r.file_path}"):
                                st.caption(f"Score: {r.score:.4f} | Type: {r.chunk_type} | Lines: {r.start_line}-{r.end_line}")
                                st.code(r.content, language="go")
                            
        else:
            with st.spinner("Agent is reasoning across the codebase..."):
                try:
                    answer = run_agent(query)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"Agent failed: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)
