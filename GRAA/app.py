
import streamlit as st
import tempfile
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

from abstraction_layer import RAGSystemBuilder

# ────────────────────────────────────────────────
# Initialize builder (cached)
# ────────────────────────────────────────────────

@st.cache_resource
def get_builder():
    return RAGSystemBuilder("config.ini")

builder = get_builder()
config = builder.get_config()

PROVIDERS_LLM = builder.get_llm_providers()
PROVIDERS_EMB = builder.get_embedding_providers()

# ────────────────────────────────────────────────
# Session state (much cleaner)
# ────────────────────────────────────────────────

if "cfg" not in st.session_state:
    st.session_state.cfg = {
        "llm": {"provider": config.get("llm", "provider"), "model": "", "key": ""},
        "embed": {"provider": config.get("embeddings", "provider"), "model": "", "key": ""},
        "ready": False,
        "docs_processed": False,
        "vector_store": None,
        "chat": []
    }

# ────────────────────────────────────────────────
# Page config + custom user image
# ────────────────────────────────────────────────

user_img = config.get("ui", "user_image_url", "").strip()
page_icon = config.get("ui", "page_icon", "📖")

if user_img:
    st.set_page_config(page_title=config.get("ui", "page_title", "AUTOSAR Q&A"), page_icon=page_icon, layout="wide")
    st.image(user_img, width=120)
else:
    st.set_page_config(page_title="AUTOSAR Q&A", page_icon=page_icon, layout="wide")

st.title("AUTOSAR Document Q&A")

# ────────────────────────────────────────────────
# Sidebar – Provider selection
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Settings")

    # LLM
    p_llm = st.selectbox("LLM", PROVIDERS_LLM, index=PROVIDERS_LLM.index(st.session_state.cfg["llm"]["provider"]), key="llm_p")
    if p_llm != st.session_state.cfg["llm"]["provider"]:
        st.session_state.cfg["llm"]["provider"] = p_llm
        st.session_state.cfg["ready"] = False

    # Embeddings
    p_emb = st.selectbox("Embedding", PROVIDERS_EMB, index=PROVIDERS_EMB.index(st.session_state.cfg["embed"]["provider"]), key="emb_p")
    if p_emb != st.session_state.cfg["embed"]["provider"]:
        st.session_state.cfg["embed"]["provider"] = p_emb
        st.session_state.cfg["ready"] = False

    if st.button("Initialize Models", type="primary"):
        with st.spinner("Loading..."):
            try:
                st.session_state.llm = builder.create_llm(
                    provider=p_llm,
                    api_key=st.session_state.cfg["llm"]["key"] or None
                )
                st.session_state.embeddings = builder.create_embeddings(
                    provider=p_emb,
                    api_key=st.session_state.cfg["embed"]["key"] or None
                )
                st.session_state.cfg["ready"] = True
                st.success("Models ready")
            except Exception as e:
                st.error(f"Failed: {str(e)[:120]}")

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Upload PDFs", "Ask Questions"])

with tab1:
    if not st.session_state.cfg["ready"]:
        st.info("Please initialize models in sidebar first.")
    else:
        files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
        if files and st.button("Process"):
            with st.spinner("Processing..."):
                all_docs = []
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        loader = PyPDFLoader(tmp.name)
                        all_docs.extend(loader.load())
                    os.unlink(tmp.name)

                chunks = builder.process_documents(all_docs)
                vs = builder.create_vector_store(chunks, st.session_state.embeddings)
                st.session_state.vector_store = vs
                st.session_state.retriever = vs.as_retriever(**builder.get_retriever_config())
                st.session_state.cfg["docs_processed"] = True
                st.success(f"Processed {len(chunks)} chunks")

with tab2:
    if not st.session_state.cfg.get("docs_processed"):
        st.info("Upload & process documents first.")
    else:
        # Very simple chat
        for q, a in st.session_state.cfg["chat"]:
            st.write(f"**Q:** {q}")
            st.write(a)
            st.markdown("---")

        question = st.text_area("Your question", height=100)
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"Answer using only this context:\n{context}\n\nQuestion:"),
                    ("human", "{question}")
                ])

                chain = prompt | st.session_state.llm | StrOutputParser()
                answer = chain.invoke({"question": question})

                st.session_state.cfg["chat"].append((question, answer))
                st.rerun()