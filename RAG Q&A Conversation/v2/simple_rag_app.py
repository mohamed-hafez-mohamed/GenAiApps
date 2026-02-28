"""
Configurable RAG Application (Simplified - No Agent)
Uses abstraction layer for easy provider switching via config.ini
Works with all LangChain versions
"""

import streamlit as st
import tempfile
import os
import time
from typing import List, Dict, Any
from datetime import datetime
import base64
from pathlib import Path
# Import abstraction layer
from abstraction_layer import RAGSystemBuilder

# LangChain imports for document loading
from langchain_community.document_loaders import PyPDFLoader

# LangChain imports for chains
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZE ABSTRACTION LAYER
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_rag_builder():
    """Initialize and cache RAG system builder."""
    return RAGSystemBuilder("config.ini")

builder = get_rag_builder()
config = builder.get_config()

# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (SIMPLIFIED)
# ════════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Load simplified custom CSS"""
    if not config.getboolean("ui", "enable_animations", True):
        return
    
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        .big-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            text-align: center;
            padding: 1rem 0;
        }
        .custom-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents."""
    return "\n\n".join(f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs))

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def get_icon_html(icon_value):
    """Generate HTML/emoji for icon, handling both images and emojis"""
    # If it's an image file path
    if icon_value.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".gif", ".ico")):
        try:
            icon_path = Path(icon_value)
            if icon_path.exists():
                # Convert image to base64
                with open(icon_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                
                # Determine MIME type
                if icon_value.endswith(".svg"):
                    mime_type = "image/svg+xml"
                elif icon_value.endswith(".jpg") or icon_value.endswith(".jpeg"):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "image/png"
                
                return f'<img src="data:{mime_type};base64,{img_base64}" style="width: 32px; height: 32px; vertical-align: middle; margin-right: 10px;">'
            else:
                # File doesn't exist, fallback to emoji
                return "🤖 "
        except Exception as e:
            print(f"Error loading icon: {e}")
            return "🤖 "  # Fallback emoji
    else:
        # It's an emoji or text
        return f"{icon_value} "
# ════════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=config.get("ui", "page_title", "Garraio RAG System"),
    page_icon="GARRAIO.png",
    layout=config.get("ui", "layout", "wide"),
    initial_sidebar_state=config.get("ui", "sidebar_state", "expanded")
)

# Load custom CSS if enabled
if config.getboolean("ui", "enable_animations", True):
    load_custom_css()

# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════════

if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vector_database' not in st.session_state:
    st.session_state.vector_database = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = "default_session"

# Ensure history_retriever is always present in session_state
if 'history_retriever' not in st.session_state:
    st.session_state.history_retriever = None

# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════

icon_config = config.get("ui", "page_icon", "🤖")
icon_html = get_icon_html(icon_config)

if config.getboolean("ui", "enable_animations", True):
    st.markdown(
        f'<h1 class="big-title">{icon_html}{config.get("ui", "page_title", "Garraio RAG System")}</h1>',
        unsafe_allow_html=True
    )
else:
    # For non-animated version, we need a different approach
    col1, col2 = st.columns([1, 20])
    with col1:
        # Try to display image or emoji
        if icon_config.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            try:
                st.image(icon_config, width=32)
            except:
                st.write(icon_config if icon_config else "🤖")
        else:
            st.write(icon_config if icon_config else "🤖")
    with col2:
        st.title(config.get("ui", "page_title", "Garraio RAG System"))

st.markdown("**Configurable RAG System** - Switch providers via `config.ini`")

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # Display current providers
    with st.expander("📋 Current Providers", expanded=True):
        llm_provider = config.get("llm", "provider", "groq")
        embed_provider = config.get("embeddings", "provider", "huggingface")
        vector_provider = config.get("vector_store", "provider", "chroma")
        
        st.info(f"**LLM:** {llm_provider}")
        st.info(f"**Embeddings:** {embed_provider}")
        st.info(f"**Vector Store:** {vector_provider}")
        st.caption("Edit `config.ini` to change")
    
    st.markdown("---")
    
    # API Key input
    with st.expander("🔑 API Configuration", expanded=True):
        api_key_input = st.text_input(
            f"{llm_provider.upper()} API Key",
            type="password",
            help=f"Enter your {llm_provider} API key"
        )
        
        if api_key_input:
            st.success("✅ API key configured")
            api_key_to_use = api_key_input
        else:
            st.warning("⚠️ API key required")
            api_key_to_use = None
        
        # Embedding API key if different provider
        if embed_provider in ["openai", "cohere"] and embed_provider != llm_provider:
            embed_api_key = st.text_input(
                f"{embed_provider.upper()} API Key",
                type="password",
                help=f"Enter your {embed_provider} API key"
            )
        else:
            embed_api_key = api_key_input
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    
    if st.session_state.llm_model:
        st.success("🤖 LLM Ready")
    else:
        st.info("🤖 LLM Pending")
    
    if st.session_state.rag_chain:
        st.success("🔗 RAG Chain Ready")
    else:
        st.info("🔗 Chain Pending")
    
    if st.session_state.vector_database:
        st.success("💾 Database Ready")
    else:
        st.info("💾 No Documents")

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZE COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

if api_key_to_use and not st.session_state.llm_model:
    try:
        with st.spinner(f"🚀 Initializing {llm_provider} LLM..."):
            st.session_state.llm_model = builder.create_llm(api_key=api_key_to_use)
            st.sidebar.success(f"✅ {llm_provider} LLM ready")
    except Exception as e:
        st.sidebar.error(f"❌ LLM Error: {e}")
        st.stop()

# Create embeddings if needed
if api_key_to_use and not st.session_state.embeddings:
    try:
        with st.spinner(f"🚀 Initializing {embed_provider} embeddings..."):
            st.session_state.embeddings = builder.create_embeddings(api_key=embed_api_key)
            st.sidebar.success(f"✅ {embed_provider} embeddings ready")
    except Exception as e:
        st.sidebar.error(f"❌ Embeddings Error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📤 Upload Documents",
    "💬 Chat",
    "📜 History"
])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1: UPLOAD
# ────────────────────────────────────────────────────────────────────────────────

with tab1:
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    st.markdown("### 📚 Document Upload")
    
    session_id = st.text_input(
        "Session Name",
        value=st.session_state.current_session_id
    )
    st.session_state.current_session_id = session_id
    
    uploaded_files = st.file_uploader(
        "📎 Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.llm_model and st.session_state.embeddings:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            documents = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load PDFs
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Loading {file.name}...")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    st.success(f"✅ {file.name}: {len(docs)} pages")
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                progress_bar.progress((idx + 1) / (len(uploaded_files) * 2))
            
            if documents:
                # Split documents
                status_text.text("Splitting documents...")
                split_docs = builder.process_documents(documents)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Pages", len(documents))
                with col2:
                    st.metric("✂️ Chunks", len(split_docs))
                
                # Show first few chunks
                with st.expander("Preview chunks"):
                    for i, chunk in enumerate(split_docs[:3]):
                        st.write(f"**Chunk {i+1}:**")
                        st.code(chunk.page_content)
        
                # Create vector store
                status_text.text(f"Creating {vector_provider} vector store...")
                vector_database = builder.create_vector_store(
                    split_docs,
                    st.session_state.embeddings
                )
                
                st.session_state.vector_database = vector_database
                
                # Create retriever
                retriever_config = builder.get_retriever_config()
                retriever = vector_database.as_retriever(**retriever_config)
                
                # Create RAG chain
                status_text.text("Building RAG chain...")
                
               # Query rephrasing (this part is fine)
                system_message_for_question_rephrasing = (
                    "Given a chat history and the latest user question, "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do not answer the question, "
                    "just reformulate it if needed otherwise return it as is."
                )
                
                query_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_message_for_question_rephrasing),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                
                # History-aware retriever
                history_retriever = query_prompt | st.session_state.llm_model | StrOutputParser() | retriever
                
                # IMPROVED: Stricter answer generation
                system_message_for_answer = (
                """You are a document question-answering assistant with STRICT limitations.
                
                CRITICAL RULES:
                1. ONLY use information from the Context section below
                2. DO NOT use external knowledge or make assumptions
                3. If the context lacks the answer, say: "I cannot find this information in the provided documents."
                4. Be concise (2-3 sentences maximum)
                5. Stay factual - only state what's explicitly in the context
                
                Context from Retrieved Documents:
                {context}
                
                Remember: Answer ONLY from the context above. If you're unsure or the answer isn't there, explicitly say so.
                """
                )
                
                answer_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_message_for_answer),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                
                # Document chain
                doc_chain = answer_prompt | st.session_state.llm_model | StrOutputParser()
                
                # Complete RAG chain
                rag_chain = (
                    RunnablePassthrough.assign(
                        context=lambda x: format_docs(history_retriever.invoke({
                            "input": x["input"],
                            "chat_history": x.get("chat_history", [])
                        })),
                        chat_history=lambda x: x.get("chat_history", []),
                        input=lambda x: x["input"]
                    )
                    | doc_chain
                )
                
                # Wrap with history
                st.session_state.rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
                
                st.session_state.docs_processed = True
                
                progress_bar.progress(1.0)
                status_text.text("")
                st.success("✅ Documents processed!")
                st.balloons()
                time.sleep(1)
                st.rerun()
    
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2: CHAT
# ────────────────────────────────────────────────────────────────────────────────

with tab2:
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    if not st.session_state.llm_model:
        st.warning("⚠️ Configure API key first")
    elif not st.session_state.docs_processed:
        st.info("📁 Upload documents first")
    else:
        st.markdown("### 💬 Chat with Documents")
        
        # Show history
        if session_id in st.session_state.store:
            history = st.session_state.store[session_id]
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 **You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.write(f"🤖 **Assistant:** {msg.content}")
                st.markdown("---")
        
        user_question = st.text_area("💭 Your Question:", height=100)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ask_button = st.button("🚀 Ask", disabled=not user_question, use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button and session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
            st.rerun()
        
        if ask_button and user_question:
            st.write(f"👤 **You:** {user_question}")
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_question},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.write(f"🤖 **Assistant:** {response}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# TAB 3: HISTORY
# ────────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📜 Conversation History")
    
    if session_id in st.session_state.store:
        history = st.session_state.store[session_id]
        if history.messages:
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 **You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.write(f"🤖 **Assistant:** {msg.content}")
                st.markdown("---")
            
            if st.button("📥 Export", use_container_width=True):
                export = f"Session: {session_id}\n\n"
                for msg in history.messages:
                    role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                    export += f"{role}: {msg.content}\n\n"
                
                st.download_button(
                    "💾 Download",
                    export,
                    f"chat_{session_id}.txt",
                    use_container_width=True
                )
        else:
            st.info("No messages yet")
    else:
        st.info("No history available")

# Footer
st.markdown("---")
st.caption(f"🔧 Configurable RAG System | {llm_provider} + {embed_provider} + {vector_provider}")
