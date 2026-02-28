"""
Document Q&A System - FINAL FIXED VERSION
With provider selection and separate API keys/tokens
"""

import streamlit as st
import tempfile
import os
import time
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from abstraction_layer import RAGSystemBuilder

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZE
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_rag_builder():
    """Initialize RAG system builder."""
    return RAGSystemBuilder("config.ini")

builder = get_rag_builder()
config = builder.get_config()

# ════════════════════════════════════════════════════════════════════════════════
# PROVIDER LISTS
# ════════════════════════════════════════════════════════════════════════════════

# Available providers from abstraction layer
AVAILABLE_LLM_PROVIDERS = builder.get_available_llm_providers()
AVAILABLE_EMBEDDING_PROVIDERS = builder.get_available_embedding_providers()

# Provider information - UPDATED with proper token requirements
PROVIDER_INFO = {
    # LLM Providers
    "groq": {
        "name": "Groq",
        "needs_key": True,
        "key_name": "API Key",
        "help": "Get from https://console.groq.com",
        "models": ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    },
    "openai": {
        "name": "OpenAI",
        "needs_key": True,
        "key_name": "API Key",
        "help": "Get from https://platform.openai.com/api-keys",
        "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "needs_key": True,
        "key_name": "API Key",
        "help": "Get from https://console.anthropic.com",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    },
    "ollama": {
        "name": "Ollama (Local)",
        "needs_key": False,
        "key_name": "Not needed",
        "help": "Run Ollama locally at http://localhost:11434",
        "models": ["llama2", "mistral", "mixtral", "codellama"]
    },
    "huggingface": {
        "name": "Hugging Face",
        "needs_key": True,
        "key_name": "HF Token",
        "help": "Get from https://huggingface.co/settings/tokens",
        "models": ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-xxl"]
    },
    
    # Embedding Providers
    "huggingface": {
        "name": "Hugging Face Embeddings",
        "needs_key": True,
        "key_name": "HF Token",
        "help": "Get from https://huggingface.co/settings/tokens (needed for some models)",
        "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-MiniLM-L12-v2"]
    },
    "openai": {
        "name": "OpenAI Embeddings",
        "needs_key": True,
        "key_name": "API Key",
        "help": "Get from https://platform.openai.com/api-keys",
        "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    },
    "cohere": {
        "name": "Cohere Embeddings",
        "needs_key": True,
        "key_name": "API Key",
        "help": "Get from https://dashboard.cohere.com/api-keys",
        "models": ["embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0"]
    },
    "ollama": {
        "name": "Ollama Embeddings (Local)",
        "needs_key": False,
        "key_name": "Not needed",
        "help": "Run Ollama locally at http://localhost:11434",
        "models": ["nomic-embed-text", "mxbai-embed-large"]
    }
}

# ════════════════════════════════════════════════════════════════════════════════
# FIXED VECTOR STORE AND RETRIEVAL FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_vector_store(documents, embeddings):
    """Create vector store with proper error handling."""
    try:
        # Try Chroma first
        from langchain_chroma import Chroma
        
        persist_directory = "./chroma_db_working"
        collection_name = "rag_documents"
        
        os.makedirs(persist_directory, exist_ok=True)
        
        st.info(f"Creating vector store with {len(documents)} chunks...")
        
        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Verify it worked
        try:
            # Try to retrieve something to verify
            test_results = vector_store.similarity_search("test", k=1)
            if test_results:
                st.success(f"✅ Vector store created and verified with {len(documents)} chunks")
            else:
                st.warning("⚠️ Vector store created but test retrieval returned no results")
        except:
            st.success(f"✅ Vector store created with {len(documents)} chunks")
        
        return vector_store
        
    except Exception as e:
        st.error(f"❌ Chroma failed: {str(e)[:100]}")
        # Fallback to FAISS
        try:
            from langchain_community.vectorstores import FAISS
            st.info("Using FAISS as fallback...")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local("./faiss_index_working")
            st.success("✅ FAISS vector store created")
            return vector_store
        except Exception as e2:
            st.error(f"❌ All vector stores failed: {str(e2)[:100]}")
            raise

def format_context_for_llm(docs):
    """Format documents properly for the LLM - CRITICAL FIX"""
    if not docs:
        return "No specific documents were retrieved for this query."
    
    context_parts = []
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        if not content:
            continue
            
        source = doc.metadata.get('source', 'Document')
        page = doc.metadata.get('page', 'N/A')
        
        # Clean source name
        source_name = os.path.basename(str(source))
        
        context_parts.append(
            f"[Document {i+1} from '{source_name}', page {page}]:\n"
            f"{content}\n"
            f"--- End of Document {i+1} ---\n"
        )
    
    if not context_parts:
        return "Documents were retrieved but contained no readable text content."
    
    return "\n".join(context_parts)

# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - IMPROVED
# ════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a document analysis assistant. Answer questions using ONLY the document context provided below.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

IMPORTANT RULES:
1. BASE YOUR ANSWER ONLY ON THE DOCUMENT CONTEXT ABOVE
2. If the context contains relevant information, use it to answer the question
3. If the context doesn't have the exact answer, say what information IS available
4. NEVER say "no context provided" or similar - always use what's available
5. Reference specific parts of the documents when possible
6. If documents don't answer directly, explain what they DO contain

ANSWER:"""

# ════════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Document Q&A - Working",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Document Q&A System")
st.markdown("Upload PDFs and ask questions - **With Provider Selection**")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE - CORRECT INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

# Initialize all session state variables safely
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_docs_info' not in st.session_state:
    st.session_state.processed_docs_info = {}
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'selected_llm_provider' not in st.session_state:
    st.session_state.selected_llm_provider = "groq"
if 'selected_embedding_provider' not in st.session_state:
    st.session_state.selected_embedding_provider = "huggingface"
if 'llm_api_key' not in st.session_state:
    st.session_state.llm_api_key = ""
if 'embedding_api_key' not in st.session_state:
    st.session_state.embedding_api_key = ""
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = ""
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = ""

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR - PROVIDER SELECTION
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔧 Provider Configuration")
    
    # LLM Provider Selection
    st.subheader("🤖 LLM Provider")
    
    selected_llm = st.selectbox(
        "Choose LLM Provider",
        AVAILABLE_LLM_PROVIDERS,
        index=AVAILABLE_LLM_PROVIDERS.index(st.session_state.selected_llm_provider) if st.session_state.selected_llm_provider in AVAILABLE_LLM_PROVIDERS else 0,
        format_func=lambda x: PROVIDER_INFO.get(x, {}).get("name", x.capitalize()),
        key="llm_provider_select"
    )
    
    if selected_llm != st.session_state.selected_llm_provider:
        st.session_state.selected_llm_provider = selected_llm
        st.session_state.llm_api_key = ""
        st.session_state.llm = None
    
    # LLM Model Selection
    if selected_llm in PROVIDER_INFO and "models" in PROVIDER_INFO[selected_llm]:
        default_model = config.get(f"llm.{selected_llm}", "model", "")
        available_models = PROVIDER_INFO[selected_llm]["models"]
        
        llm_model = st.selectbox(
            "LLM Model",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key="llm_model_select"
        )
        st.session_state.llm_model = llm_model
    
    # LLM API Key/Token Input
    provider_info = PROVIDER_INFO.get(selected_llm, {})
    if provider_info.get("needs_key", True):
        key_name = provider_info.get("key_name", "API Key")
        llm_api_key = st.text_input(
            f"{provider_info['name']} {key_name}",
            type="password",
            placeholder=f"Enter {provider_info['name']} {key_name}",
            help=provider_info["help"],
            value=st.session_state.llm_api_key,
            key="llm_api_key_input"
        )
        
        if llm_api_key != st.session_state.llm_api_key:
            st.session_state.llm_api_key = llm_api_key
            st.session_state.llm = None  # Reset LLM if key changes
        
        # Special note for Hugging Face
        if selected_llm == "huggingface" and not llm_api_key:
            st.warning("⚠️ HF Token required for accessing models")
    else:
        st.info(f"✅ {provider_info['name']} - no API key/token needed")
        st.session_state.llm_api_key = None
    
    st.markdown("---")
    
    # Embedding Provider Selection
    st.subheader("🔤 Embedding Provider")
    
    selected_embedding = st.selectbox(
        "Choose Embedding Provider",
        AVAILABLE_EMBEDDING_PROVIDERS,
        index=AVAILABLE_EMBEDDING_PROVIDERS.index(st.session_state.selected_embedding_provider) if st.session_state.selected_embedding_provider in AVAILABLE_EMBEDDING_PROVIDERS else 0,
        format_func=lambda x: PROVIDER_INFO.get(x, {}).get("name", x.capitalize()),
        key="embedding_provider_select"
    )
    
    if selected_embedding != st.session_state.selected_embedding_provider:
        st.session_state.selected_embedding_provider = selected_embedding
        st.session_state.embedding_api_key = ""
        st.session_state.embeddings = None
    
    # Embedding Model Selection
    if selected_embedding in PROVIDER_INFO and "models" in PROVIDER_INFO[selected_embedding]:
        default_model = config.get(f"embeddings.{selected_embedding}", "model", "")
        available_models = PROVIDER_INFO[selected_embedding]["models"]
        
        embedding_model = st.selectbox(
            "Embedding Model",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key="embedding_model_select"
        )
        st.session_state.embedding_model = embedding_model
    
    # Embedding API Key/Token Input
    embedding_info = PROVIDER_INFO.get(selected_embedding, {})
    if embedding_info.get("needs_key", True):
        key_name = embedding_info.get("key_name", "API Key")
        embedding_api_key = st.text_input(
            f"{embedding_info['name']} {key_name}",
            type="password",
            placeholder=f"Enter {embedding_info['name']} {key_name}",
            help=embedding_info["help"],
            value=st.session_state.embedding_api_key,
            key="embedding_api_key_input"
        )
        
        if embedding_api_key != st.session_state.embedding_api_key:
            st.session_state.embedding_api_key = embedding_api_key
            st.session_state.embeddings = None  # Reset embeddings if key changes
        
        # Special note for Hugging Face embeddings
        if selected_embedding == "huggingface":
            if not embedding_api_key:
                st.warning("⚠️ HF Token recommended for better model access")
            else:
                st.info("✅ HF Token provided - will use for embeddings")
    else:
        st.info(f"✅ {embedding_info['name']} - no API key/token needed")
        st.session_state.embedding_api_key = None
    
    st.markdown("---")
    
    # Initialize button
    init_disabled = False
    warning_messages = []
    
    # Check LLM key requirement
    if provider_info.get("needs_key", True) and not st.session_state.llm_api_key:
        warning_messages.append(f"{provider_info['name']} {provider_info.get('key_name', 'API Key')}")
        init_disabled = True
    
    # Check embedding key requirement - for Hugging Face, we'll make it optional but warn
    if embedding_info.get("needs_key", True) and not st.session_state.embedding_api_key:
        if selected_embedding == "huggingface":
            # For Hugging Face embeddings, we'll allow it but warn
            warning_messages.append(f"{embedding_info['name']} token (optional but recommended)")
            # Don't disable for Hugging Face - it might work without token for some models
            init_disabled = False if not init_disabled else init_disabled
        else:
            warning_messages.append(f"{embedding_info['name']} {embedding_info.get('key_name', 'API Key')}")
            init_disabled = True
    
    if st.button("🚀 Initialize AI Components", 
                 type="primary", 
                 use_container_width=True, 
                 key="init_btn",
                 disabled=init_disabled):
        
        with st.spinner("Initializing AI components..."):
            try:
                # Set environment variables for API keys/tokens
                if st.session_state.llm_api_key:
                    if selected_llm == "huggingface":
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.llm_api_key
                    elif selected_llm == "openai":
                        os.environ["OPENAI_API_KEY"] = st.session_state.llm_api_key
                    elif selected_llm == "anthropic":
                        os.environ["ANTHROPIC_API_KEY"] = st.session_state.llm_api_key
                    elif selected_llm == "groq":
                        os.environ["GROQ_API_KEY"] = st.session_state.llm_api_key
                
                if st.session_state.embedding_api_key:
                    if selected_embedding == "huggingface":
                        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.embedding_api_key
                    elif selected_embedding == "openai":
                        os.environ["OPENAI_API_KEY"] = st.session_state.embedding_api_key
                    elif selected_embedding == "cohere":
                        os.environ["COHERE_API_KEY"] = st.session_state.embedding_api_key
                
                # Get API keys for initialization
                llm_key = st.session_state.llm_api_key if provider_info.get("needs_key", True) else None
                embedding_key = st.session_state.embedding_api_key if embedding_info.get("needs_key", True) else None
                
                # Initialize LLM with provider selection
                st.session_state.llm = builder.create_llm(
                    api_key=llm_key,
                    provider=selected_llm
                )
                
                # Update model if specified
                if st.session_state.llm_model and hasattr(st.session_state.llm, 'model'):
                    st.session_state.llm.model = st.session_state.llm_model
                
                st.success(f"✅ {provider_info['name']} initialized")
                
                # Initialize embeddings with provider selection
                st.session_state.embeddings = builder.create_embeddings(
                    api_key=embedding_key,
                    provider=selected_embedding
                )
                
                # Update model if specified for embeddings
                if selected_embedding == "huggingface" and st.session_state.embedding_model:
                    if hasattr(st.session_state.embeddings, 'model_name'):
                        st.session_state.embeddings.model_name = st.session_state.embedding_model
                
                st.success(f"✅ {embedding_info['name']} initialized")
                
                # Reset document processing state
                st.session_state.documents_processed = False
                st.session_state.vector_store = None
                st.session_state.retriever = None
                
                st.balloons()
                st.success("✅ All AI components ready!")
                
            except Exception as e:
                error_msg = str(e)[:200]
                
                # Provide specific error messages
                if "API key" in error_msg or "api_key" in error_msg or "token" in error_msg:
                    if selected_llm == "huggingface" or selected_embedding == "huggingface":
                        st.error(f"❌ Hugging Face token may be invalid or required for this model")
                    else:
                        st.error(f"❌ Invalid or missing API key/token")
                elif "connection" in error_msg.lower() and selected_llm == "ollama":
                    st.error(f"❌ Cannot connect to Ollama. Make sure it's running at http://localhost:11434")
                elif "401" in error_msg:
                    st.error(f"❌ Authentication failed. Check your API key/token")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    st.error(f"❌ Model not found or inaccessible. Check model name and permissions")
                else:
                    st.error(f"❌ Error: {error_msg}")
    
    if warning_messages:
        st.warning(f"⚠️ Please provide: {', '.join(warning_messages)}")
    
    st.markdown("---")
    
    # Status
    st.header("📊 Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.llm:
            st.success("🤖 LLM Ready")
        else:
            st.warning("LLM: Not ready")
    
    with col2:
        if st.session_state.embeddings:
            st.success("🔤 Embed Ready")
        else:
            st.warning("Embed: Not ready")
    
    with col3:
        if st.session_state.documents_processed:
            files_count = len(st.session_state.processed_docs_info.get('files', []))
            st.success(f"📁 {files_count}")
        else:
            st.info("No docs")
    
    st.markdown("---")
    
    # Provider Summary
    with st.expander("🔍 Current Configuration", expanded=False):
        st.write(f"**LLM Provider:** {provider_info['name']}")
        if st.session_state.llm_model:
            st.write(f"**LLM Model:** {st.session_state.llm_model}")
        
        st.write(f"**Embedding Provider:** {embedding_info['name']}")
        if st.session_state.embedding_model:
            st.write(f"**Embedding Model:** {st.session_state.embedding_model}")
        
        if provider_info.get("needs_key", True):
            status = "✅ Provided" if st.session_state.llm_api_key else "❌ Missing"
            st.write(f"**LLM {provider_info.get('key_name', 'API Key')}:** {status}")
        else:
            st.write("**LLM API Key:** Not required")
        
        if embedding_info.get("needs_key", True):
            status = "✅ Provided" if st.session_state.embedding_api_key else "⚠️ Missing (may work for some models)"
            st.write(f"**Embedding {embedding_info.get('key_name', 'API Key')}:** {status}")
        else:
            st.write("**Embedding API Key:** Not required")
    
    st.markdown("---")
    
    # Environment Note
    with st.expander("🌐 Environment Setup", expanded=False):
        st.markdown("""
        ### For Hugging Face Models:
        
        If you're having issues with Hugging Face, try setting the token in your environment:
        
        ```bash
        # In terminal before running the app
        export HUGGINGFACEHUB_API_TOKEN="your_token_here"
        
        # Or create a .env file with:
        HUGGINGFACEHUB_API_TOKEN=your_token_here
        ```
        
        The app will use the token from the input field, but environment variables can help with certain models.
        """)
    
    st.markdown("---")
    
    # Actions
    if st.button("🔄 Reset All", use_container_width=True, key="reset_all"):
        keys_to_keep = ['chat_history']  # Keep chat history
        new_state = {}
        for key in keys_to_keep:
            if key in st.session_state:
                new_state[key] = st.session_state[key]
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        for key, value in new_state.items():
            st.session_state[key] = value
        
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ════════════════════════════════════════════════════════════════════════════════

# Create tabs
upload_tab, chat_tab = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

with upload_tab:
    st.header("Step 1: Upload PDF Documents")
    
    if not st.session_state.llm or not st.session_state.embeddings:
        st.warning("⚠️ Please initialize AI components using the sidebar first")
        st.info("""
        1. Select LLM and Embedding providers
        2. Enter API keys/tokens if required
        3. Click 'Initialize AI Components'
        """)
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select PDF documents to analyze",
            key="file_uploader"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True, key="process_btn"):
                with st.spinner("Processing documents..."):
                    all_docs = []
                    file_info = []
                    
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                            
                            file_info.append({
                                'name': file.name,
                                'pages': len(docs),
                                'size': file.size
                            })
                            
                            os.unlink(tmp_path)
                            st.success(f"✅ {file.name}: {len(docs)} pages")
                            
                        except Exception as e:
                            st.error(f"❌ {file.name}: {str(e)[:100]}")
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    
                    if all_docs:
                        # Split documents
                        split_docs = builder.process_documents(all_docs)
                        
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files", len(uploaded_files))
                        with col2:
                            st.metric("Pages", len(all_docs))
                        with col3:
                            st.metric("Chunks", len(split_docs))
                        
                        # Create vector store
                        try:
                            vector_store = create_vector_store(split_docs, st.session_state.embeddings)
                            
                            # Create retriever
                            retriever_config = builder.get_retriever_config()
                            retriever = vector_store.as_retriever(**retriever_config)
                            
                            # Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.retriever = retriever
                            st.session_state.processed_docs_info = {
                                'files': file_info,
                                'total_pages': len(all_docs),
                                'total_chunks': len(split_docs)
                            }
                            
                            # Create RAG function
                            prompt = ChatPromptTemplate.from_messages([
                                SystemMessage(content=SYSTEM_PROMPT),
                                ("human", "{question}")
                            ])
                            
                            def create_rag_function(llm, retriever):
                                chain = prompt | llm | StrOutputParser()
                                
                                def rag_function(question):
                                    # Retrieve documents
                                    docs = retriever.invoke(question)
                                    
                                    # Format context PROPERLY
                                    context = format_context_for_llm(docs)
                                    
                                    # Generate answer
                                    answer = chain.invoke({
                                        "context": context,
                                        "question": question
                                    })
                                    
                                    return answer, len(docs) if docs else 0
                                
                                return rag_function
                            
                            st.session_state.rag_function = create_rag_function(
                                st.session_state.llm,
                                retriever
                            )
                            
                            st.session_state.documents_processed = True
                            
                            st.balloons()
                            st.success("✅ Documents processed successfully!")
                            
                            # Show what to do next
                            with st.expander("🎯 Next Steps", expanded=True):
                                st.markdown("""
                                ### Ready to ask questions!
                                
                                1. Go to the **"Ask Questions"** tab
                                2. Try questions like:
                                   - "What is this document about?"
                                   - "What are the main points?"
                                   - "What topics are discussed?"
                                3. The system will find relevant information from your documents
                                """)
                            
                            time.sleep(2)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)[:200]}")

with chat_tab:
    st.header("Step 2: Ask Questions")
    
    if not st.session_state.documents_processed:
        st.info("📁 Please upload and process documents first")
        st.info("Go to the 'Upload Documents' tab")
    else:
        # Show document info
        with st.expander("📚 Loaded Documents", expanded=False):
            info = st.session_state.processed_docs_info
            st.write(f"**Files:** {len(info.get('files', []))}")
            st.write(f"**Pages:** {info.get('total_pages', 0)}")
            st.write(f"**Searchable chunks:** {info.get('total_chunks', 0)}")
            
            for file_info in info.get('files', []):
                st.write(f"• {file_info['name']} ({file_info['pages']} pages)")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation")
            for i, (question, answer, doc_count) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                    if doc_count > 0:
                        st.caption(f"📄 Found relevant content in {doc_count} document section(s)")
                    st.markdown("---")
        
        # Question input - using a form to avoid session state issues
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area(
                "Your question:",
                placeholder="Ask anything about your documents...\n\nExamples:\n• What is this document about?\n• What are the main topics?\n• What information is provided?",
                height=100,
                key="question_input_area"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit_btn = st.form_submit_button("Ask Question", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Process question when submitted
        if submit_btn and question:
            # Add to history with placeholder
            st.session_state.chat_history.append((question, "Thinking...", 0))
            
            # Show the question
            st.markdown(f"**You asked:** {question}")
            st.markdown("---")
            
            # Generate answer
            with st.spinner("🔍 Searching documents..."):
                try:
                    answer, doc_count = st.session_state.rag_function(question)
                    
                    # Update history
                    st.session_state.chat_history[-1] = (question, answer, doc_count)
                    
                    # Show answer
                    st.markdown(f"**Answer:** {answer}")
                    
                    # Show document usage
                    if doc_count > 0:
                        st.success(f"✅ Used information from {doc_count} document section(s)")
                    else:
                        st.info("ℹ️ General response based on document themes")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)[:150]}"
                    st.error(error_msg)
                    st.session_state.chat_history[-1] = (question, error_msg, 0)
            
            # Auto-scroll
            st.rerun()

# Footer
st.markdown("---")
st.caption(f"✅ **Document Q&A System** • Using {PROVIDER_INFO[st.session_state.selected_llm_provider]['name']} & {PROVIDER_INFO[st.session_state.selected_embedding_provider]['name']}")