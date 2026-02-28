"""
Configurable Agent-Based RAG Application
Uses abstraction layer for easy provider switching via config.ini
"""

import streamlit as st
import tempfile
import os
import time
from typing import List, Dict, Any
from datetime import datetime

# Import abstraction layer
from abstraction_layer import RAGSystemBuilder

# LangChain imports for document loading
from langchain_community.document_loaders import PyPDFLoader

# LangChain imports for agent creation
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZE ABSTRACTION LAYER
# ════════════════════════════════════════════════════════════════════════════════

# Create RAG system builder with configuration file
@st.cache_resource
def get_rag_builder():
    """Initialize and cache RAG system builder."""
    return RAGSystemBuilder("config.ini")

builder = get_rag_builder()
config = builder.get_config()

# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS (FROM CONFIG)
# ════════════════════════════════════════════════════════════════════════════════

def load_custom_css():
    """Load custom CSS - theme configurable via config.ini"""
    theme = config.get("ui", "theme", "purple_gradient")
    
    # Theme color schemes
    themes = {
        "purple_gradient": {
            "main": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "user": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "ai": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "process": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "success": "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
        },
        "blue_ocean": {
            "main": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
            "user": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
            "ai": "linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%)",
            "process": "linear-gradient(135deg, #00b4db 0%, #0083b0 100%)",
            "success": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
        },
        "green_forest": {
            "main": "linear-gradient(135deg, #134e5e 0%, #71b280 100%)",
            "user": "linear-gradient(135deg, #134e5e 0%, #71b280 100%)",
            "ai": "linear-gradient(135deg, #56ab2f 0%, #a8e063 100%)",
            "process": "linear-gradient(135deg, #02aab0 0%, #00cdac 100%)",
            "success": "linear-gradient(135deg, #7ec850 0%, #56c596 100%)"
        },
        "orange_sunset": {
            "main": "linear-gradient(135deg, #ff6a00 0%, #ee0979 100%)",
            "user": "linear-gradient(135deg, #ff6a00 0%, #ee0979 100%)",
            "ai": "linear-gradient(135deg, #f857a6 0%, #ff5858 100%)",
            "process": "linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%)",
            "success": "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)"
        }
    }
    
    colors = themes.get(theme, themes["purple_gradient"])
    
    # Only load CSS if enabled in config
    if not config.getboolean("ui", "enable_animations", True):
        return
    
    st.markdown(f"""
        <style>
        .main {{
            background: {colors['main']};
            background-attachment: fixed;
        }}
        
        .big-title {{
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(120deg, #ffffff, #a8edea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 2rem 0;
            animation: fadeInDown 1s ease-in;
        }}
        
        .custom-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            margin: 1rem 0;
            animation: fadeIn 0.8s ease-in;
        }}
        
        .user-message {{
            background: {colors['user']};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 1rem 0;
            max-width: 80%;
            float: right;
            clear: both;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .ai-message {{
            background: {colors['ai']};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 1rem 0;
            max-width: 80%;
            float: left;
            clear: both;
            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.5rem;
        }}
        
        .status-ready {{
            background: {colors['success']};
            color: white;
        }}
        
        .thinking {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: {colors['process']};
            color: white;
            border-radius: 20px;
            font-weight: 600;
            animation: pulse 2s ease-in-out infinite;
        }}
        
        .tool-used {{
            display: inline-block;
            padding: 0.3rem 0.8rem;
            background: {colors['success']};
            color: white;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 0.2rem;
            font-weight: 500;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents."""
    return "\n\n".join(f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs))

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_agent_tools(retriever) -> List[Tool]:
    """Create agent tools based on configuration."""
    enabled_tools = eval(config.get("agent", "enabled_tools", 
        '["search_documents", "summarize_documents", "get_document_stats"]'))
    
    tools = []
    
    if "search_documents" in enabled_tools:
        def search_documents(query: str) -> str:
            """Search through uploaded PDF documents."""
            try:
                docs = retriever.get_relevant_documents(query)
                return format_docs(docs) if docs else "No relevant documents found."
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.append(Tool(
            name="search_documents",
            func=search_documents,
            description="Search through uploaded PDF documents for specific information."
        ))
    
    if "summarize_documents" in enabled_tools:
        def summarize_documents(topic: str) -> str:
            """Get summary of documents on a topic."""
            try:
                docs = retriever.get_relevant_documents(topic, k=5)
                if not docs:
                    return "No documents found on this topic."
                content = format_docs(docs)
                return f"Summary: {content[:1000]}..."
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.append(Tool(
            name="summarize_documents",
            func=summarize_documents,
            description="Get a summary of documents related to a topic."
        ))
    
    if "get_document_stats" in enabled_tools:
        def get_document_stats(query: str = "") -> str:
            """Get document statistics."""
            if st.session_state.vector_database:
                try:
                    collection = st.session_state.vector_database._collection
                    count = collection.count()
                    return f"Database contains {count} text chunks."
                except:
                    return "Statistics available after upload."
            return "No documents uploaded."
        
        tools.append(Tool(
            name="get_document_stats",
            func=get_document_stats,
            description="Get statistics about the document collection."
        ))
    
    return tools

def create_agent_with_tools(llm, tools: List[Tool]):
    """Create agent with tools."""
    system_message = SystemMessage(
        content="""You are an intelligent document assistant agent.
        
Use the available tools to find information in documents.
Provide concise, accurate answers based on the document content.
If information isn't in the documents, say so clearly.
Use 3-4 sentences maximum for answers.
"""
    )
    
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.getboolean("agent", "verbose", True),
        handle_parsing_errors=config.getboolean("agent", "handle_parsing_errors", True),
        max_iterations=config.getint("agent", "max_iterations", 3),
        return_intermediate_steps=config.getboolean("agent", "return_intermediate_steps", True)
    )

# ════════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=config.get("ui", "page_title", "RAG System"),
    page_icon=config.get("ui", "page_icon", "🤖"),
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
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = "default_session"
if 'agent_thoughts' not in st.session_state:
    st.session_state.agent_thoughts = []

# ════════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════════

if config.getboolean("ui", "enable_animations", True):
    st.markdown(
        f'<h1 class="big-title">{config.get("ui", "page_icon", "🤖")} {config.get("ui", "page_title", "RAG System")}</h1>',
        unsafe_allow_html=True
    )
else:
    st.title(f'{config.get("ui", "page_icon", "🤖")} {config.get("ui", "page_title", "RAG System")}')

st.markdown("**Powered by Configurable Abstraction Layer** - Switch providers via `config.ini`")

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # Display current providers from config
    with st.expander("📋 Current Providers", expanded=True):
        llm_provider = config.get("llm", "provider", "groq")
        embed_provider = config.get("embeddings", "provider", "huggingface")
        vector_provider = config.get("vector_store", "provider", "chroma")
        
        st.info(f"**LLM:** {llm_provider}")
        st.info(f"**Embeddings:** {embed_provider}")
        st.info(f"**Vector Store:** {vector_provider}")
        
        st.caption("Edit `config.ini` to change providers")
    
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
        st.markdown('<div class="status-badge status-ready">🤖 LLM Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">🤖 LLM Pending</div>', unsafe_allow_html=True)
    
    if st.session_state.agent_executor:
        st.markdown('<div class="status-badge status-ready">🧠 Agent Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">🧠 Agent Pending</div>', unsafe_allow_html=True)
    
    if st.session_state.vector_database:
        st.markdown('<div class="status-badge status-ready">💾 Database Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge">💾 No Documents</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Available providers info
    with st.expander("ℹ️ Available Providers"):
        st.write("**LLM Providers:**")
        for provider in builder.get_available_llm_providers():
            st.write(f"• {provider}")
        
        st.write("**Embedding Providers:**")
        for provider in builder.get_available_embedding_providers():
            st.write(f"• {provider}")
        
        st.write("**Vector Stores:**")
        for provider in builder.get_available_vector_store_providers():
            st.write(f"• {provider}")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload Documents",
    "💬 Chat with Agent",
    "📜 History",
    "🔍 Insights"
])

# TAB 1: UPLOAD
with tab1:
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    st.markdown("### 📚 Document Upload Center")
    
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
                status_text.text(f"Processing {file.name}...")
                
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
                # Split documents using abstraction layer
                status_text.text("Splitting documents...")
                split_docs = builder.process_documents(documents)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Pages", len(documents))
                with col2:
                    st.metric("✂️ Chunks", len(split_docs))
                
                # Create vector store using abstraction layer
                status_text.text(f"Creating {vector_provider} vector store...")
                vector_database = builder.create_vector_store(
                    split_docs,
                    st.session_state.embeddings
                )
                
                st.session_state.vector_database = vector_database
                
                # Create retriever with config
                retriever_config = builder.get_retriever_config()
                retriever = vector_database.as_retriever(**retriever_config)
                
                # Create agent
                tools = create_agent_tools(retriever)
                agent_executor = create_agent_with_tools(st.session_state.llm_model, tools)
                
                st.session_state.agent_executor = RunnableWithMessageHistory(
                    agent_executor,
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

# TAB 2: CHAT
with tab2:
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    if not st.session_state.llm_model:
        st.warning("⚠️ Configure API key first")
    elif not st.session_state.docs_processed:
        st.info("📁 Upload documents first")
    else:
        st.markdown("### 💬 Chat with Agent")
        
        # Show history
        if session_id in st.session_state.store:
            history = st.session_state.store[session_id]
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    if config.getboolean("ui", "enable_chat_bubbles", True):
                        st.markdown(f'<div class="user-message">👤 {msg.content}</div>', unsafe_allow_html=True)
                    else:
                        st.write(f"👤 **You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    if config.getboolean("ui", "enable_chat_bubbles", True):
                        st.markdown(f'<div class="ai-message">🤖 {msg.content}</div>', unsafe_allow_html=True)
                    else:
                        st.write(f"🤖 **Agent:** {msg.content}")
        
        st.markdown("---")
        
        user_question = st.text_area("💭 Your Question:", height=100)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ask_button = st.button("🚀 Ask Agent", disabled=not user_question, use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button and session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
            st.rerun()
        
        if ask_button and user_question:
            if config.getboolean("ui", "enable_chat_bubbles", True):
                st.markdown(f'<div class="user-message">👤 {user_question}</div>', unsafe_allow_html=True)
            
            thinking = st.empty()
            if config.getboolean("ui", "enable_animations", True):
                thinking.markdown('<div class="thinking">🧠 Thinking...</div>', unsafe_allow_html=True)
            
            try:
                result = st.session_state.agent_executor.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}}
                )
                
                response = result.get('output', 'No response')
                
                thinking.empty()
                
                # Show tools used
                if config.getboolean("ui", "show_tool_usage", True):
                    steps = result.get('intermediate_steps', [])
                    if steps:
                        tools_used = []
                        for step in steps:
                            if len(step) >= 1 and hasattr(step[0], 'tool'):
                                tools_used.append(step[0].tool)
                        
                        if tools_used:
                            tools_html = "".join([f'<span class="tool-used">🔧 {t}</span>' for t in tools_used])
                            st.markdown(f'<div>{tools_html}</div>', unsafe_allow_html=True)
                
                # Show response
                if config.getboolean("ui", "enable_chat_bubbles", True):
                    st.markdown(f'<div class="ai-message">🤖 {response}</div>', unsafe_allow_html=True)
                else:
                    st.write(f"🤖 **Agent:** {response}")
                
            except Exception as e:
                thinking.empty()
                st.error(f"❌ Error: {e}")
    
    if config.getboolean("ui", "enable_animations", True):
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: HISTORY
with tab3:
    st.markdown("### 📜 Conversation History")
    
    if session_id in st.session_state.store:
        history = st.session_state.store[session_id]
        if history.messages:
            for msg in history.messages:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 **You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.write(f"🤖 **Agent:** {msg.content}")
                st.markdown("---")
            
            if st.button("📥 Export", use_container_width=True):
                export = f"Session: {session_id}\n\n"
                for msg in history.messages:
                    role = "You" if isinstance(msg, HumanMessage) else "Agent"
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

# TAB 4: INSIGHTS
with tab4:
    st.markdown("### 🔍 Configuration & Insights")
    
    st.markdown("**Current Configuration:**")
    st.json({
        "llm_provider": config.get("llm", "provider"),
        "embedding_provider": config.get("embeddings", "provider"),
        "vector_store": config.get("vector_store", "provider"),
        "chunk_size": config.getint("document_processing", "chunk_size"),
        "retrieval_k": config.getint("document_processing", "retrieval_k"),
        "theme": config.get("ui", "theme")
    })
    
    st.markdown("---")
    st.info("💡 Edit `config.ini` to change any of these settings and restart the app")

# Footer
st.markdown("---")
st.caption(f"🔧 Configurable RAG System | Providers: {llm_provider}, {embed_provider}, {vector_provider}")
