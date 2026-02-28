import streamlit as st
import os
import time
import sys

from abstraction_layer import (
    lazy_llm,
    lazy_embeddings,
    build_vectorstore
)

from config_schema import (
    AVAILABLE_LLMS,
    AVAILABLE_EMBEDDINGS,
    AUTOSAR_RELEASES,
    RELEASE_FOLDER_MAP
)

from config_ui import PAGE_CONFIG

from agent import build_agent


st.set_page_config(layout=PAGE_CONFIG["layout"])
st.title(PAGE_CONFIG["title"])


# =============================
# Get the script directory (where this file is located)
# =============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {SCRIPT_DIR}")


# =============================
# Session State
# =============================
if "history" not in st.session_state:
    st.session_state.history = {}

if "agent" not in st.session_state:
    st.session_state.agent = None

if "initialization_status" not in st.session_state:
    st.session_state.initialization_status = None

if "vectorstore_stats" not in st.session_state:
    st.session_state.vectorstore_stats = None


# =============================
# Debug function to show path info
# =============================
def debug_path_info():
    """Display detailed path information for debugging"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Path Debug Info")
    
    # Current working directory
    cwd = os.getcwd()
    st.sidebar.write(f"**Current working directory:** `{cwd}`")
    
    # Script directory
    st.sidebar.write(f"**Script directory:** `{SCRIPT_DIR}`")
    
    # List all files and folders in script directory
    st.sidebar.write("**Contents of script directory:**")
    try:
        items = os.listdir(SCRIPT_DIR)
        for item in sorted(items):
            item_path = os.path.join(SCRIPT_DIR, item)
            item_type = "📁" if os.path.isdir(item_path) else "📄"
            st.sidebar.write(f"{item_type} `{item}`")
    except Exception as e:
        st.sidebar.write(f"Error listing directory: {e}")
    
    # Check specifically for the autosar folder in script directory
    target_folder = "autosar_public_docs_4_4_0"
    target_path = os.path.join(SCRIPT_DIR, target_folder)
    
    if os.path.exists(target_path):
        st.sidebar.success(f"✅ Target folder exists: `{target_path}`")
    else:
        st.sidebar.error(f"❌ Target folder NOT found: `{target_path}`")
        
        # Try case-insensitive search in script directory
        found = False
        for item in os.listdir(SCRIPT_DIR):
            item_path = os.path.join(SCRIPT_DIR, item)
            if os.path.isdir(item_path) and item.lower() == target_folder.lower():
                st.sidebar.warning(f"⚠️ Found with different case: `{item}` at `{item_path}`")
                found = True
                break
        
        if not found:
            # Search for any folder containing "autosar" in script directory
            autosar_folders = []
            for item in os.listdir(SCRIPT_DIR):
                item_path = os.path.join(SCRIPT_DIR, item)
                if os.path.isdir(item_path) and "autosar" in item.lower():
                    autosar_folders.append(item)
            
            if autosar_folders:
                st.sidebar.info(f"Found folders containing 'autosar': {autosar_folders}")


# =============================
# Helper function to get docs folder path (always relative to script directory)
# =============================
def get_docs_folder_path(release):
    """Return the path to the AUTOSAR documents folder based on release"""
    
    # Get folder name from the mapping
    folder_name = RELEASE_FOLDER_MAP.get(release)
    
    # If no mapping found, try a default pattern
    if folder_name is None:
        folder_name = f"autosar_public_docs_{release.lower()}"
    
    # Always look in the script directory
    folder_path = os.path.join(SCRIPT_DIR, folder_name)
    
    print(f"Looking for folder at: {folder_path}")
    
    # Check if folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"✅ Found folder at: {folder_path}")
        return folder_path
    
    # Try case-insensitive search in script directory
    for item in os.listdir(SCRIPT_DIR):
        item_path = os.path.join(SCRIPT_DIR, item)
        if os.path.isdir(item_path) and item.lower() == folder_name.lower():
            print(f"✅ Found folder with different case: {item_path}")
            return item_path
    
    print(f"❌ Folder not found: {folder_path}")
    return None


# =============================
# Sidebar Configuration
# =============================
with st.sidebar:

    st.header("⚙️ Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        list(AVAILABLE_LLMS.keys())
    )

    llm_model = st.selectbox(
        "LLM Model",
        AVAILABLE_LLMS[llm_provider]
    )

    llm_key = st.text_input(
        "LLM API Key (Not needed for Ollama)",
        type="password"
    )

    emb_provider = st.selectbox(
        "Embedding Provider",
        list(AVAILABLE_EMBEDDINGS.keys())
    )

    emb_model = st.selectbox(
        "Embedding Model",
        AVAILABLE_EMBEDDINGS[emb_provider]
    )

    emb_key = st.text_input(
        "Embedding API Key / HF Token",
        type="password"
    )

    release = st.selectbox(
        "AUTOSAR Release",
        AUTOSAR_RELEASES
    )

    # Add debug toggle
    show_debug = st.checkbox("🔧 Show Path Debug Info", value=False)
    
    if show_debug:
        debug_path_info()

    # Add manual folder input as fallback
    manual_folder = st.text_input(
        "Or enter folder name manually (if auto-detection fails)",
        placeholder="e.g., autosar_public_docs_4_4_0"
    )

    if st.button("🚀 Initialize Agent"):

        # Determine doc path
        if manual_folder:
            # If manual folder is provided, check if it's a full path or just a name
            if os.path.isabs(manual_folder):
                doc_path = manual_folder
            else:
                # If it's just a name, look in script directory
                doc_path = os.path.join(SCRIPT_DIR, manual_folder)
            st.info(f"Using manual folder: {doc_path}")
        else:
            # Get the docs folder path
            doc_path = get_docs_folder_path(release)
        
        if doc_path is None or not os.path.exists(doc_path):
            st.error(f"❌ Could not find documents folder for {release}")
            
            # Show the expected folder path
            expected_folder = RELEASE_FOLDER_MAP.get(release, f"autosar_public_docs_{release.lower()}")
            expected_path = os.path.join(SCRIPT_DIR, expected_folder)
            st.info(f"Expected path: `{expected_path}`")
            
            # List all folders in script directory
            try:
                all_items = os.listdir(SCRIPT_DIR)
                folders = [d for d in all_items if os.path.isdir(os.path.join(SCRIPT_DIR, d))]
                
                st.write(f"📂 **Available folders in script directory ({len(folders)}):**")
                for folder in sorted(folders):
                    folder_path = os.path.join(SCRIPT_DIR, folder)
                    st.write(f"   - `{folder}`")
                    st.write(f"     (full path: `{folder_path}`)")
                
                # Check if the folder exists but with different name
                expected_lower = expected_folder.lower()
                similar_folders = [f for f in folders if expected_lower in f.lower()]
                if similar_folders:
                    st.info(f"💡 Found similar folders: {similar_folders}")
            except Exception as e:
                st.error(f"Error listing directory: {e}")
            
            st.info("💡 Tip: You can manually enter the folder name in the text box above")
        else:
            # Show success message with full path
            st.success(f"✅ Found documents folder: {doc_path}")
            
            # Show warning about loading time
            st.warning(f"⚠️ Loading documents from '{os.path.basename(doc_path)}'... This may take a few minutes depending on the number of PDFs.")
            
            # Create containers for progress tracking
            progress_bar = st.progress(0)
            status_container = st.container()
            stats_container = st.container()
            
            try:
                with status_container:
                    st.subheader("📊 Initialization Progress")
                
                # Step 1: Initializing LLM
                with status_container:
                    st.info("🔄 Step 1/5: Initializing Language Model...")
                progress_bar.progress(10)
                time.sleep(0.5)  # Small delay for UX
                
                # Initialize LLM
                llm = lazy_llm(llm_provider, llm_model, llm_key)
                
                with status_container:
                    st.success("✅ Step 1/5: Language Model initialized successfully")
                
                # Step 2: Initializing Embeddings
                with status_container:
                    st.info("🔄 Step 2/5: Initializing Embedding Model...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                # Initialize Embeddings
                embeddings = lazy_embeddings(
                    emb_provider,
                    emb_model,
                    emb_key
                )
                
                with status_container:
                    st.success("✅ Step 2/5: Embedding Model initialized successfully")
                
                # Step 3: Loading Documents
                with status_container:
                    st.info(f"🔄 Step 3/5: Loading PDF documents from '{os.path.basename(doc_path)}'...")
                progress_bar.progress(40)
                
                # Create a custom loader with progress tracking
                from langchain_community.document_loaders import PyPDFDirectoryLoader
                loader = PyPDFDirectoryLoader(doc_path)
                
                # Load documents
                documents = loader.load()
                num_docs = len(documents)
                
                with status_container:
                    st.success(f"✅ Step 3/5: Loaded {num_docs} PDF documents successfully")
                
                # Step 4: Splitting Documents into Chunks
                with status_container:
                    st.info("🔄 Step 4/5: Splitting documents into smaller chunks...")
                progress_bar.progress(60)
                time.sleep(0.5)
                
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                chunks = splitter.split_documents(documents)
                num_chunks = len(chunks)
                
                with status_container:
                    st.success(f"✅ Step 4/5: Created {num_chunks} document chunks")
                
                # Show stats with safe division
                with stats_container:
                    st.subheader("📈 Document Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📄 PDF Documents", num_docs)
                    with col2:
                        st.metric("🔹 Text Chunks", num_chunks)
                    
                    # Safely calculate average chunk size
                    if num_chunks > 0:
                        avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / num_chunks
                        st.metric("📏 Average Chunk Size", f"{avg_chunk_size:.0f} characters")
                    else:
                        st.metric("📏 Average Chunk Size", "N/A (no chunks)")
                        st.warning("⚠️ No chunks were created from the documents. Check if documents contain text.")
                
                # Step 5: Creating Vector Store
                with status_container:
                    st.info("🔄 Step 5/5: Creating vector store and generating embeddings...")
                    st.info("⏳ This is the slowest step. Please wait...")
                progress_bar.progress(80)
                
                # Build vectorstore
                vectorstore = build_vectorstore(
                    doc_path,
                    embeddings
                )
                
                # Build agent
                st.session_state.agent = build_agent(
                    llm,
                    vectorstore
                )
                
                # Store stats in session state
                st.session_state.vectorstore_stats = {
                    "num_docs": num_docs,
                    "num_chunks": num_chunks,
                    "doc_path": os.path.basename(doc_path),
                    "full_path": doc_path,
                    "emb_provider": emb_provider,
                    "emb_model": emb_model
                }
                
                # Complete
                progress_bar.progress(100)
                with status_container:
                    st.success("✅ Step 5/5: Vector store created successfully!")
                    st.balloons()
                    st.success("🎉 **Initialization Complete! You can now ask questions about AUTOSAR.**")
                
                st.session_state.initialization_status = "success"
                
            except Exception as e:
                st.session_state.initialization_status = "error"
                with status_container:
                    st.error(f"❌ Initialization failed: {str(e)}")
                print(f"Detailed error: {str(e)}")  # For debugging


# =============================
# Main Area - Show Status or Chat
# =============================

# Show vectorstore stats if available
if st.session_state.vectorstore_stats:
    stats = st.session_state.vectorstore_stats
    
    with st.expander("📊 Current Vector Store Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Documents Folder", stats["doc_path"])
        with col2:
            st.metric("📄 PDF Documents", stats["num_docs"])
        with col3:
            st.metric("🔹 Text Chunks", stats["num_chunks"])
        
        st.info(f"📍 Full path: {stats['full_path']}")
        st.info(f"🤖 Using {stats['emb_provider']} embeddings: {stats['emb_model']}")

# Show agent status
if st.session_state.agent is None:
    st.info("👋 **Welcome!** Please configure and initialize the agent using the sidebar to start asking questions about AUTOSAR.")
    
    # Show helpful tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - **LLM Provider**: Choose your preferred language model provider
        - **Embedding Provider**: Select the model for document embeddings
        - **AUTOSAR Release**: Pick the AUTOSAR version you want to query
        - If auto-detection fails, check the debug info or manually enter the folder name
        - Click **🚀 Initialize Agent** to start loading documents
        
        The initialization process may take a few minutes depending on the number of PDFs.
        """)
else:
    # Chat Interface
    st.subheader("💬 Ask Questions About AUTOSAR")
    
    query = st.chat_input(
        "Ask about AUTOSAR feature..."
    )

    if query:

        # Cache check
        if query in st.session_state.history:
            st.chat_message("assistant").write(
                st.session_state.history[query]
            )

        else:
            with st.chat_message("assistant"):
                try:
                    # Show a thinking indicator with status
                    with st.status("🤔 Thinking...", expanded=False) as status:
                        status.write("🔍 Searching relevant documents...")
                        response = st.session_state.agent.invoke(
                            {"messages": [("user", query)]}
                        )
                        status.write("📝 Generating response...")
                        time.sleep(0.5)
                        status.update(label="✅ Response ready!", state="complete")
                    
                    answer = response["messages"][-1].content
                    
                    st.write(answer)
                    st.session_state.history[query] = answer
                    
                    # Show sources if available
                    with st.expander("📚 View Sources"):
                        st.write("This would show the source documents used for the answer")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")