import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load all environment variables from .env file
load_dotenv()

# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY", "")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
langchain_project = os.getenv("LANGCHAIN_PROJECT", "RAG-QA-Conversation")
huggingfacehub_api_token = os.getenv("HF_TOKEN", "")

# Set environment variables only if they exist and are not empty
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# LangSmith Tracking configuration
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# ── UI ───────────────────────────────────────────────────────────────
st.title("Conversational RAG Q&A with PDF Uploads and Chat History")
st.write("Upload a PDF file and ask questions about its content")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vector_database' not in st.session_state:
    st.session_state.vector_database = None
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'docs_processed' not in st.session_state:
    st.session_state.docs_processed = False

# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    
    # API Key handling
    use_env_key = st.checkbox("Use environment variable", value=bool(groq_api_key))
    
    if use_env_key and groq_api_key:
        st.success("Using API key from environment")
        api_key_to_use = groq_api_key
    else:
        api_key_input = st.text_input("Enter your Groq API Key", type="password", value="")
        if api_key_input:
            api_key_to_use = api_key_input
            st.success("Using manually entered API key")
        else:
            api_key_to_use = None
            st.warning("Please enter your Groq API Key")

# Initialize LLM if we have an API key
if api_key_to_use and not st.session_state.llm_model:
    try:
        st.session_state.llm_model = ChatGroq(
            api_key=api_key_to_use, 
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        st.sidebar.success("✅ LLM initialized successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize LLM: {e}")
        st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["Upload PDFs", "Ask Questions", "Chat History"])

with tab1:
    st.header("Upload PDF Files")
    session_id = st.text_input("Enter a session ID for chat history", value="default_session", key="session_input")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload multiple PDF files"
    )
    
    # Process uploaded files
    if uploaded_files and len(uploaded_files) > 0 and st.session_state.llm_model:
        if st.button("Process PDFs", type="primary"):
            documents = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    # Get the file bytes
                    file_bytes = uploaded_file.getvalue()
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    st.success(f"✅ Loaded {len(docs)} pages from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if documents:
                status_text.text("Processing documents...")
                
                # Text Splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = text_splitter.split_documents(documents)
                
                st.info(f"📄 Total pages loaded: {len(documents)}")
                st.info(f"✂️ Total chunks created: {len(split_docs)}")
                
                # Create Embeddings and store in vector db
                status_text.text("Creating embeddings...")
                embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Clear existing vector database if any
                if st.session_state.vector_database:
                    try:
                        st.session_state.vector_database.delete_collection()
                    except:
                        pass
                
                vector_database = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embedding,
                    collection_name=f"pdf_collection_{session_id}"
                )
                
                # Store vector database in session state
                st.session_state.vector_database = vector_database
                
                # Create retriever
                retriever = vector_database.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                # Create system message for question contextualization
                query_rephrasing_system_message = (
                    "Given the latest user question and a chat history "
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do not answer the question, "
                    "just reformulate it if needed otherwise return it as is."
                )
                
                query_rephrasing_prompt = ChatPromptTemplate.from_messages([
                    ("system", query_rephrasing_system_message),
                    (MessagesPlaceholder(variable_name="chat_history")),
                    ("human", "{input}")
                ])
                
                # Create prompt template for answer generation
                system_prompt = (
                    "You are a helpful AI assistant that helps people find information "
                    "from the provided context. Use the context to answer the question at the end. "
                    "If you don't know the answer, just say you don't know. "
                    "Use three sentences maximum to answer the question. "
                    "Keep the answer concise and to the point."
                    "\n\n"
                    "Context: {context}"
                )
                
                answer_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    (MessagesPlaceholder(variable_name="chat_history")),
                    ("human", "{input}")
                ])
                
                # Create history aware retriever chain
                history_aware_retriever_chain = (
                    query_rephrasing_prompt 
                    | st.session_state.llm_model 
                    | StrOutputParser() 
                    | retriever
                )
                
                # Create document chain for answer generation
                document_chain = answer_generation_prompt | st.session_state.llm_model | StrOutputParser()
                
                # Create retrieval chain
                retrieval_chain = (
                    RunnablePassthrough.assign(
                        # Get context using the history-aware retriever
                        context=history_aware_retriever_chain,
                        # Pass through chat_history for the answer generation prompt
                        chat_history=lambda input_dict: input_dict.get("chat_history", []),
                        # Pass through the original input
                        input=lambda input_dict: input_dict["input"]
                    )
                    | document_chain
                )
                
                def get_session_history(session_id: str) -> BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id] = ChatMessageHistory()
                    return st.session_state.store[session_id]
                
                # Wrap the retrieval chain with history
                conversational_rag_chain = RunnableWithMessageHistory(
                    retrieval_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                )
                
                # Store chain in session state
                st.session_state.conversational_rag_chain = conversational_rag_chain
                st.session_state.docs_processed = True
                
                status_text.text("")
                progress_bar.empty()
                st.success("✅ PDFs processed and RAG system ready!")
                
                # Clear uploaded files from UI
                st.rerun()

with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.llm_model:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
    elif not st.session_state.docs_processed:
        st.info("📁 Please upload and process PDF files in the 'Upload PDFs' tab first.")
    else:
        user_question = st.text_area(
            "Enter your question about the PDF content:", 
            height=100,
            placeholder="Type your question here...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask Question", type="primary", disabled=not user_question)
        with col2:
            clear_chat = st.button("Clear Chat History")
        
        if clear_chat and session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
            st.success("Chat history cleared!")
            st.rerun()
        
        if ask_button and user_question:
            with st.spinner("Thinking..."):
                try:
                    # Get response
                    response = st.session_state.conversational_rag_chain.invoke(
                        {"input": user_question},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"Error getting response: {e}")
                    st.error("Please try re-uploading the PDF files.")

with tab3:
    st.header("Chat History")
    
    if not session_id or session_id not in st.session_state.store:
        st.info("No chat history available. Start a conversation in the 'Ask Questions' tab.")
    else:
        session_history = st.session_state.store[session_id]
        
        if not session_history.messages:
            st.info("No messages in chat history yet.")
        else:
            for i, message in enumerate(session_history.messages):
                with st.chat_message("user" if message.type == "human" else "assistant"):
                    st.write(message.content)
            
            # Show statistics
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Messages", len(session_history.messages))
            with col2:
                user_msgs = len([m for m in session_history.messages if m.type == "human"])
                st.metric("User Messages", user_msgs)

# Footer
st.divider()
with st.expander("System Status"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**LLM Status:** {'✅ Ready' if st.session_state.llm_model else '❌ Not Ready'}")
    with col2:
        st.write(f"**Vector DB:** {'✅ Ready' if st.session_state.vector_database else '❌ Not Ready'}")
    with col3:
        st.write(f"**RAG Chain:** {'✅ Ready' if st.session_state.conversational_rag_chain else '❌ Not Ready'}")
    
    st.write(f"**Current Session ID:** {session_id}")
    st.write(f"**Total Sessions:** {len(st.session_state.store)}")
    if session_id in st.session_state.store:
        st.write(f"**Messages in Session:** {len(st.session_state.store[session_id].messages)}")