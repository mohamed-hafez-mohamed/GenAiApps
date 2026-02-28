from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

# =========================
# Lazy LLM Loader
# =========================
def lazy_llm(provider, model, api_key=None):

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            streaming=True
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model)

    if provider == "huggingface":
        # Set HF_TOKEN environment variable
        if api_key:
            os.environ["HF_TOKEN"] = api_key
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            temperature=0.1,
            max_new_tokens=512
        )
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=api_key,
            streaming=True
        )

    raise ValueError("Unsupported LLM provider")


# =========================
# Lazy Embeddings Loader
# =========================
def lazy_embeddings(provider, model, api_key=None):

    if provider == "huggingface":
        # Set HF_TOKEN environment variable
        if api_key:
            os.environ["HF_TOKEN"] = api_key
            
        # Use the correct langchain_huggingface package
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Simple and clean initialization
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key
        )

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=model)

    raise ValueError("Unsupported embedding provider")


# =========================
# Vector Store Builder
# =========================
def build_vectorstore(doc_path, embedding):

    # Check if path exists
    if not os.path.exists(doc_path):
        raise ValueError(f"Document path '{doc_path}' does not exist")
    
    print(f"Loading documents from: {os.path.abspath(doc_path)}")  # Debug output
    
    # Load documents
    loader = PyPDFDirectoryLoader(doc_path)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No PDF documents found in {doc_path}")

    print(f"✅ Loaded {len(documents)} PDF documents")  # Debug output

    # Check if documents have content
    empty_docs = [i for i, doc in enumerate(documents) if not doc.page_content.strip()]
    if empty_docs:
        print(f"⚠️ Warning: {len(empty_docs)} documents have no text content")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    
    if not chunks:
        # Try with a different splitter configuration if no chunks created
        print("⚠️ No chunks created with default settings. Trying alternative configuration...")
        
        # Alternative splitter with smaller chunks
        alt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = alt_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No chunks created from documents even with alternative settings. Check if documents contain extractable text.")

    print(f"✅ Created {len(chunks)} document chunks")  # Debug output
    
    # Validate chunks have content
    chunks_with_content = [chunk for chunk in chunks if chunk.page_content.strip()]
    if len(chunks_with_content) < len(chunks):
        print(f"⚠️ Warning: {len(chunks) - len(chunks_with_content)} chunks have no text content")

    # Create vectorstore with explicit error handling
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=None  # In-memory for now
        )
        print("✅ Vectorstore created successfully")  # Debug output
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vectorstore: {str(e)}")
        # Try with a sample to debug
        print("Testing embedding with sample text...")
        sample_text = "This is a test sentence for embedding."
        try:
            sample_embedding = embedding.embed_query(sample_text)
            print(f"✅ Sample embedding generated successfully. Length: {len(sample_embedding)}")
            print(f"First few values: {sample_embedding[:5]}")
        except Exception as embed_error:
            print(f"❌ Embedding test failed: {str(embed_error)}")
        raise e