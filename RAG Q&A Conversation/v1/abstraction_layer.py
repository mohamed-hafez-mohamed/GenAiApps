"""
Abstraction Layer for RAG System
Provides unified interface for different LLM providers, embeddings, and vector stores
"""

import os
import logging
from typing import Optional, Dict, Any, List
from configparser import ConfigParser
from abc import ABC, abstractmethod

# Core LangChain imports (always needed)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGER
# ════════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """
    Centralized configuration management.
    Loads and provides access to all configuration settings.
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigParser()
        self.config.read(config_path)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger("RAGSystem")
        
        level = self.config.get("logging", "level", fallback="INFO")
        logger.setLevel(getattr(logging, level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        log_format = self.config.get(
            "logging", 
            "format", 
            fallback="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if self.config.getboolean("logging", "enable_file_logging", fallback=False):
            log_file = self.config.get("logging", "log_file", fallback="./logs/rag_system.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)
        
        return logger
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self.config.get(section, key)
        except:
            return fallback
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer configuration value."""
        try:
            return self.config.getint(section, key)
        except:
            return fallback
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float configuration value."""
        try:
            return self.config.getfloat(section, key)
        except:
            return fallback
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value."""
        try:
            return self.config.getboolean(section, key)
        except:
            return fallback

# ════════════════════════════════════════════════════════════════════════════════
# LAZY IMPORT HELPER
# ════════════════════════════════════════════════════════════════════════════════

def lazy_import_llm(provider: str):
    """
    Import LLM provider only when needed.
    This prevents import errors for providers you're not using.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def lazy_import_embeddings(provider: str):
    """
    Import embeddings provider only when needed.
    """
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings
    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings
    elif provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")

def lazy_import_vector_store(provider: str):
    """
    Import vector store provider only when needed.
    """
    if provider == "chroma":
        from langchain_chroma import Chroma
        return Chroma
    elif provider == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS
    elif provider == "pinecone":
        try:
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore
        except ImportError:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client langchain-pinecone")
    elif provider == "qdrant":
        try:
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore
        except ImportError:
            raise ImportError("Qdrant not installed. Run: pip install qdrant-client langchain-qdrant")
    elif provider == "weaviate":
        try:
            from langchain_weaviate import WeaviateVectorStore
            return WeaviateVectorStore
        except ImportError:
            raise ImportError("Weaviate not installed. Run: pip install weaviate-client langchain-weaviate")
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

# ════════════════════════════════════════════════════════════════════════════════
# LLM ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_llm(self, api_key: str, **kwargs):
        """Create and return LLM instance."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_llm(self, api_key: str, **kwargs):
        """Create Groq LLM instance."""
        ChatGroq = lazy_import_llm("groq")
        return ChatGroq(
            api_key=api_key,
            model=self.config.get("llm.groq", "model", "llama-3.1-8b-instant"),
            temperature=self.config.getfloat("llm.groq", "temperature", 0.2),
            max_tokens=self.config.getint("llm.groq", "max_tokens", 1024),
            streaming=self.config.getboolean("llm.groq", "streaming", True),
            **kwargs
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_llm(self, api_key: str, **kwargs):
        """Create OpenAI LLM instance."""
        ChatOpenAI = lazy_import_llm("openai")
        return ChatOpenAI(
            api_key=api_key,
            model=self.config.get("llm.openai", "model", "gpt-4-turbo-preview"),
            temperature=self.config.getfloat("llm.openai", "temperature", 0.2),
            max_tokens=self.config.getint("llm.openai", "max_tokens", 1024),
            streaming=self.config.getboolean("llm.openai", "streaming", True),
            **kwargs
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_llm(self, api_key: str, **kwargs):
        """Create Anthropic LLM instance."""
        ChatAnthropic = lazy_import_llm("anthropic")
        return ChatAnthropic(
            api_key=api_key,
            model=self.config.get("llm.anthropic", "model", "claude-3-5-sonnet-20241022"),
            temperature=self.config.getfloat("llm.anthropic", "temperature", 0.2),
            max_tokens=self.config.getint("llm.anthropic", "max_tokens", 1024),
            streaming=self.config.getboolean("llm.anthropic", "streaming", True),
            **kwargs
        )


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider (local)."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_llm(self, api_key: str = None, **kwargs):
        """Create Ollama LLM instance."""
        Ollama = lazy_import_llm("ollama")
        return Ollama(
            model=self.config.get("llm.ollama", "model", "llama2"),
            base_url=self.config.get("llm.ollama", "base_url", "http://localhost:11434"),
            temperature=self.config.getfloat("llm.ollama", "temperature", 0.2),
            **kwargs
        )


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace LLM provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_llm(self, api_key: str, **kwargs):
        """Create HuggingFace LLM instance."""
        HuggingFaceEndpoint = lazy_import_llm("huggingface")
        return HuggingFaceEndpoint(
            repo_id=self.config.get("llm.huggingface", "model", "meta-llama/Llama-2-7b-chat-hf"),
            huggingfacehub_api_token=api_key,
            temperature=self.config.getfloat("llm.huggingface", "temperature", 0.2),
            max_new_tokens=self.config.getint("llm.huggingface", "max_tokens", 1024),
            **kwargs
        )


class LLMFactory:
    """
    Factory class for creating LLM instances.
    Automatically selects provider based on configuration.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize LLM factory.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.providers = {
            "groq": GroqProvider(config),
            "openai": OpenAIProvider(config),
            "anthropic": AnthropicProvider(config),
            "ollama": OllamaProvider(config),
            "huggingface": HuggingFaceProvider(config),
        }
    
    def create(self, api_key: Optional[str] = None, provider: Optional[str] = None, **kwargs):
        """
        Create LLM instance based on configuration.
        
        Args:
            api_key: API key for the provider (if required)
            provider: Override configured provider
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM instance
        """
        provider_name = provider or self.config.get("llm", "provider", "groq")
        
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        
        self.config.logger.info(f"Creating LLM with provider: {provider_name}")
        return self.providers[provider_name].create_llm(api_key, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

# ════════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def create_embeddings(self, api_key: str, **kwargs):
        """Create and return embeddings instance."""
        pass


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embeddings provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_embeddings(self, api_key: str = None, **kwargs):
        """Create HuggingFace embeddings instance."""
        HuggingFaceEmbeddings = lazy_import_embeddings("huggingface")
        return HuggingFaceEmbeddings(
            model_name=self.config.get("embeddings.huggingface", "model", "all-MiniLM-L6-v2"),
            model_kwargs={
                'device': self.config.get("embeddings.huggingface", "device", "cpu")
            },
            encode_kwargs={
                'normalize_embeddings': self.config.getboolean(
                    "embeddings.huggingface", "normalize_embeddings", True
                ),
                'batch_size': self.config.getint("embeddings.huggingface", "batch_size", 50)
            },
            **kwargs
        )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_embeddings(self, api_key: str, **kwargs):
        """Create OpenAI embeddings instance."""
        OpenAIEmbeddings = lazy_import_embeddings("openai")
        return OpenAIEmbeddings(
            api_key=api_key,
            model=self.config.get("embeddings.openai", "model", "text-embedding-3-small"),
            dimensions=self.config.getint("embeddings.openai", "dimensions", 1536),
            **kwargs
        )


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embeddings provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_embeddings(self, api_key: str, **kwargs):
        """Create Cohere embeddings instance."""
        CohereEmbeddings = lazy_import_embeddings("cohere")
        return CohereEmbeddings(
            cohere_api_key=api_key,
            model=self.config.get("embeddings.cohere", "model", "embed-english-v3.0"),
            **kwargs
        )


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embeddings provider (local)."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_embeddings(self, api_key: str = None, **kwargs):
        """Create Ollama embeddings instance."""
        OllamaEmbeddings = lazy_import_embeddings("ollama")
        return OllamaEmbeddings(
            model=self.config.get("embeddings.ollama", "model", "nomic-embed-text"),
            base_url=self.config.get("embeddings.ollama", "base_url", "http://localhost:11434"),
            **kwargs
        )


class EmbeddingFactory:
    """
    Factory class for creating embedding instances.
    Automatically selects provider based on configuration.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize embedding factory.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.providers = {
            "huggingface": HuggingFaceEmbeddingProvider(config),
            "openai": OpenAIEmbeddingProvider(config),
            "cohere": CohereEmbeddingProvider(config),
            "ollama": OllamaEmbeddingProvider(config),
        }
    
    def create(self, api_key: Optional[str] = None, provider: Optional[str] = None, **kwargs):
        """
        Create embeddings instance based on configuration.
        
        Args:
            api_key: API key for the provider (if required)
            provider: Override configured provider
            **kwargs: Additional arguments for embeddings
            
        Returns:
            Embeddings instance
        """
        provider_name = provider or self.config.get("embeddings", "provider", "huggingface")
        
        if provider_name not in self.providers:
            raise ValueError(f"Unknown embeddings provider: {provider_name}")
        
        self.config.logger.info(f"Creating embeddings with provider: {provider_name}")
        return self.providers[provider_name].create_embeddings(api_key, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

# ════════════════════════════════════════════════════════════════════════════════
# VECTOR STORE ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""
    
    @abstractmethod
    def create_from_documents(self, documents: List[Document], embeddings, **kwargs):
        """Create vector store from documents."""
        pass
    
    @abstractmethod
    def load_existing(self, embeddings, **kwargs):
        """Load existing vector store."""
        pass


class ChromaProvider(BaseVectorStoreProvider):
    """Chroma vector store provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_from_documents(self, documents: List[Document], embeddings, **kwargs):
        """Create Chroma vector store from documents."""
        Chroma = lazy_import_vector_store("chroma")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.config.get(
                "vector_store.chroma", "persist_directory", "./chroma_db"
            ),
            collection_name=self.config.get(
                "vector_store.chroma", "collection_name", "rag_documents"
            ),
            **kwargs
        )
    
    def load_existing(self, embeddings, **kwargs):
        """Load existing Chroma vector store."""
        Chroma = lazy_import_vector_store("chroma")
        return Chroma(
            embedding_function=embeddings,
            persist_directory=self.config.get(
                "vector_store.chroma", "persist_directory", "./chroma_db"
            ),
            collection_name=self.config.get(
                "vector_store.chroma", "collection_name", "rag_documents"
            ),
            **kwargs
        )


class FAISSProvider(BaseVectorStoreProvider):
    """FAISS vector store provider."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
    
    def create_from_documents(self, documents: List[Document], embeddings, **kwargs):
        """Create FAISS vector store from documents."""
        FAISS = lazy_import_vector_store("faiss")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
            **kwargs
        )
        
        # Save index
        save_dir = self.config.get("vector_store.faiss", "save_directory", "./faiss_index")
        os.makedirs(save_dir, exist_ok=True)
        vector_store.save_local(save_dir)
        
        return vector_store
    
    def load_existing(self, embeddings, **kwargs):
        """Load existing FAISS vector store."""
        FAISS = lazy_import_vector_store("faiss")
        save_dir = self.config.get("vector_store.faiss", "save_directory", "./faiss_index")
        return FAISS.load_local(save_dir, embeddings, **kwargs)


class VectorStoreFactory:
    """
    Factory class for creating vector store instances.
    Automatically selects provider based on configuration.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize vector store factory.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.providers = {
            "chroma": ChromaProvider(config),
            "faiss": FAISSProvider(config),
        }
    
    def create_from_documents(
        self, 
        documents: List[Document], 
        embeddings, 
        provider: Optional[str] = None,
        **kwargs
    ):
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to store
            embeddings: Embeddings instance
            provider: Override configured provider
            **kwargs: Additional arguments for vector store
            
        Returns:
            Vector store instance
        """
        provider_name = provider or self.config.get("vector_store", "provider", "chroma")
        
        if provider_name not in self.providers:
            raise ValueError(f"Unknown vector store provider: {provider_name}")
        
        self.config.logger.info(f"Creating vector store with provider: {provider_name}")
        return self.providers[provider_name].create_from_documents(documents, embeddings, **kwargs)
    
    def load_existing(self, embeddings, provider: Optional[str] = None, **kwargs):
        """
        Load existing vector store.
        
        Args:
            embeddings: Embeddings instance
            provider: Override configured provider
            **kwargs: Additional arguments
            
        Returns:
            Vector store instance
        """
        provider_name = provider or self.config.get("vector_store", "provider", "chroma")
        
        if provider_name not in self.providers:
            raise ValueError(f"Unknown vector store provider: {provider_name}")
        
        self.config.logger.info(f"Loading vector store with provider: {provider_name}")
        return self.providers[provider_name].load_existing(embeddings, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

# ════════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSING ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """
    Handles document processing with configurable settings.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize document processor.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter based on configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.getint("document_processing", "chunk_size", 600),
            chunk_overlap=self.config.getint("document_processing", "chunk_overlap", 100),
            length_function=len,
            separators=eval(self.config.get(
                "document_processing", 
                "separators", 
                '["\\n\\n", "\\n", ". ", " ", ""]'
            ))
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        self.config.logger.info(f"Splitting {len(documents)} documents")
        chunks = self.text_splitter.split_documents(documents)
        self.config.logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get retriever configuration."""
        search_type = self.config.get("document_processing", "search_type", "similarity")
        k = self.config.getint("document_processing", "retrieval_k", 5)
        
        config = {
            "search_type": search_type,
            "search_kwargs": {"k": k}
        }
        
        # Add additional config based on search type
        if search_type == "mmr":
            config["search_kwargs"]["fetch_k"] = self.config.getint(
                "document_processing", "fetch_k", 20
            )
            config["search_kwargs"]["lambda_mult"] = self.config.getfloat(
                "document_processing", "lambda_mult", 0.5
            )
        elif search_type == "similarity_score_threshold":
            config["search_kwargs"]["score_threshold"] = self.config.getfloat(
                "document_processing", "score_threshold", 0.7
            )
        
        return config

# ════════════════════════════════════════════════════════════════════════════════
# UNIFIED RAG SYSTEM BUILDER
# ════════════════════════════════════════════════════════════════════════════════

class RAGSystemBuilder:
    """
    Unified builder for RAG system components.
    Provides high-level interface for creating all components with proper configuration.
    """
    
    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize RAG system builder.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.llm_factory = LLMFactory(self.config)
        self.embedding_factory = EmbeddingFactory(self.config)
        self.vector_store_factory = VectorStoreFactory(self.config)
        self.document_processor = DocumentProcessor(self.config)
    
    def create_llm(self, api_key: Optional[str] = None, **kwargs):
        """Create LLM instance."""
        return self.llm_factory.create(api_key, **kwargs)
    
    def create_embeddings(self, api_key: Optional[str] = None, **kwargs):
        """Create embeddings instance."""
        return self.embedding_factory.create(api_key, **kwargs)
    
    def create_vector_store(self, documents: List[Document], embeddings, **kwargs):
        """Create vector store from documents."""
        return self.vector_store_factory.create_from_documents(documents, embeddings, **kwargs)
    
    def load_vector_store(self, embeddings, **kwargs):
        """Load existing vector store."""
        return self.vector_store_factory.load_existing(embeddings, **kwargs)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents (split into chunks)."""
        return self.document_processor.split_documents(documents)
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get retriever configuration."""
        return self.document_processor.get_retriever_config()
    
    def get_config(self) -> ConfigManager:
        """Get configuration manager."""
        return self.config
    
    def get_available_llm_providers(self) -> List[str]:
        """Get available LLM providers."""
        return self.llm_factory.get_available_providers()
    
    def get_available_embedding_providers(self) -> List[str]:
        """Get available embedding providers."""
        return self.embedding_factory.get_available_providers()
    
    def get_available_vector_store_providers(self) -> List[str]:
        """Get available vector store providers."""
        return self.vector_store_factory.get_available_providers()

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Create RAG system with abstraction layer
    
    # Initialize builder
    builder = RAGSystemBuilder("config.ini")
    
    # Print available providers
    print("Available LLM providers:", builder.get_available_llm_providers())
    print("Available embedding providers:", builder.get_available_embedding_providers())
    print("Available vector store providers:", builder.get_available_vector_store_providers())
    
    print("\nAbstraction layer ready!")
    print("Edit config.ini to switch between providers")
