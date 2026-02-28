# abstraction_layer.py (with vector store added back for config)
"""
Abstraction Layer for RAG System
"""

import os
import logging
from typing import Optional, List, Dict, Any
from configparser import ConfigParser
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

class ConfigManager:
    def __init__(self, config_path: str = "config.ini"):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging.INFO)

    def get(self, section: str, key: str, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback=0):
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback=0.0):
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback=False):
        return self.config.getboolean(section, key, fallback=fallback)

# ────────────────────────────────────────────────
# Lazy imports
# ────────────────────────────────────────────────

def lazy_llm(provider: str):
    if provider == "groq":         from langchain_groq import ChatGroq; return ChatGroq
    if provider == "openai":       from langchain_openai import ChatOpenAI; return ChatOpenAI
    if provider == "anthropic":    from langchain_anthropic import ChatAnthropic; return ChatAnthropic
    if provider == "ollama":       from langchain_community.llms import Ollama; return Ollama
    if provider == "huggingface":  from langchain_huggingface import HuggingFaceEndpoint; return HuggingFaceEndpoint
    raise ValueError(f"Unknown LLM: {provider}")

def lazy_embedding(provider: str):
    if provider == "huggingface":  from langchain_huggingface import HuggingFaceEmbeddings; return HuggingFaceEmbeddings
    if provider == "openai":       from langchain_openai import OpenAIEmbeddings; return OpenAIEmbeddings
    if provider == "cohere":       from langchain_cohere import CohereEmbeddings; return CohereEmbeddings
    if provider == "ollama":       from langchain_community.embeddings import OllamaEmbeddings; return OllamaEmbeddings
    raise ValueError(f"Unknown embedding: {provider}")

def lazy_vector_store(provider: str):
    if provider == "chroma": from langchain_chroma import Chroma; return Chroma
    if provider == "faiss": from langchain_community.vectorstores import FAISS; return FAISS
    raise ValueError(f"Unknown vector store: {provider}")

# ────────────────────────────────────────────────
# LLM Factories (simplified)
# ────────────────────────────────────────────────

class LLMFactory:
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg

    def create(self, provider: str = None, api_key: Optional[str] = None, model: Optional[str] = None):
        provider = provider or self.cfg.get("llm", "provider", "groq")
        model = model or self.cfg.get(f"llm.{provider}", "model")
        cls = lazy_llm(provider)

        common = {
            "temperature": self.cfg.getfloat("llm", "temperature", 0.2),
            "max_tokens": self.cfg.getint("llm", "max_tokens", 1024),
        }

        if provider == "groq":
            return cls(api_key=api_key, model=model, **common)
        if provider == "openai":
            return cls(api_key=api_key, model=model, **common)
        if provider == "anthropic":
            return cls(api_key=api_key, model=model, **common)
        if provider == "ollama":
            return cls(model=model, base_url=self.cfg.get("llm.ollama", "base_url", "http://localhost:11434"))
        if provider == "huggingface":
            return cls(repo_id=model, huggingfacehub_api_token=api_key, **common)

        raise ValueError(f"Unsupported LLM provider: {provider}")

# ────────────────────────────────────────────────
# Embeddings (similar simplification)
# ────────────────────────────────────────────────

class EmbeddingFactory:
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg

    def create(self, provider: str = None, api_key: Optional[str] = None, model: Optional[str] = None):
        provider = provider or self.cfg.get("embeddings", "provider", "huggingface")
        model = model or self.cfg.get(f"embeddings.{provider}", "model")
        cls = lazy_embedding(provider)

        if provider == "huggingface":
            return cls(
                model_name=model,
                model_kwargs={"device": self.cfg.get("embeddings.huggingface", "device", "cpu")},
                encode_kwargs={"normalize_embeddings": True}
            )
        if provider == "openai":
            return cls(api_key=api_key, model=model)
        if provider == "cohere":
            return cls(cohere_api_key=api_key, model=model)
        if provider == "ollama":
            return cls(model=model, base_url=self.cfg.get("embeddings.ollama", "base_url", "http://localhost:11434"))

        raise ValueError(f"Unsupported embedding provider: {provider}")

# ────────────────────────────────────────────────
# Vector Store
# ────────────────────────────────────────────────

class VectorStoreFactory:
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg

    def create_from_docs(self, docs: List[Document], embeddings, provider: str = None):
        provider = provider or self.cfg.get("vector_store", "provider", "chroma")
        cls = lazy_vector_store(provider)

        if provider == "chroma":
            return cls.from_documents(
                docs, embeddings,
                persist_directory=self.cfg.get("vector_store", "persist_directory", "./chroma_db"),
                collection_name=self.cfg.get("vector_store", "collection_name", "rag_docs")
            )
        if provider == "faiss":
            vs = cls.from_documents(docs, embeddings)
            vs.save_local("./faiss_index")
            return vs

# ────────────────────────────────────────────────
# Document processing
# ────────────────────────────────────────────────

class DocumentProcessor:
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.getint("document_processing", "chunk_size", 600),
            chunk_overlap=cfg.getint("document_processing", "chunk_overlap", 120),
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

    def retriever_config(self) -> Dict:
        return {
            "search_type": self.cfg.get("document_processing", "search_type", "similarity"),
            "search_kwargs": {"k": self.cfg.getint("document_processing", "retrieval_k", 4)}
        }

# ────────────────────────────────────────────────
# Main Builder
# ────────────────────────────────────────────────

class RAGSystemBuilder:
    def __init__(self, config_path="config.ini"):
        self.cfg = ConfigManager(config_path)
        self.llm_factory = LLMFactory(self.cfg)
        self.embed_factory = EmbeddingFactory(self.cfg)
        self.vs_factory = VectorStoreFactory(self.cfg)
        self.doc_processor = DocumentProcessor(self.cfg)

    def create_llm(self, provider=None, api_key=None, model=None):
        return self.llm_factory.create(provider, api_key, model)

    def create_embeddings(self, provider=None, api_key=None, model=None):
        return self.embed_factory.create(provider, api_key, model)

    def create_vector_store(self, docs, embeddings, provider=None):
        return self.vs_factory.create_from_docs(docs, embeddings, provider)

    def process_documents(self, docs):
        return self.doc_processor.split(docs)

    def get_retriever_config(self):
        return self.doc_processor.retriever_config()

    def get_config(self):
        return self.cfg

    def get_llm_providers(self):
        return ["groq", "openai", "anthropic", "ollama", "huggingface"]

    def get_embedding_providers(self):
        return ["huggingface", "openai", "cohere", "ollama"]