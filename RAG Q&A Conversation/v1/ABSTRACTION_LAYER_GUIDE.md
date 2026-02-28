# Abstraction Layer Documentation

## 🎯 Overview

The abstraction layer provides a **unified interface** for switching between different:
- **LLM providers** (Groq, OpenAI, Anthropic, Ollama, HuggingFace)
- **Embedding models** (HuggingFace, OpenAI, Cohere, Ollama)
- **Vector stores** (Chroma, FAISS, Pinecone, Qdrant, Weaviate)

All controlled through a **single configuration file** (`config.ini`) - no code changes needed!

---

## 🚀 Quick Start

### 1. Basic Usage

```python
from abstraction_layer import RAGSystemBuilder

# Initialize with config file
builder = RAGSystemBuilder("config.ini")

# Create components (uses providers from config.ini)
llm = builder.create_llm(api_key="your-api-key")
embeddings = builder.create_embeddings(api_key="your-api-key")
vector_store = builder.create_vector_store(documents, embeddings)
```

### 2. Switching Providers

**Option A: Edit config.ini**
```ini
[llm]
provider = openai  # Change from groq to openai

[embeddings]
provider = cohere  # Change from huggingface to cohere
```

**Option B: Override in code**
```python
# Override configured provider
llm = builder.create_llm(api_key="key", provider="anthropic")
embeddings = builder.create_embeddings(api_key="key", provider="openai")
```

---

## 📋 Configuration File Structure

### config.ini Sections

```ini
[llm]                      # LLM provider selection
[llm.groq]                 # Groq-specific settings
[llm.openai]               # OpenAI-specific settings
[llm.anthropic]            # Anthropic-specific settings
[llm.ollama]               # Ollama-specific settings
[llm.huggingface]          # HuggingFace-specific settings

[embeddings]               # Embedding provider selection
[embeddings.huggingface]   # HuggingFace embedding settings
[embeddings.openai]        # OpenAI embedding settings
[embeddings.cohere]        # Cohere embedding settings
[embeddings.ollama]        # Ollama embedding settings

[vector_store]             # Vector store selection
[vector_store.chroma]      # Chroma settings
[vector_store.faiss]       # FAISS settings
[vector_store.pinecone]    # Pinecone settings

[document_processing]      # Document processing settings
[agent]                    # Agent configuration
[ui]                       # UI settings
[advanced]                 # Advanced features
[logging]                  # Logging configuration
```

---

## 🤖 LLM Providers

### Available Providers

| Provider | Type | Requires API Key | Local/Cloud |
|----------|------|------------------|-------------|
| **Groq** | Cloud | Yes | Cloud |
| **OpenAI** | Cloud | Yes | Cloud |
| **Anthropic** | Cloud | Yes | Cloud |
| **Ollama** | Local | No | Local |
| **HuggingFace** | Cloud | Yes | Cloud |

### Configuration Examples

#### Groq (Default)
```ini
[llm]
provider = groq

[llm.groq]
model = llama-3.1-8b-instant
temperature = 0.2
max_tokens = 1024
streaming = true
```

**Available Models:**
- `llama-3.1-8b-instant` - Fastest
- `llama-3.1-70b-versatile` - Most capable
- `mixtral-8x7b-32768` - Large context
- `gemma2-9b-it` - Alternative

#### OpenAI
```ini
[llm]
provider = openai

[llm.openai]
model = gpt-4-turbo-preview
temperature = 0.2
max_tokens = 1024
streaming = true
```

**Available Models:**
- `gpt-4-turbo-preview` - Latest GPT-4
- `gpt-4` - Standard GPT-4
- `gpt-3.5-turbo` - Faster, cheaper
- `gpt-3.5-turbo-16k` - Large context

#### Anthropic (Claude)
```ini
[llm]
provider = anthropic

[llm.anthropic]
model = claude-3-5-sonnet-20241022
temperature = 0.2
max_tokens = 1024
streaming = true
```

**Available Models:**
- `claude-3-5-sonnet-20241022` - Best balance
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Fast
- `claude-3-haiku-20240307` - Fastest

#### Ollama (Local)
```ini
[llm]
provider = ollama

[llm.ollama]
model = llama2
base_url = http://localhost:11434
temperature = 0.2
streaming = true
```

**Setup:**
1. Install Ollama: https://ollama.ai
2. Pull model: `ollama pull llama2`
3. Run: `ollama serve`

**Available Models:**
- `llama2` - Meta's Llama 2
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B
- `codellama` - Code-focused

---

## 🧮 Embedding Providers

### Available Providers

| Provider | Dimensions | Cost | Quality |
|----------|-----------|------|---------|
| **HuggingFace** | 384-768 | Free | Good |
| **OpenAI** | 1536-3072 | Paid | Excellent |
| **Cohere** | Variable | Paid | Excellent |
| **Ollama** | Variable | Free | Good |

### Configuration Examples

#### HuggingFace (Default - Free!)
```ini
[embeddings]
provider = huggingface

[embeddings.huggingface]
model = all-MiniLM-L6-v2
device = cpu
batch_size = 50
normalize_embeddings = true
```

**Available Models:**
- `all-MiniLM-L6-v2` - Fast, 384 dim
- `all-mpnet-base-v2` - Quality, 768 dim
- `multi-qa-MiniLM-L6-cos-v1` - Q&A optimized
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual

#### OpenAI
```ini
[embeddings]
provider = openai

[embeddings.openai]
model = text-embedding-3-small
dimensions = 1536
```

**Available Models:**
- `text-embedding-3-small` - 1536 dim, cheaper
- `text-embedding-3-large` - 3072 dim, better quality
- `text-embedding-ada-002` - Legacy, 1536 dim

#### Cohere
```ini
[embeddings]
provider = cohere

[embeddings.cohere]
model = embed-english-v3.0
input_type = search_document
```

---

## 💾 Vector Store Providers

### Available Providers

| Provider | Type | Persistence | Scalability |
|----------|------|-------------|-------------|
| **Chroma** | Local | File | Medium |
| **FAISS** | Local | File | High |
| **Pinecone** | Cloud | Database | Very High |
| **Qdrant** | Cloud/Local | Database | Very High |
| **Weaviate** | Cloud/Local | Database | Very High |

### Configuration Examples

#### Chroma (Default - Easy!)
```ini
[vector_store]
provider = chroma

[vector_store.chroma]
persist_directory = ./chroma_db
collection_name = rag_documents
```

**Pros:**
- ✅ Easy setup
- ✅ No dependencies
- ✅ Good performance
- ✅ Local storage

#### FAISS (Fast!)
```ini
[vector_store]
provider = faiss

[vector_store.faiss]
index_type = FlatL2
save_directory = ./faiss_index
```

**Index Types:**
- `FlatL2` - Exact search, slower
- `FlatIP` - Inner product
- `IVFFlat` - Fast approximate
- `HNSW` - Very fast, memory-intensive

**Pros:**
- ✅ Very fast
- ✅ Scalable
- ✅ Multiple index types

#### Pinecone (Cloud)
```ini
[vector_store]
provider = pinecone

[vector_store.pinecone]
environment = us-west1-gcp
index_name = rag-index
dimension = 1536
metric = cosine
```

**Setup:**
1. Sign up: https://pinecone.io
2. Get API key
3. Set environment variable: `PINECONE_API_KEY`

**Pros:**
- ✅ Serverless
- ✅ Highly scalable
- ✅ Managed service

---

## 🔧 Complete Configuration Examples

### Example 1: All Free (Local)

```ini
[llm]
provider = ollama

[llm.ollama]
model = llama2
base_url = http://localhost:11434

[embeddings]
provider = ollama

[embeddings.ollama]
model = nomic-embed-text
base_url = http://localhost:11434

[vector_store]
provider = faiss

[vector_store.faiss]
save_directory = ./faiss_index
```

**Cost:** $0
**Setup:** Install Ollama
**Best for:** Development, testing, privacy

### Example 2: Best Quality (Cloud)

```ini
[llm]
provider = anthropic

[llm.anthropic]
model = claude-3-5-sonnet-20241022

[embeddings]
provider = openai

[embeddings.openai]
model = text-embedding-3-large

[vector_store]
provider = pinecone

[vector_store.pinecone]
index_name = rag-index
```

**Cost:** ~$0.01-0.05 per query
**Setup:** API keys for Anthropic, OpenAI, Pinecone
**Best for:** Production, best answers

### Example 3: Balanced (Mixed)

```ini
[llm]
provider = groq

[llm.groq]
model = llama-3.1-8b-instant

[embeddings]
provider = huggingface

[embeddings.huggingface]
model = all-MiniLM-L6-v2

[vector_store]
provider = chroma

[vector_store.chroma]
persist_directory = ./chroma_db
```

**Cost:** Free for Groq tier, $0 for embeddings/vector
**Setup:** Groq API key only
**Best for:** Most use cases

---

## 📝 Code Examples

### Example 1: Basic Usage

```python
from abstraction_layer import RAGSystemBuilder
from langchain_community.document_loaders import PyPDFLoader

# Initialize
builder = RAGSystemBuilder("config.ini")

# Create components
llm = builder.create_llm(api_key="your-groq-key")
embeddings = builder.create_embeddings()  # No key for HuggingFace

# Load and process documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()
split_docs = builder.process_documents(documents)

# Create vector store
vector_store = builder.create_vector_store(split_docs, embeddings)

# Create retriever
retriever_config = builder.get_retriever_config()
retriever = vector_store.as_retriever(**retriever_config)

# Query
docs = retriever.get_relevant_documents("What is this about?")
```

### Example 2: Switching Providers

```python
# Start with Groq
llm_groq = builder.create_llm(api_key="groq-key", provider="groq")

# Switch to OpenAI
llm_openai = builder.create_llm(api_key="openai-key", provider="openai")

# Switch to Anthropic
llm_anthropic = builder.create_llm(api_key="anthropic-key", provider="anthropic")

# Use same embeddings for all
embeddings = builder.create_embeddings()
```

### Example 3: Multiple Vector Stores

```python
# Create Chroma store
chroma_store = builder.create_vector_store(
    docs, embeddings, provider="chroma"
)

# Create FAISS store
faiss_store = builder.create_vector_store(
    docs, embeddings, provider="faiss"
)

# Compare retrieval
chroma_results = chroma_store.similarity_search("query", k=5)
faiss_results = faiss_store.similarity_search("query", k=5)
```

### Example 4: Configuration Inspection

```python
# Get available providers
print("LLM providers:", builder.get_available_llm_providers())
print("Embedding providers:", builder.get_available_embedding_providers())
print("Vector stores:", builder.get_available_vector_store_providers())

# Get current config
config = builder.get_config()
print("Current LLM:", config.get("llm", "provider"))
print("Chunk size:", config.getint("document_processing", "chunk_size"))
```

---

## 🎨 UI Configuration

### Theme Selection

```ini
[ui]
theme = purple_gradient
```

**Available Themes:**
- `purple_gradient` - Purple/violet (default)
- `blue_ocean` - Blue tones
- `green_forest` - Green tones
- `orange_sunset` - Orange/red tones

### Feature Toggles

```ini
[ui]
show_agent_reasoning = true
show_tool_usage = true
enable_animations = true
enable_chat_bubbles = true
enable_status_badges = true
```

### Page Settings

```ini
[ui]
page_title = Advanced RAG Agent System
page_icon = 🤖
layout = wide
sidebar_state = expanded
```

---

## 🔍 Advanced Configuration

### Document Processing

```ini
[document_processing]
# Text splitting
chunk_size = 600
chunk_overlap = 100
separators = ["\n\n", "\n", ". ", " ", ""]

# Retrieval
search_type = similarity
retrieval_k = 5

# For MMR search
fetch_k = 20
lambda_mult = 0.5

# For threshold search
score_threshold = 0.7
```

**Search Types:**
- `similarity` - Simple cosine similarity
- `mmr` - Maximum Marginal Relevance (diverse results)
- `similarity_score_threshold` - Only results above threshold

### Agent Configuration

```ini
[agent]
max_iterations = 3
verbose = true
return_intermediate_steps = true
handle_parsing_errors = true
enabled_tools = ["search_documents", "summarize_documents", "get_document_stats"]
```

### Logging

```ini
[logging]
level = INFO
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = ./logs/rag_system.log
enable_file_logging = false
```

**Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## 🚀 Running the Configurable App

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: For specific providers
pip install pinecone-client langchain-pinecone  # Pinecone
pip install qdrant-client langchain-qdrant      # Qdrant
pip install cohere                               # Cohere embeddings
```

### 2. Configure

Edit `config.ini`:
```ini
[llm]
provider = groq  # or openai, anthropic, ollama

[embeddings]
provider = huggingface  # or openai, cohere, ollama

[vector_store]
provider = chroma  # or faiss, pinecone
```

### 3. Run

```bash
streamlit run configurable_rag_app.py
```

### 4. Switch Providers

1. Stop the app (Ctrl+C)
2. Edit `config.ini`
3. Restart the app
4. Enter new API keys if needed

**No code changes required!**

---

## 💡 Best Practices

### Provider Selection

**For Development:**
```ini
[llm]
provider = ollama  # Free, local

[embeddings]
provider = huggingface  # Free

[vector_store]
provider = chroma  # Simple, local
```

**For Production:**
```ini
[llm]
provider = groq  # Fast, good quality

[embeddings]
provider = openai  # Best quality

[vector_store]
provider = pinecone  # Scalable, managed
```

**For Privacy:**
```ini
[llm]
provider = ollama  # Everything local

[embeddings]
provider = ollama

[vector_store]
provider = faiss
```

### Performance Tuning

**Fast Responses:**
- Use Groq or OpenAI with small models
- Small chunk size (400-500)
- Low retrieval K (3-4)
- FAISS vector store

**Best Quality:**
- Use Claude 3.5 Sonnet or GPT-4
- Larger chunk size (800-1000)
- Higher retrieval K (6-8)
- OpenAI embeddings

**Cost Optimization:**
- Use Ollama (free) or Groq (generous free tier)
- HuggingFace embeddings (free)
- Chroma/FAISS (free)

---

## 🔧 Troubleshooting

### Issue: Provider not available

**Error:** `Unknown LLM provider: xyz`

**Solution:**
1. Check provider name spelling in config.ini
2. Verify provider is in available list
3. Install required packages (e.g., `pip install langchain-openai`)

### Issue: API key error

**Error:** `Invalid API key`

**Solution:**
1. Verify API key is correct
2. Check key has proper permissions
3. For Ollama, ensure server is running

### Issue: Vector store creation fails

**Error:** Cannot create vector store

**Solution:**
1. Check embeddings are created successfully
2. Verify persist directory exists and is writable
3. For cloud providers (Pinecone), check API key

### Issue: Import errors

**Error:** `Module not found`

**Solution:**
```bash
# Install all optional dependencies
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-cohere
pip install pinecone-client
pip install qdrant-client
```

---

## 📊 Comparison Tables

### LLM Providers

| Provider | Speed | Quality | Cost | Context |
|----------|-------|---------|------|---------|
| Groq | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Free tier | 8k-32k |
| OpenAI | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ | 8k-128k |
| Anthropic | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | $$ | 200k |
| Ollama | ⚡⚡⚡ | ⭐⭐⭐ | Free | Variable |

### Embedding Providers

| Provider | Speed | Quality | Cost | Dimension |
|----------|-------|---------|------|-----------|
| HuggingFace | ⚡⚡⚡⚡ | ⭐⭐⭐ | Free | 384-768 |
| OpenAI | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | $ | 1536-3072 |
| Cohere | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | $ | Variable |
| Ollama | ⚡⚡⚡ | ⭐⭐⭐ | Free | Variable |

### Vector Stores

| Provider | Setup | Speed | Scalability | Cost |
|----------|-------|-------|-------------|------|
| Chroma | Easy | ⚡⚡⚡⚡ | Medium | Free |
| FAISS | Easy | ⚡⚡⚡⚡⚡ | High | Free |
| Pinecone | Medium | ⚡⚡⚡⚡⚡ | Very High | $$ |
| Qdrant | Medium | ⚡⚡⚡⚡⚡ | Very High | $ |

---

## 🎓 Learning Resources

### Configuration Files
- INI file format: https://docs.python.org/3/library/configparser.html
- Best practices: Keep sensitive info in environment variables

### Provider Documentation
- Groq: https://console.groq.com/docs
- OpenAI: https://platform.openai.com/docs
- Anthropic: https://docs.anthropic.com
- Ollama: https://ollama.ai/docs

### LangChain
- Documentation: https://python.langchain.com/docs
- Provider guides: Check integration docs

---

## 📝 Summary

### Key Benefits

1. **No Code Changes** - Switch providers via config file
2. **Unified Interface** - Same code works with any provider
3. **Easy Testing** - Compare providers quickly
4. **Production Ready** - Proper error handling and logging
5. **Flexible** - Override config in code when needed

### Quick Reference

```python
# Initialize
builder = RAGSystemBuilder("config.ini")

# Create components (from config)
llm = builder.create_llm(api_key)
embeddings = builder.create_embeddings(api_key)
vector_store = builder.create_vector_store(docs, embeddings)

# Or override provider
llm = builder.create_llm(api_key, provider="openai")
```

### Configuration Template

```ini
[llm]
provider = groq

[embeddings]
provider = huggingface

[vector_store]
provider = chroma

[document_processing]
chunk_size = 600
retrieval_k = 5
```

**That's it! You now have a fully configurable RAG system!** 🚀
