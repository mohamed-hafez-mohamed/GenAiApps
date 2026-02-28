# 🚀 QUICK START - Import Errors Fixed!

## ✅ All Import Errors Resolved!

Choose one of these approaches:

---

## 🎯 **RECOMMENDED: Simple Version**

Use the **simple version** which works with all LangChain versions:

### **Step 1: Install**
```bash
pip install -r requirements_minimal.txt
```

### **Step 2: Run**
```bash
streamlit run simple_rag_app.py
```

### **Step 3: Use**
- Enter Groq API key (get free from https://console.groq.com/keys)
- Upload PDFs
- Ask questions!

**Why Simple Version?**
- ✅ No agent import errors
- ✅ Works with any LangChain version
- ✅ Still fully configurable
- ✅ All provider switching works
- ✅ Faster and more reliable

---

## 🤖 **Alternative: Agent Version**

If you want the agent features:

### **Option A: Install LangGraph**
```bash
pip install -r requirements_minimal.txt
pip install langgraph>=0.2.0
```

Then run:
```bash
streamlit run configurable_rag_app.py
```

### **Option B: Downgrade LangChain**
```bash
pip uninstall langchain langchain-core langchain-community -y
pip install langchain==0.1.20 langchain-core==0.1.52 langchain-community==0.0.38
pip install -r requirements_minimal.txt
```

Then run:
```bash
streamlit run configurable_rag_app.py
```

---

## 📊 Version Comparison

| Feature | simple_rag_app.py | configurable_rag_app.py |
|---------|-------------------|-------------------------|
| **Provider Switching** | ✅ Yes | ✅ Yes |
| **Q&A** | ✅ Yes | ✅ Yes |
| **Chat History** | ✅ Yes | ✅ Yes |
| **Streaming** | ✅ Yes | ✅ Yes |
| **Agent Reasoning** | ❌ No | ✅ Yes |
| **Multi-tool** | ❌ No | ✅ Yes |
| **Import Errors** | ✅ None | ⚠️ Needs LangGraph |
| **Version Compatibility** | ✅ All | ⚠️ Needs setup |

---

## 🔑 Get API Key

**Groq (FREE):**
1. Go to https://console.groq.com/keys
2. Sign up (free)
3. Create API key
4. Copy key
5. Paste in app sidebar

---

## 📁 Which Files to Use?

### **Core Files (Required for All)**
- `abstraction_layer.py` - Provider abstraction
- `config.ini` - Configuration
- `requirements_minimal.txt` - Dependencies

### **Choose One App File:**

**For Maximum Compatibility:**
- `simple_rag_app.py` ✅ RECOMMENDED

**For Agent Features:**
- `configurable_rag_app.py` (needs LangGraph)
- `advanced_agent_rag.py` (needs LangGraph)

**For No Configuration:**
- `enhanced_rag_app.py` (fixed providers, no abstraction layer)

---

## 🛠️ Installation Commands

### **Minimal Setup (Works Everywhere)**
```bash
pip install streamlit langchain langchain-core langchain-community
pip install langchain-groq langchain-huggingface sentence-transformers
pip install langchain-chroma chromadb pypdf
streamlit run simple_rag_app.py
```

### **Full Setup (With Agents)**
```bash
pip install -r requirements_minimal.txt
pip install langgraph
streamlit run configurable_rag_app.py
```

---

## ❓ Common Questions

### **Q: Which version should I use?**
A: Use `simple_rag_app.py` - it's the most reliable.

### **Q: Do I lose features with simple version?**
A: You lose agent reasoning display, but still get:
- Full Q&A functionality
- Provider switching
- Chat history
- Streaming responses
- All configurations

### **Q: Can I switch providers with simple version?**
A: Yes! Just edit `config.ini` exactly the same way.

### **Q: What if I want agent features?**
A: Install LangGraph: `pip install langgraph>=0.2.0`

### **Q: Still getting errors?**
A: Use simple version - it works with all LangChain versions.

---

## 📝 Configuration Example

**config.ini works the same for all versions:**

```ini
[llm]
provider = groq  # or openai, anthropic, ollama

[embeddings]
provider = huggingface  # or openai, cohere

[vector_store]
provider = chroma  # or faiss
```

Change providers, restart app - that's it!

---

## 🎉 You're Ready!

**Simplest path:**
```bash
pip install -r requirements_minimal.txt
streamlit run simple_rag_app.py
```

Enter API key → Upload PDFs → Ask questions!

No import errors, guaranteed! ✅
