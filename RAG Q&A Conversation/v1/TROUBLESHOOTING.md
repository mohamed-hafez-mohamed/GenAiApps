# 🔧 Troubleshooting Guide

## ✅ **FIXED: Import Error Solution**

The original code had an issue where it imported **all providers** even if you weren't using them. This caused errors like:

```
ModuleNotFoundError: No module named 'langchain_anthropic'
```

### **What Was Fixed:**

The new `abstraction_layer.py` uses **lazy imports** - it only imports providers when you actually use them!

```python
# OLD (caused errors):
from langchain_anthropic import ChatAnthropic  # Imported even if not used!

# NEW (only imports when needed):
def lazy_import_llm(provider):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # Only imported if used
        return ChatAnthropic
```

---

## 🚀 Quick Fix Instructions

### **Step 1: Use the Fixed File**

The error is already fixed! Just use the updated `abstraction_layer.py` file I provided.

### **Step 2: Install Minimal Dependencies**

```bash
pip install -r requirements_minimal.txt
```

This installs only what you need for the **default setup**:
- ✅ Groq (LLM)
- ✅ HuggingFace (Embeddings)
- ✅ Chroma (Vector Store)

### **Step 3: Run the App**

```bash
streamlit run configurable_rag_app.py
```

**That's it!** No more import errors!

---

## 📋 Common Import Errors & Solutions

### **Error 1: ModuleNotFoundError: No module named 'langchain_anthropic'**

**Cause:** Old abstraction layer imported all providers

**Solution:** Use the fixed `abstraction_layer.py` (already provided)

---

### **Error 2: ModuleNotFoundError: No module named 'langchain_groq'**

**Cause:** Groq package not installed

**Solution:**
```bash
pip install langchain-groq
```

---

### **Error 3: ModuleNotFoundError: No module named 'langchain_huggingface'**

**Cause:** HuggingFace package not installed

**Solution:**
```bash
pip install langchain-huggingface sentence-transformers
```

---

### **Error 4: ModuleNotFoundError: No module named 'langchain_chroma'**

**Cause:** Chroma package not installed

**Solution:**
```bash
pip install langchain-chroma chromadb
```

---

### **Error 5: ModuleNotFoundError: No module named 'langchain_openai'**

**Cause:** You set `provider = openai` in config but didn't install it

**Solution:**
```bash
pip install langchain-openai
```

---

### **Error 6: Import error with any other provider**

**Pattern:** `ModuleNotFoundError: No module named 'langchain_XXX'`

**Solution:**

1. Check which provider you're using in `config.ini`:
```ini
[llm]
provider = XXX  # <-- This one

[embeddings]
provider = YYY  # <-- And this one

[vector_store]
provider = ZZZ  # <-- And this one
```

2. Install the corresponding package:

| Provider | Install Command |
|----------|----------------|
| **groq** | `pip install langchain-groq` |
| **openai** | `pip install langchain-openai` |
| **anthropic** | `pip install langchain-anthropic` |
| **cohere** | `pip install langchain-cohere` |
| **faiss** | `pip install faiss-cpu` |
| **pinecone** | `pip install pinecone-client langchain-pinecone` |
| **qdrant** | `pip install qdrant-client langchain-qdrant` |
| **weaviate** | `pip install weaviate-client langchain-weaviate` |

---

## 🎯 Recommended Installation Paths

### **Path 1: Minimal (Default Config)**

For Groq + HuggingFace + Chroma (FREE):

```bash
pip install -r requirements_minimal.txt
```

**Includes:**
- ✅ streamlit
- ✅ langchain + langchain-core + langchain-community
- ✅ langchain-groq
- ✅ langchain-huggingface + sentence-transformers
- ✅ langchain-chroma + chromadb
- ✅ pypdf

**Total:** ~8 packages, all essentials

---

### **Path 2: Add OpenAI**

If you want to use OpenAI (GPT-4):

```bash
# Install minimal first
pip install -r requirements_minimal.txt

# Add OpenAI
pip install langchain-openai
```

Then edit `config.ini`:
```ini
[llm]
provider = openai  # Changed from groq

[llm.openai]
model = gpt-4-turbo-preview
```

---

### **Path 3: Add Anthropic (Claude)**

If you want to use Claude:

```bash
# Install minimal first
pip install -r requirements_minimal.txt

# Add Anthropic
pip install langchain-anthropic
```

Then edit `config.ini`:
```ini
[llm]
provider = anthropic  # Changed from groq

[llm.anthropic]
model = claude-3-5-sonnet-20241022
```

---

### **Path 4: Install Everything**

If you want all providers available:

```bash
pip install -r requirements_configurable.txt
```

**Warning:** This installs many packages. Only do this if you plan to switch between multiple providers.

---

## 🔍 How to Verify Installation

### **Check What's Installed:**

```bash
pip list | grep langchain
```

You should see:
```
langchain              0.x.x
langchain-chroma       0.x.x
langchain-community    0.x.x
langchain-core         0.x.x
langchain-groq         0.x.x
langchain-huggingface  0.x.x
```

### **Test Import:**

```python
# Test in Python
from abstraction_layer import RAGSystemBuilder

builder = RAGSystemBuilder("config.ini")
print("Available LLM providers:", builder.get_available_llm_providers())
```

If this runs without errors, you're good! ✅

---

## 🐛 Still Having Issues?

### **Problem: Import errors persist**

**Try:**
```bash
# Uninstall all langchain packages
pip uninstall langchain langchain-core langchain-community langchain-groq langchain-huggingface langchain-chroma -y

# Fresh install
pip install -r requirements_minimal.txt
```

### **Problem: Version conflicts**

**Try:**
```bash
# Create virtual environment
python -m venv rag_env

# Activate it
# Windows:
rag_env\Scripts\activate
# Mac/Linux:
source rag_env/bin/activate

# Install fresh
pip install -r requirements_minimal.txt
```

### **Problem: Streamlit won't start**

**Check:**
1. Is streamlit installed? `pip install streamlit`
2. Run from correct directory: `cd` to folder with `configurable_rag_app.py`
3. Use full command: `streamlit run configurable_rag_app.py`

---

## 📝 Understanding the Fix

### **Why Lazy Imports?**

**Before:**
```python
# Top of file - imports everything
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# etc...

# Problem: All must be installed even if you only use Groq!
```

**After:**
```python
# Top of file - imports nothing
# ...

# Only import when needed
def lazy_import_llm(provider):
    if provider == "groq":
        from langchain_groq import ChatGroq  # Only imported if using Groq
        return ChatGroq
    elif provider == "openai":
        from langchain_openai import ChatOpenAI  # Only if using OpenAI
        return ChatOpenAI
    # etc...
```

**Benefits:**
- ✅ Only install what you use
- ✅ No import errors for unused providers
- ✅ Faster startup
- ✅ Smaller dependency footprint

---

## ✅ Installation Checklist

Before running the app:

- [ ] Python 3.8+ installed
- [ ] Using the **fixed** `abstraction_layer.py`
- [ ] Installed minimal requirements: `pip install -r requirements_minimal.txt`
- [ ] Have Groq API key (get free from https://console.groq.com/keys)
- [ ] In correct directory (where `configurable_rag_app.py` is)
- [ ] Config file `config.ini` present in same directory

If all checked, run:
```bash
streamlit run configurable_rag_app.py
```

---

## 🎉 Success Indicators

You'll know it's working when:

1. **App starts without errors**
2. **Sidebar shows:** "Current Providers" with Groq, HuggingFace, Chroma
3. **Status shows:** "🤖 LLM Pending" (waiting for API key)
4. **You can enter API key** without errors
5. **After entering key:** Status changes to "🤖 LLM Ready"

---

## 📞 Quick Reference Commands

```bash
# Install minimal dependencies
pip install -r requirements_minimal.txt

# Run the app
streamlit run configurable_rag_app.py

# Check what's installed
pip list | grep langchain

# Verify abstraction layer
python -c "from abstraction_layer import RAGSystemBuilder; print('OK')"

# Test config reading
python -c "from abstraction_layer import ConfigManager; c=ConfigManager('config.ini'); print(c.get('llm', 'provider'))"
```

---

## 🔗 Helpful Resources

- **Groq API Keys:** https://console.groq.com/keys
- **Streamlit Docs:** https://docs.streamlit.io
- **LangChain Docs:** https://python.langchain.com/docs
- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html

---

## 💡 Pro Tips

1. **Use Virtual Environment:** Always use a venv to avoid conflicts
2. **Install Minimal First:** Start with `requirements_minimal.txt`
3. **Add Providers Later:** Only install additional providers when needed
4. **Check Config:** Make sure `config.ini` matches your installed providers
5. **Read Error Messages:** They usually tell you exactly which package is missing

---

**You should now have a working system!** 🚀

If you still have issues, check that you're using the **fixed** `abstraction_layer.py` file I just provided.
