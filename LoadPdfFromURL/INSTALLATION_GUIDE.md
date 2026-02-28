# PDF Loading Solutions - Installation Guide

## 🎯 BEST SOLUTION: PyMuPDFLoader (LangChain)

**File:** `load_pdf_pymupdf.py`

**Why this is best:**
- ✅ Uses LangChain (Document objects work with LangChain chains)
- ✅ Only needs ONE dependency: `pymupdf`
- ✅ Fast and reliable
- ✅ Works with URLs directly
- ✅ Built-in text splitting support

**Installation:**
```bash
pip install langchain-community pymupdf
```

**Run:**
```bash
python load_pdf_pymupdf.py
```

---

## Alternative 1: Simple pypdf (No LangChain)

**File:** `load_pdf_simple.py`

**Use if:** You don't need LangChain at all

**Installation:**
```bash
pip install pypdf requests
```

**Run:**
```bash
python load_pdf_simple.py
```

---

## ❌ Don't Use These:

### OnlinePDFLoader
- **Problem:** Requires `pdfminer.six` AND `unstructured_inference` (heavy!)
- **Status:** Not recommended

### PyPDFLoader
- **Problem:** Pulls in transformers, torch, sentence_transformers (2GB+)
- **Status:** Not recommended unless you need those libraries anyway

---

## Quick Start

**Just want it to work? Run these two commands:**

```bash
pip install langchain-community pymupdf
python load_pdf_pymupdf.py
```

Done! 🎉

---

## Comparison Table

| Loader | Dependencies | Size | Speed | LangChain | Recommended |
|--------|-------------|------|-------|-----------|-------------|
| **PyMuPDFLoader** | pymupdf | ~15MB | ⚡ Fast | ✅ Yes | ⭐ **YES** |
| Simple pypdf | pypdf, requests | ~5MB | ⚡ Fast | ❌ No | ✅ If no LangChain needed |
| OnlinePDFLoader | pdfminer, unstructured | ~50MB+ | 🐌 Slow | ✅ Yes | ❌ No |
| PyPDFLoader | torch, transformers | ~2GB+ | 🐌 Very Slow | ✅ Yes | ❌ No |

---

## Common Errors & Solutions

### Error: "No module named 'fitz'"
```bash
pip install pymupdf
```

### Error: "No module named 'pdfminer'"
You're trying to use OnlinePDFLoader. Switch to PyMuPDFLoader instead.

### Error: "No module named 'unstructured_inference'"
You're trying to use OnlinePDFLoader. Switch to PyMuPDFLoader instead.

### Error: Import hangs for several minutes
You're trying to use PyPDFLoader which loads heavy ML libraries. Switch to PyMuPDFLoader instead.

---

## Features Comparison

### All loaders support:
- ✅ Loading from URL
- ✅ Text extraction
- ✅ Page-by-page access
- ✅ Metadata

### Only PyMuPDFLoader adds:
- ✅ Built-in `load_and_split()` method
- ✅ Native LangChain Document objects
- ✅ Works seamlessly with LangChain chains and RAG pipelines
- ✅ Better text extraction quality

---

## Which Should You Use?

**Choose PyMuPDFLoader if:**
- You're using LangChain for RAG, chains, or agents
- You want the best text extraction quality
- You want minimal dependencies

**Choose Simple pypdf if:**
- You're NOT using LangChain
- You just need basic PDF text extraction
- You want the absolute smallest dependency footprint
