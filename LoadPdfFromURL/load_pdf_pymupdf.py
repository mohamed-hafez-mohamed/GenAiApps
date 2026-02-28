"""
Load PDF from URL using LangChain - PyMuPDFLoader
This is the most reliable LangChain loader with minimal dependencies
"""

from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf_with_pymupdf(pdf_url):
    """
    Load PDF using PyMuPDFLoader
    This loader has minimal dependencies and is very reliable
    """
    print(f"Loading PDF from: {pdf_url}")
    loader = PyMuPDFLoader(pdf_url)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    pdf_url = "https://www.autosar.org/fileadmin/standards/R18-10_R4.4.0_R1.5.0/CP/AUTOSAR_SWS_CANNetworkManagement.pdf"
    
    print("="*80)
    print("Loading PDF with PyMuPDFLoader")
    print("="*80)
    
    try:
        documents = load_pdf_with_pymupdf(pdf_url)
        
        print(f"\nSuccessfully loaded {len(documents)} pages")
        print(f"\nFirst page preview:")
        print("-" * 80)
        print(documents[0].page_content[:500])
        print("-" * 80)
        
        print(f"\nMetadata:")
        print(documents[0].metadata)
        
        # Show statistics
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"\nTotal characters: {total_chars:,}")
        
        # Search example
        search_term = "CAN"
        pages_with_term = [doc for doc in documents if search_term in doc.page_content]
        print(f"\nPages containing '{search_term}': {len(pages_with_term)}")
        
        # You can also use load_and_split() for automatic text splitting
        print("\n" + "="*80)
        print("With automatic splitting:")
        print("="*80)
        loader2 = PyMuPDFLoader(pdf_url)
        chunks = loader2.load_and_split()
        print(f"Total chunks: {len(chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"\nMake sure you have installed: pip install pymupdf")
