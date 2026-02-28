"""
Load PDF from URL using LangChain with alternative loader
This avoids the heavy sentence_transformers dependency
"""

from langchain_community.document_loaders import OnlinePDFLoader
# Alternative: Use UnstructuredPDFLoader if you have unstructured installed

def load_pdf_with_langchain_online(pdf_url):
    """
    Load PDF using LangChain's OnlinePDFLoader
    This is a simpler loader that doesn't trigger heavy dependencies
    """
    print(f"Loading PDF from: {pdf_url}")
    loader = OnlinePDFLoader(pdf_url)
    documents = loader.load()
    return documents


def load_pdf_with_pypdf_loader_manual():
    """
    Alternative: Manual implementation that mimics PyPDFLoader
    without importing the problematic modules
    """
    import requests
    import tempfile
    from pypdf import PdfReader
    from langchain_core.documents import Document
    from pathlib import Path
    
    def load_from_url(pdf_url):
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        reader = PdfReader(tmp_path)
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            doc = Document(
                page_content=text,
                metadata={
                    'source': pdf_url,
                    'page': page_num
                }
            )
            documents.append(doc)
        
        Path(tmp_path).unlink()
        return documents
    
    return load_from_url


# Main execution
if __name__ == "__main__":
    pdf_url = "https://www.autosar.org/fileadmin/standards/R18-10_R4.4.0_R1.5.0/CP/AUTOSAR_SWS_CANNetworkManagement.pdf"
    
    print("="*80)
    print("Method 1: OnlinePDFLoader (Recommended)")
    print("="*80)
    
    try:
        documents = load_pdf_with_langchain_online(pdf_url)
        print(f"\nSuccessfully loaded {len(documents)} pages")
        print(f"\nFirst page preview:")
        print("-" * 80)
        print(documents[0].page_content[:500])
        print("-" * 80)
    except Exception as e:
        print(f"Error with OnlinePDFLoader: {e}")
        print("\nTrying alternative method...")
        
        # Fallback to manual implementation
        loader_func = load_pdf_with_pypdf_loader_manual()
        documents = loader_func(pdf_url)
        print(f"\nSuccessfully loaded {len(documents)} pages")
        print(f"\nFirst page preview:")
        print("-" * 80)
        print(documents[0].page_content[:500])
        print("-" * 80)
