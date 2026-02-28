"""
Simple PDF loading from URL without heavy dependencies
Uses pypdf directly to avoid unnecessary imports
"""

import requests
import tempfile
from pypdf import PdfReader
from pathlib import Path

def load_pdf_from_url(pdf_url):
    """
    Load PDF from URL and extract text from all pages
    
    Args:
        pdf_url: URL of the PDF file
        
    Returns:
        list of dicts with page content and metadata
    """
    print(f"Downloading PDF from: {pdf_url}")
    
    # Download the PDF
    response = requests.get(pdf_url, stream=True)
    response.raise_for_status()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    
    print(f"PDF downloaded successfully")
    
    # Read the PDF
    reader = PdfReader(tmp_path)
    documents = []
    
    print(f"Processing {len(reader.pages)} pages...")
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        documents.append({
            'page_content': text,
            'metadata': {
                'source': pdf_url,
                'page': page_num,
                'total_pages': len(reader.pages)
            }
        })
    
    # Clean up temp file
    Path(tmp_path).unlink()
    
    return documents


# Main execution
if __name__ == "__main__":
    pdf_url = "https://www.autosar.org/fileadmin/standards/R18-10_R4.4.0_R1.5.0/CP/AUTOSAR_SWS_CANNetworkManagement.pdf"
    
    print("="*80)
    print("Loading PDF from URL")
    print("="*80)
    
    try:
        documents = load_pdf_from_url(pdf_url)
        
        print(f"\nSuccessfully loaded {len(documents)} pages")
        print(f"\nFirst page preview:")
        print("-" * 80)
        print(documents[0]['page_content'][:500])
        print("-" * 80)
        
        print(f"\nMetadata of first page:")
        print(documents[0]['metadata'])
        
        # Show statistics
        total_chars = sum(len(doc['page_content']) for doc in documents)
        print(f"\nTotal characters: {total_chars:,}")
        
        # Search example
        search_term = "CAN"
        pages_with_term = [doc for doc in documents if search_term in doc['page_content']]
        print(f"\nPages containing '{search_term}': {len(pages_with_term)}")
        
    except Exception as e:
        print(f"Error: {e}")
