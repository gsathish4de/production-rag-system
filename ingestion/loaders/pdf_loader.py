"""
PDF Document Loader

Extracts text from PDF files for ingestion into the RAG system.
"""

from pathlib import Path
from typing import List, Dict, Optional
import pypdf
from datetime import datetime


class PDFLoader:
    """
    Load and extract text from PDF documents.
    
    Handles multi-page PDFs and extracts metadata.
    """
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    def load(self, file_path: str) -> Dict:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dict with keys:
                - text: Extracted text content
                - metadata: File metadata (filename, pages, etc.)
                - pages: List of page texts
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")
        
        print(f"Loading PDF: {file_path.name}")
        
        # Open PDF
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            
            # Extract text from all pages
            pages = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text
                    })
            
            # Combine all text
            full_text = "\n\n".join(p['text'] for p in pages)
            
            # Extract metadata
            metadata = {
                'filename': file_path.name,
                'filepath': str(file_path.absolute()),
                'num_pages': len(reader.pages),
                'num_chars': len(full_text),
                'num_words': len(full_text.split()),
                'loaded_at': datetime.now().isoformat()
            }
            
            # Add PDF metadata if available
            if reader.metadata:
                pdf_meta = reader.metadata
                if pdf_meta.title:
                    metadata['title'] = pdf_meta.title
                if pdf_meta.author:
                    metadata['author'] = pdf_meta.author
                if pdf_meta.subject:
                    metadata['subject'] = pdf_meta.subject
        
        # print(f"  ✓ Extracted {metadata['num_pages']} pages, "
        #       f"{metadata['num_words']} words")
        print(f"  Extracted {metadata['num_pages']} pages, "
      f"{metadata['num_words']} words")
        
        return {
            'text': full_text,
            'metadata': metadata,
            'pages': pages
        }
    
    def load_directory(self, directory: str) -> List[Dict]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of document dicts (one per PDF)
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all PDFs
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory}")
            return []
        
        print(f"\nFound {len(pdf_files)} PDF files in {directory.name}/")
        
        # Load each PDF
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.load(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(documents)}/{len(pdf_files)} PDFs")
        return documents
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text.
        
        Removes excessive whitespace, weird characters, etc.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive newlines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Replace multiple spaces with single space
        import re
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()


if __name__ == "__main__":
    # Test the PDF loader
    import sys
    
    loader = PDFLoader()
    
    # Test with a single file if provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        doc = loader.load(pdf_path)
        
        print("\n" + "="*60)
        print("METADATA:")
        print("="*60)
        for key, value in doc['metadata'].items():
            print(f"{key}: {value}")
        
        print("\n" + "="*60)
        print("TEXT PREVIEW (first 500 chars):")
        print("="*60)
        print(doc['text'][:500])
        print("...")
    else:
        # Test with sample_docs directory
        docs = loader.load_directory("data/sample_docs")
        
        if docs:
            print("\n" + "="*60)
            print("FIRST DOCUMENT PREVIEW:")
            print("="*60)
            print(f"File: {docs[0]['metadata']['filename']}")
            print(f"Pages: {docs[0]['metadata']['num_pages']}")
            print(f"Words: {docs[0]['metadata']['num_words']}")
            print(f"\nFirst 300 chars:\n{docs[0]['text'][:300]}...")
        
    print("\n✓ PDF loader test completed!")