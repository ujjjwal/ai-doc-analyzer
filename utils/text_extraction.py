# Handles text extraction from PDF, DOCX, and TXT files
import pymupdf  # PyMuPDF
from docx import Document
import streamlit as st

def extract_text_from_pdf(file):
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        file: Uploaded file object
    
    Returns:
        str: Extracted text from PDF
    """
    try:
        # Read PDF from uploaded file
        pdf_bytes = file.read()
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        # Iterate through all pages
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def extract_text_from_docx(file):
    """
    Extract text from DOCX file using python-docx.
    
    Args:
        file: Uploaded file object
    
    Returns:
        str: Extracted text from DOCX
    """
    try:
        # Read DOCX from uploaded file
        doc = Document(file)
        
        text = ""
        # Extract text from all paragraphs
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error extracting DOCX: {str(e)}")
        return None

def extract_text_from_txt(file):
    """
    Extract text from TXT file.
    
    Args:
        file: Uploaded file object
    
    Returns:
        str: Extracted text from TXT
    """
    try:
        # Read TXT file and decode
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error extracting TXT: {str(e)}")
        return None

def extract_text(file):
    """
    Main function to extract text based on file type.
    
    Args:
        file: Uploaded file object
    
    Returns:
        str: Extracted text
    """
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type == 'docx':
        return extract_text_from_docx(file)
    elif file_type == 'txt':
        return extract_text_from_txt(file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None
