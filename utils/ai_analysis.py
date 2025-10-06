# AI content analysis functions
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

def initialize_gemini_client():
    """
    Initialize Google Gemini client with API key.
    
    Returns:
        genai.Client: Initialized Gemini client
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file")
        return None
    
    return genai.Client(api_key=api_key)

def summarize_document(text):
    """
    Generate AI summary of document using Gemini.
    
    Args:
        text: Document text to summarize
    
    Returns:
        str: Generated summary
    """
    try:
        client = initialize_gemini_client()
        if not client:
            return None
        
        # Create prompt for summarization
        prompt = f"""
        Please provide a concise and comprehensive summary of the following document.
        Focus on the main points and key information.
        
        Document:
        {text}
        
        Summary:
        """
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for more focused output
                max_output_tokens=1024
            )
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def answer_question(text, question):
    """
    Answer questions about the document using Gemini.
    
    Args:
        text: Document text
        question: User's question
    
    Returns:
        str: AI-generated answer
    """
    try:
        client = initialize_gemini_client()
        if not client:
            return None
        
        # Create prompt for Q&A
        prompt = f"""
        Based on the following document, please answer the question below.
        If the answer is not in the document, say so clearly.
        
        Document:
        {text}
        
        Question: {question}
        
        Answer:
        """
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=512
            )
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return None
