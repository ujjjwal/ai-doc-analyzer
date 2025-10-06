# Main Streamlit app

import streamlit as st
from utils.text_extraction import extract_text
from utils.ai_analysis import summarize_document, answer_question
from utils.summarizer import extract_keywords, analyze_sentiment

# Page configuration
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide"
)

# App title
st.title("📄 AI-Powered Document Analyzer")
st.markdown("Upload a document and analyze it with AI")

# Initialize session state for storing extracted text
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'filename' not in st.session_state:
    st.session_state.filename = None

# File uploader
uploaded_file = st.file_uploader(
    "Upload Document",
    type=['pdf', 'docx', 'txt'],
    help="Supported formats: PDF, DOCX, TXT"
)

# Process uploaded file
if uploaded_file is not None:
    # Check if new file is uploaded
    if st.session_state.filename != uploaded_file.name:
        st.session_state.filename = uploaded_file.name
        
        with st.spinner("Extracting text from document..."):
            # Extract text from uploaded file
            extracted_text = extract_text(uploaded_file)
            
            if extracted_text:
                st.session_state.extracted_text = extracted_text
                st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
            else:
                st.error("Failed to extract text from document")

# Display extracted text if available
if st.session_state.extracted_text:
    st.markdown("---")
    
    # Display document text
    st.subheader("📖 Document Content")
    with st.expander("View Full Text", expanded=False):
        st.text_area(
            "Extracted Text",
            st.session_state.extracted_text,
            height=300,
            disabled=True
        )
    
    # Show text statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        word_count = len(st.session_state.extracted_text.split())
        st.metric("Word Count", word_count)
    with col2:
        char_count = len(st.session_state.extracted_text)
        st.metric("Character Count", char_count)
    with col3:
        line_count = len(st.session_state.extracted_text.split('\n'))
        st.metric("Line Count", line_count)
    
    st.markdown("---")
    
    # AI Analysis Section
    st.subheader("🤖 AI Analysis")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns(3)
    
    # Summarize button
    with col1:
        if st.button("📝 Summarize", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = summarize_document(st.session_state.extracted_text)
                if summary:
                    st.markdown("### Summary")
                    st.info(summary)
    
    # Extract Keywords button
    with col2:
        if st.button("🔑 Extract Keywords", use_container_width=True):
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(st.session_state.extracted_text)
                if keywords:
                    st.markdown("### Keywords")
                    for keyword, score in keywords:
                        st.write(f"- **{keyword}** (score: {score:.3f})")
    
    # Analyze Sentiment button
    with col3:
        if st.button("😊 Analyze Sentiment", use_container_width=True):
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(st.session_state.extracted_text)
                if sentiment:
                    st.markdown("### Sentiment Analysis")
                    st.write(f"**Overall Sentiment:** {sentiment['label']}")
                    st.write(f"**Polarity:** {sentiment['polarity']} (range: -1 to 1)")
                    st.write(f"**Subjectivity:** {sentiment['subjectivity']} (range: 0 to 1)")
    
    st.markdown("---")
    
    # Q&A Section
    st.subheader("❓ Ask Questions")
    question = st.text_input(
        "Type your question about the document",
        placeholder="e.g., What is the main topic of this document?"
    )
    
    if st.button("Get Answer", use_container_width=False):
        if question.strip():
            with st.spinner("Finding answer..."):
                answer = answer_question(st.session_state.extracted_text, question)
                if answer:
                    st.markdown("### Answer")
                    st.success(answer)
        else:
            st.warning("Please enter a question")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a document to begin analysis")
    
    # Feature list
    st.markdown("### Features:")
    st.markdown("""
    - 📄 **Upload & Read** - Support for PDF, DOCX, and TXT files
    - 📖 **Display Text** - View extracted document content
    - 📝 **AI Summary** - Get intelligent summaries powered by Google Gemini
    - ❓ **Q&A** - Ask questions about your document
    - 🔑 **Keywords** - Extract important keywords automatically
    - 😊 **Sentiment** - Analyze document sentiment
    """)
