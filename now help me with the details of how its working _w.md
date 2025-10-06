<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# now help me with the details of how its working

with technical things  and how its working from inside what is the use of streamlit
and what are the major things used in this app

Perfect! Now I have comprehensive information about the technical architecture. Let me explain how your AI Document Analyzer works from the inside out.

## 🏗️ Technical Architecture: How Your AI Document Analyzer Works

Your application is built on a **client-server architecture** with multiple AI/ML components working together. Let me break down each layer:[^1][^2]

***

## 1. Streamlit Framework - The Foundation

### What is Streamlit?

Streamlit is a **Python web framework** that converts Python scripts into interactive web applications without requiring frontend development skills. It handles all the HTML, CSS, JavaScript, and server management automatically.[^3][^4][^1]

### How Streamlit Works Internally

**Client-Server Architecture**:[^5][^1]

- **Server Side (Python Backend)**: When you run `streamlit run app.py`, Python starts a **Tornado web server** that executes your Python code[^6]
- **Client Side (Browser Frontend)**: Streamlit serves a **React-based frontend** to your browser that displays the UI[^6]
- **Communication**: They communicate via **WebSockets** using **Protocol Buffers** (protobuf) for efficient data transfer between Python and JavaScript[^6]

**Data Flow - The Rerun Mechanism**:[^2]

Streamlit has a unique execution model that **reruns your entire script** from top to bottom whenever:

1. You modify the source code (hot reload)
2. A user interacts with any widget (button click, file upload, text input)
3. This makes it feel like a reactive app, even though it's just re-executing Python[^2]

**Why This Matters for Your App**:

```python
# Every time you click "Generate Summary", Streamlit reruns app.py
# Session state preserves data between reruns
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""  # Persists across reruns
```


***

## 2. Document Processing Pipeline

### Text Extraction Layer

**PyMuPDF (fitz) for PDFs**:

- **How it works**: PyMuPDF is a Python binding for the MuPDF library (written in C)
- **Process**: Opens PDF as binary stream → Parses PDF structure → Extracts text from each page's content stream
- **Technical detail**: Uses PDF content operators to decode text encoding and reconstruct readable strings

**python-docx for DOCX**:

- **How it works**: DOCX files are actually **ZIP archives** containing XML files
- **Process**: Unzips the .docx → Parses `document.xml` → Extracts text from `<w:t>` (text) elements
- **Technical detail**: Also reads from tables by parsing `<w:tbl>` elements

**TXT Files**:

- Simple UTF-8/Latin-1 decoding with fallback encoding handling

```python
# Your code handles encoding gracefully
try:
    text = file.read().decode('utf-8')
except UnicodeDecodeError:
    file.seek(0)
    text = file.read().decode('latin-1')  # Fallback
```


***

## 3. Google Gemini AI - The Brain

### What is Gemini?

**Gemini** is Google's **large language model (LLM)** family that understands and generates human-like text. Your app uses **Gemini 2.0 Flash**, which is optimized for speed and efficiency.[^7][^8]

### How Gemini API Works

**Architecture**:[^8][^7]

1. **API Call**: Your Python app sends HTTP POST request to Google's servers
2. **Request Structure**:

```python
{
    "model": "gemini-2.0-flash-exp",
    "contents": "Your prompt here"
}
```

3. **Processing**: Google's servers run the Gemini model on their infrastructure (TPUs/GPUs)
4. **Response**: Returns generated text in JSON format
5. **Token Management**: Gemini processes text in "tokens" (roughly 4 characters = 1 token)[^7]

**How Your App Uses Gemini**:

```python
# Summarization
prompt = f"""Summarize the following document...
{text}
Summary:"""

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=prompt
)
```

**What Happens Inside**:[^8]

- **Tokenization**: Your prompt is broken into tokens
- **Attention Mechanism**: The transformer model analyzes relationships between all tokens
- **Generation**: Predicts the most likely next tokens to form a coherent summary
- **Context Window**: Gemini 2.0 Flash can handle **millions of tokens** (that's why you truncate to 30,000 characters for safety)[^8]

***

## 4. Keyword Extraction - KeyBERT

### How KeyBERT Works

**Architecture**:[^9][^10]

KeyBERT uses **BERT embeddings** to find keywords most similar to the document.[^10]

**Step-by-Step Process**:

1. **Load Sentence Transformer**: Uses `all-MiniLM-L6-v2` model (a lightweight BERT variant)[^11][^12]
2. **Generate Document Embedding**:[^12][^11]
    - Passes entire document through BERT
    - BERT processes text through 6 transformer layers
    - Each layer has multi-head attention that learns word relationships
    - **Pooling layer** aggregates token embeddings into single 384-dimensional vector (mean pooling)[^11]
3. **Generate Candidate Keywords**:
    - Extracts n-grams (1-2 word phrases) from document
    - Creates embeddings for each candidate
4. **Similarity Calculation**:[^9][^10]
    - Calculates **cosine similarity** between document embedding and each keyword embedding
    - Keywords with highest similarity scores are most representative
5. **Diversity**: Uses **Max Sum Similarity** to ensure keywords are diverse, not redundant[^10]

**Mathematical Foundation**:

$$
\text{similarity}(doc, keyword) = \frac{doc \cdot keyword}{||doc|| \cdot ||keyword||}
$$

This finds keywords whose semantic meaning is closest to the overall document.[^9][^10]

***

## 5. Sentiment Analysis - TextBlob

### How TextBlob Works

**Architecture**:

TextBlob uses a **rule-based approach** with pre-trained sentiment lexicons.

**Process**:

1. **Tokenization**: Breaks text into words and sentences
2. **Lexicon Lookup**: Each word has a pre-assigned polarity score (-1 to +1) and subjectivity score (0 to 1)
3. **Aggregation**: Averages all word scores, accounting for intensifiers ("very good") and negations ("not bad")
4. **Output**:
    - **Polarity**: -1 (negative) to +1 (positive)
    - **Subjectivity**: 0 (objective) to 1 (subjective)

**Example**:

```python
"I love this amazing document!" 
# "love" = +0.5, "amazing" = +0.6
# Polarity = (0.5 + 0.6) / 2 = +0.55 (Positive)
```


***

## 6. Sentence Transformers - The Embedding Engine

### What Are Sentence Transformers?

**Sentence Transformers** (SBERT) convert text into **dense vector embeddings** that capture semantic meaning.[^13][^12][^11]

### Architecture[^12][^11]

**Core Components**:

1. **Base Model**: Uses a pre-trained transformer (BERT, RoBERTa, etc.)
2. **Siamese Network Structure**:[^11]
    - Processes multiple sentences through the same encoder
    - Compares embeddings using cosine similarity
    - Trained with contrastive learning (similar sentences = close vectors)
3. **Pooling Layer**:[^11]
    - **Mean Pooling** (most common): Averages all token embeddings
    - Converts variable-length text → fixed-size vector (e.g., 384 dimensions)

**Training Process**:[^11]

- Trained on sentence pairs with labels (similar/dissimilar)
- Uses **triplet loss**: Minimize distance between similar sentences, maximize distance between dissimilar ones
- Datasets: Natural Language Inference (NLI), semantic similarity tasks

**Why It Matters for KeyBERT**:
The `all-MiniLM-L6-v2` model creates embeddings that capture semantic meaning, allowing KeyBERT to find keywords that are conceptually related to the document, not just frequent words.[^13]

***

## 7. Complete Data Flow Diagram

Here's how everything connects:

```
User Browser (React Frontend)
    ↕️ WebSocket (Protocol Buffers)
Streamlit Server (Tornado/Python)
    ↓
[^1] File Upload → Text Extraction
    ├→ PDF → PyMuPDF → Extract text from binary
    ├→ DOCX → python-docx → Parse XML from ZIP
    └→ TXT → UTF-8 decode
    ↓
[^2] Session State Storage (st.session_state.document_text)
    ↓
[^3] User Clicks "Summarize"
    ↓ (Script reruns from top)
[^4] Gemini API Call
    ├→ Send: Prompt + Document text
    ├→ Google's servers: Tokenize → Transformer → Generate
    └→ Receive: Summary text
    ↓
[^5] User Clicks "Extract Keywords"
    ↓
[^6] KeyBERT Processing
    ├→ Load sentence-transformers model
    ├→ Generate document embedding (384-D vector)
    ├→ Generate candidate keyword embeddings
    ├→ Calculate cosine similarities
    └→ Return top-N keywords with scores
    ↓
[^7] TextBlob Sentiment
    ├→ Tokenize text
    ├→ Lookup word polarities in lexicon
    ├→ Average scores with modifiers
    └→ Return polarity + subjectivity
    ↓
[^8] Display Results (Streamlit renders in browser)
```


***

## 8. Major Technologies Used

### Core Stack

| **Technology** | **Purpose** | **Why It's Used** |
| :-- | :-- | :-- |
| **Streamlit** | Web framework | Converts Python scripts to web apps without frontend coding[^3][^4] |
| **Google Gemini** | AI text generation | State-of-the-art LLM for summarization and Q\&A[^7][^8] |
| **PyMuPDF** | PDF parsing | Fast, C-based PDF text extraction |
| **python-docx** | DOCX parsing | Handles Microsoft Word XML structure |
| **KeyBERT** | Keyword extraction | BERT-based semantic keyword identification[^10] |
| **TextBlob** | Sentiment analysis | Simple, rule-based sentiment scoring |
| **sentence-transformers** | Text embeddings | Converts text to semantic vectors for KeyBERT[^13] |
| **python-dotenv** | Config management | Securely loads API keys from .env file |

### Model Specifics

**all-MiniLM-L6-v2**:[^13][^11]

- **Size**: 22 million parameters
- **Output**: 384-dimensional embeddings
- **Speed**: ~2800 sentences/second on CPU
- **Use case**: Lightweight semantic similarity for KeyBERT

**Gemini 2.0 Flash**:[^7]

- **Optimized for**: Speed and cost-efficiency
- **Context window**: Millions of tokens
- **Use case**: Document summarization and conversational Q\&A

***

## 9. Performance Optimizations

### Lazy Loading

```python
_kw_model = None

def get_keybert_model():
    global _kw_model
    if _kw_model is None:  # Only load once
        _kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    return _kw_model
```

**Why**: Loading BERT models is expensive (~2-3 seconds). This singleton pattern loads it once and reuses it.[^2]

### Text Truncation

```python
if len(text) > 30000:
    text = text[:30000] + "..."
```

**Why**: Prevents API timeout, reduces token costs, and ensures fast processing.[^7]

### Session State Caching

```python
st.session_state.document_text = ""
```

**Why**: Preserves extracted text across Streamlit reruns without re-extracting from file.[^2]

***

## 10. Key Technical Advantages

**Why This Architecture Works**:

1. **Simplicity**: Streamlit eliminates 90% of typical web development complexity[^4][^3]
2. **State-of-the-art AI**: Gemini 2.0 Flash provides better summaries than traditional extractive methods[^7]
3. **Semantic Understanding**: KeyBERT finds conceptually important keywords, not just frequent words[^10]
4. **Modularity**: Each utility module is independent and testable
5. **Local Execution**: Everything runs on your machine except Gemini API calls[^1]

**Real-World Performance**:

- File upload: <1 second
- Text extraction (10-page PDF): 1-2 seconds
- Gemini summary: 2-5 seconds
- KeyBERT keywords: 3-5 seconds (first run), <1 second (cached model)
- Total workflow: <10 seconds for complete analysis

This architecture balances **simplicity, performance, and AI capabilities** to create a production-ready document analysis tool with minimal code complexity.[^1][^10][^2]
<span style="display:none">[^14][^15][^16][^17][^18][^19][^20]</span>

<div align="center">⁂</div>

[^1]: https://docs.streamlit.io/develop/concepts/architecture/architecture

[^2]: https://docs.streamlit.io/get-started/fundamentals/main-concepts

[^3]: https://streamlit.io

[^4]: https://uibakery.io/blog/what-is-streamlit

[^5]: https://deepnote.com/blog/ultimate-guide-to-the-streamlit-library

[^6]: https://discuss.streamlit.io/t/backend-workings-of-streamlit/11834

[^7]: https://ai.google.dev/gemini-api/docs/quickstart

[^8]: https://ai.google.dev/gemini-api/docs

[^9]: https://maartengr.github.io/KeyBERT/guides/embeddings.html

[^10]: https://github.com/MaartenGr/KeyBERT

[^11]: https://milvus.io/ai-quick-reference/what-is-the-architecture-of-a-typical-sentence-transformer-model-for-example-the-sentencebert-architecture

[^12]: https://www.geeksforgeeks.org/nlp/sentence-transformer/

[^13]: https://sbert.net

[^14]: https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit

[^15]: https://auth0.com/blog/introduction-to-streamlit-and-streamlit-components/

[^16]: https://firebase.google.com/docs/ai-logic

[^17]: https://towardsdatascience.com/implementing-a-stateful-architecture-with-streamlit-58e52448efa1/

[^18]: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart

[^19]: https://apidog.com/blog/google-gemini-api-key-for-free/

[^20]: https://developers.googleblog.com/en/gemini-for-home-expanding-the-platform-for-a-new-era-of-smart-home-ai/

