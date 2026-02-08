# RAG and Knowledge Systems: Comprehensive Guide

## Table of Contents
1. [RAG Fundamentals](#rag-fundamentals)
2. [Document Processing](#document-processing)
3. [Chunking Strategies](#chunking-strategies)
4. [Embedding Models](#embedding-models)
5. [Vector Databases](#vector-databases)
6. [Retrieval Methods](#retrieval-methods)
7. [Advanced RAG Patterns](#advanced-rag-patterns)
8. [Knowledge Graphs](#knowledge-graphs)
9. [Evaluation](#evaluation)

---

## 1. RAG Fundamentals

### 1.1 What is RAG and Why It Matters

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant context from external knowledge sources before generating responses. Instead of relying solely on the model's training data, RAG retrieves pertinent information from a knowledge base and includes it in the prompt.

**Why RAG Matters:**

1. **Up-to-date Information**: LLMs have training cutoffs. RAG allows access to current information without retraining.
2. **Domain-Specific Knowledge**: Incorporate proprietary or specialized knowledge without fine-tuning.
3. **Transparency**: Users can see the sources of information used in generation.
4. **Cost Efficiency**: More economical than fine-tuning large models.
5. **Reduced Hallucination**: Grounding responses in retrieved documents reduces fabrication.

### 1.2 RAG vs Fine-tuning vs Prompt Engineering

| Aspect | RAG | Fine-tuning | Prompt Engineering |
|--------|-----|-------------|-------------------|
| **Cost** | Low to Medium | High | Very Low |
| **Time to Deploy** | Hours to Days | Days to Weeks | Minutes |
| **Knowledge Update** | Real-time | Requires retraining | Manual prompt updates |
| **Domain Adaptation** | Excellent | Excellent | Limited |
| **Hallucination Control** | High (source-grounded) | Medium | Low |
| **Scalability** | High | Medium | Low |
| **Transparency** | High (citable sources) | Low | Medium |
| **Best For** | Dynamic knowledge, citations | Task-specific behavior | Simple tasks, few examples |

**When to Use Each:**

- **RAG**: When you need current information, domain-specific knowledge, or source citations
- **Fine-tuning**: When you need to change model behavior, style, or task-specific optimization
- **Prompt Engineering**: For simple tasks, few-shot learning, or rapid prototyping

### 1.3 The RAG Pipeline

The RAG pipeline consists of several interconnected stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                              │
└─────────────────────────────────────────────────────────────────┘

1. INGEST
   ┌─────────────┐
   │  Documents  │ → PDF, HTML, Markdown, CSV, Databases, APIs
   └─────────────┘

2. PROCESS & CHUNK
   ┌─────────────┐     ┌─────────────┐
   │   Extract   │ →  │    Chunk    │ → Fixed, Semantic, Document-aware
   │    Text     │     │  Strategies │
   └─────────────┘     └─────────────┘

3. EMBED
   ┌─────────────┐     ┌─────────────┐
   │   Chunks    │ →  │  Embedding  │ → Vector representations
   │             │     │    Model    │
   └─────────────┘     └─────────────┘

4. STORE
   ┌─────────────┐     ┌─────────────┐
   │  Vectors +  │ →  │    Vector   │ → ChromaDB, Pinecone, Weaviate
   │  Metadata   │     │  Database   │
   └─────────────┘     └─────────────┘

5. RETRIEVE (Query Time)
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │    User     │ →  │   Embed     │ →  │   Search    │
   │   Query     │     │   Query     │     │  Similarity │
   └─────────────┘     └─────────────┘     └─────────────┘

6. AUGMENT
   ┌─────────────┐     ┌─────────────┐
   │ Retrieved   │ →  │  Construct  │ → Context + Query → Prompt
   │  Context    │     │   Prompt    │
   └─────────────┘     └─────────────┘

7. GENERATE
   ┌─────────────┐     ┌─────────────┐
   │   Prompt    │ →  │     LLM     │ → Final Response
   │             │     │             │
   └─────────────┘     └─────────────┘
```

**Python Implementation - Basic RAG Pipeline:**

```python
from typing import List, Dict
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class BasicRAGPipeline:
    def __init__(self, documents: List[str], embedding_model: str = "text-embedding-3-small"):
        """
        Initialize RAG pipeline components.
        
        Args:
            documents: List of document texts
            embedding_model: Name of embedding model to use
        """
        self.documents = documents
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.retriever = None
        
    def ingest(self) -> List[str]:
        """Step 1: Process documents into chunks."""
        print("Step 1: Ingesting and chunking documents...")
        chunks = []
        for doc in self.documents:
            doc_chunks = self.text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
        print(f"Created {len(chunks)} chunks from {len(self.documents)} documents")
        return chunks
    
    def embed_and_store(self, chunks: List[str]):
        """Steps 2-3: Embed chunks and store in vector database."""
        print("Step 2-3: Embedding and storing chunks...")
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embedding_model,
            persist_directory="./chroma_db"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Vector store created successfully")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Step 4: Retrieve relevant chunks for query."""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call embed_and_store first.")
        
        docs = self.retriever.get_relevant_documents(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    
    def augment_and_generate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Steps 5-6: Augment prompt with context and generate response."""
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        
        prompt = f"""Use the following context to answer the question. 
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        llm = OpenAI(temperature=0)
        response = llm(prompt)
        return response
    
    def query(self, query: str) -> Dict:
        """Complete RAG query pipeline."""
        retrieved_docs = self.retrieve(query)
        answer = self.augment_and_generate(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_docs
        }

# Example Usage
if __name__ == "__main__":
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning involves training models on data to make predictions.",
        "RAG combines retrieval and generation for better AI responses."
    ]
    
    rag = BasicRAGPipeline(documents)
    chunks = rag.ingest()
    rag.embed_and_store(chunks)
    
    result = rag.query("What is Python?")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} documents retrieved")
```

### 1.4 When to Use RAG

**Use RAG When:**

1. **Dynamic Knowledge Requirements**: Information changes frequently (news, product catalogs, documentation)
2. **Large Knowledge Bases**: Too much information to fit in context window
3. **Source Attribution**: Need to cite sources for legal/compliance reasons
4. **Domain-Specific Applications**: Medical, legal, technical documentation
5. **Cost Constraints**: Cannot afford fine-tuning large models
6. **Multi-Modal Knowledge**: Combining text, images, structured data

**Don't Use RAG When:**

1. **Simple Tasks**: Basic Q&A that doesn't require external knowledge
2. **Real-time Requirements**: Latency-sensitive applications (<100ms)
3. **Small Knowledge Base**: Can fit entire knowledge in prompt
4. **Behavioral Changes**: Need to change how model reasons, not what it knows

---

## 2. Document Processing

### 2.1 Document Loaders

Document loaders extract content from various sources. Different formats require specialized handling.

**PDF Loaders:**

```python
from langchain.document_loaders import PyPDFLoader, PDFPlumberLoader
from pypdf import PdfReader
import fitz  # PyMuPDF

class PDFProcessor:
    """Handle PDF document loading with multiple backends."""
    
    @staticmethod
    def load_with_pypdf(file_path: str):
        """Using PyPDF - good for text extraction."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    
    @staticmethod
    def load_with_pdfplumber(file_path: str):
        """Using PDFPlumber - better for tables and complex layouts."""
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        return documents
    
    @staticmethod
    def load_with_pymupdf(file_path: str):
        """Using PyMuPDF - fast and handles images."""
        doc = fitz.open(file_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append({
                "page": page_num + 1,
                "content": text,
                "metadata": {
                    "source": file_path,
                    "page": page_num + 1
                }
            })
        return pages
    
    @staticmethod
    def extract_with_ocr(file_path: str):
        """Extract text from scanned PDFs using OCR."""
        import pytesseract
        from pdf2image import convert_from_path
        
        images = convert_from_path(file_path)
        text_content = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            text_content.append({
                "page": i + 1,
                "content": text,
                "metadata": {"source": file_path, "page": i + 1, "method": "OCR"}
            })
        return text_content

# Example Usage
processor = PDFProcessor()
documents = processor.load_with_pypdf("document.pdf")
```

**HTML Loaders:**

```python
from langchain.document_loaders import WebBaseLoader, UnstructuredHTMLLoader
from bs4 import BeautifulSoup
import requests

class HTMLProcessor:
    """Process HTML documents and web pages."""
    
    @staticmethod
    def load_from_url(url: str):
        """Load content from a web URL."""
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    
    @staticmethod
    def load_from_file(file_path: str):
        """Load HTML from local file."""
        loader = UnstructuredHTMLLoader(file_path)
        documents = loader.load()
        return documents
    
    @staticmethod
    def extract_with_beautifulsoup(html_content: str):
        """Custom extraction using BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            "content": text,
            "title": soup.title.string if soup.title else None,
            "metadata": {
                "source": "html_content",
                "extraction_method": "beautifulsoup"
            }
        }
    
    @staticmethod
    def extract_structured_data(html_content: str):
        """Extract structured data from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        data = {
            "headings": [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])],
            "links": [{"text": a.get_text(), "href": a.get('href')} 
                     for a in soup.find_all('a', href=True)],
            "images": [{"alt": img.get('alt'), "src": img.get('src')} 
                      for img in soup.find_all('img', src=True)],
            "tables": []
        }
        
        # Extract table data
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                rows.append(cells)
            data["tables"].append(rows)
        
        return data

# Example Usage
html_processor = HTMLProcessor()
documents = html_processor.load_from_url("https://example.com/article")
```

**Markdown Loaders:**

```python
from langchain.document_loaders import UnstructuredMarkdownLoader
import markdown
from markdown.extensions import codehilite, fenced_code

class MarkdownProcessor:
    """Process Markdown documents."""
    
    @staticmethod
    def load_markdown(file_path: str):
        """Load markdown file."""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        return documents
    
    @staticmethod
    def parse_markdown(content: str):
        """Parse markdown with extensions."""
        md = markdown.Markdown(extensions=['codehilite', 'fenced_code', 'tables'])
        html = md.convert(content)
        
        # Extract structure
        structure = {
            "headings": [],
            "code_blocks": [],
            "links": [],
            "images": []
        }
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                structure["headings"].append({
                    "level": level,
                    "text": line.lstrip('# ').strip(),
                    "line": i
                })
        
        return {
            "html": html,
            "structure": structure,
            "raw_content": content
        }
```

**CSV Loaders:**

```python
from langchain.document_loaders import CSVLoader
import pandas as pd

class CSVProcessor:
    """Process CSV files."""
    
    @staticmethod
    def load_csv(file_path: str):
        """Load CSV with LangChain."""
        loader = CSVLoader(file_path)
        documents = loader.load()
        return documents
    
    @staticmethod
    def load_with_pandas(file_path: str, text_columns: List[str] = None):
        """Load CSV with pandas for more control."""
        df = pd.read_csv(file_path)
        
        if text_columns is None:
            text_columns = df.columns.tolist()
        
        documents = []
        for idx, row in df.iterrows():
            # Combine specified columns into text
            text_parts = [f"{col}: {row[col]}" for col in text_columns]
            text = "\n".join(text_parts)
            
            documents.append({
                "content": text,
                "metadata": {
                    "source": file_path,
                    "row": idx,
                    **{col: row[col] for col in df.columns if col not in text_columns}
                }
            })
        
        return documents
```

**Database Loaders:**

```python
from langchain.document_loaders import SQLDatabaseLoader
from sqlalchemy import create_engine, text

class DatabaseProcessor:
    """Load documents from databases."""
    
    @staticmethod
    def load_from_sql(query: str, connection_string: str):
        """Load data using SQL query."""
        engine = create_engine(connection_string)
        loader = SQLDatabaseLoader(query, engine)
        documents = loader.load()
        return documents
    
    @staticmethod
    def load_table_as_documents(table_name: str, connection_string: str, 
                                text_columns: List[str]):
        """Load entire table as documents."""
        engine = create_engine(connection_string)
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        documents = []
        for idx, row in df.iterrows():
            text = "\n".join([f"{col}: {row[col]}" for col in text_columns])
            documents.append({
                "content": text,
                "metadata": {
                    "table": table_name,
                    "row_id": idx,
                    **{k: v for k, v in row.items() if k not in text_columns}
                }
            })
        
        return documents
```

### 2.2 Text Extraction and Cleaning

Raw extracted text often needs cleaning before chunking and embedding.

```python
import re
from typing import List
import unicodedata

class TextCleaner:
    """Clean and normalize extracted text."""
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters."""
        if keep_punctuation:
            # Keep letters, numbers, spaces, and common punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
        else:
            # Keep only alphanumeric and spaces
            text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        return text
    
    @staticmethod
    def remove_headers_footers(text: str, header_pattern: str = None, 
                               footer_pattern: str = None) -> str:
        """Remove headers and footers."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip header if pattern matches
            if header_pattern and re.match(header_pattern, line):
                continue
            # Skip footer if pattern matches
            if footer_pattern and re.match(footer_pattern, line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def clean_text(text: str, 
                   remove_whitespace: bool = True,
                   normalize_unicode: bool = True,
                   remove_special: bool = False) -> str:
        """Comprehensive text cleaning."""
        if normalize_unicode:
            text = TextCleaner.normalize_unicode(text)
        
        if remove_special:
            text = TextCleaner.remove_special_characters(text)
        
        if remove_whitespace:
            text = TextCleaner.remove_extra_whitespace(text)
        
        return text

# Example Usage
cleaner = TextCleaner()
dirty_text = "This   is    a    test\n\n\n\nWith   multiple   spaces"
clean_text = cleaner.clean_text(dirty_text)
print(clean_text)  # "This is a test\n\nWith multiple spaces"
```

### 2.3 Metadata Extraction

Metadata helps with filtering, organization, and source attribution.

```python
from datetime import datetime
from pathlib import Path
import hashlib

class MetadataExtractor:
    """Extract and manage document metadata."""
    
    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict:
        """Extract metadata from file."""
        path = Path(file_path)
        
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix,
            "file_size": path.stat().st_size,
            "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        return metadata
    
    @staticmethod
    def extract_content_metadata(text: str) -> Dict:
        """Extract metadata from text content."""
        lines = text.split('\n')
        
        metadata = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "has_code": bool(re.search(r'```|def |class |import ', text)),
            "has_tables": bool(re.search(r'\|.*\|', text)),
            "has_links": bool(re.search(r'http[s]?://', text))
        }
        
        return metadata
    
    @staticmethod
    def generate_document_id(text: str, source: str) -> str:
        """Generate unique document ID."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{content_hash}"
    
    @staticmethod
    def extract_structured_metadata(text: str) -> Dict:
        """Extract structured information from text."""
        metadata = {}
        
        # Extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}'
        dates = re.findall(date_pattern, text)
        if dates:
            metadata["dates"] = dates
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            metadata["emails"] = emails
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        if urls:
            metadata["urls"] = urls
        
        # Extract potential entities (simple heuristic)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        if len(capitalized_words) > 10:
            metadata["potential_entities"] = list(set(capitalized_words[:20]))
        
        return metadata

# Example Usage
extractor = MetadataExtractor()
file_meta = extractor.extract_file_metadata("document.pdf")
content_meta = extractor.extract_content_metadata(text_content)
structured_meta = extractor.extract_structured_metadata(text_content)

full_metadata = {
    **file_meta,
    **content_meta,
    **structured_meta,
    "document_id": extractor.generate_document_id(text_content, file_meta["source"])
}
```

---

## 3. Chunking Strategies

Chunking is critical for RAG performance. Poor chunking leads to incomplete context or irrelevant retrieval.

### 3.1 Fixed-Size Chunking

Simplest approach: split text into fixed-size chunks.

```python
from langchain.text_splitter import CharacterTextSplitter

class FixedSizeChunker:
    """Fixed-size chunking implementation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_with_metadata(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk with metadata preservation."""
        chunks = self.chunk(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **(metadata or {}),
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            }
            result.append({
                "content": chunk,
                "metadata": chunk_meta
            })
        
        return result

# Example
chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
text = "Your long document text here..."
chunks = chunker.chunk(text)
```

**Pros:**
- Simple and fast
- Predictable chunk sizes
- Easy to implement

**Cons:**
- May split sentences/paragraphs mid-way
- Loses semantic coherence
- Can break context

### 3.2 Recursive Character Splitting

Splits on multiple separators recursively, preserving structure.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RecursiveChunker:
    """Recursive character splitting with hierarchy."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try in order
        )
    
    def chunk(self, text: str) -> List[str]:
        """Split recursively."""
        return self.splitter.split_text(text)
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents with metadata."""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.splitter.split_text(doc["content"])
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks_in_doc": len(chunks)
                    }
                })
        
        return all_chunks

# Example
recursive_chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = recursive_chunker.chunk(long_text)
```

**Separator Priority:**
1. Paragraph breaks (`\n\n`)
2. Line breaks (`\n`)
3. Sentences (`. `)
4. Words (` `)
5. Characters (``)

### 3.3 Semantic Chunking

Chunks based on semantic similarity, keeping related content together.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """Chunk based on semantic similarity."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 similarity_threshold: float = 0.7):
        """
        Args:
            embedding_model: Sentence transformer model name
            similarity_threshold: Minimum similarity to keep in same chunk
        """
        self.model = SentenceTransformer(embedding_model)
        self.threshold = similarity_threshold
    
    def chunk(self, text: str, min_chunk_size: int = 100) -> List[str]:
        """Semantic chunking."""
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return [text]
        
        # Embed sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = cosine_similarity(
                [current_embedding],
                [embeddings[i]]
            )[0][0]
            
            if similarity >= self.threshold:
                # Add to current chunk
                current_chunk.append(sentences[i])
                # Update chunk embedding (average)
                current_embedding = (current_embedding + embeddings[i]) / 2
            else:
                # Start new chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks

# Example
semantic_chunker = SemanticChunker(similarity_threshold=0.75)
semantic_chunks = semantic_chunker.chunk(long_document)
```

**Pros:**
- Preserves semantic coherence
- Better retrieval quality
- Adapts to content structure

**Cons:**
- Slower (requires embeddings)
- Variable chunk sizes
- More complex implementation

### 3.4 Document-Aware Chunking

Respects document structure (headers, paragraphs, sections).

```python
import re
from typing import List, Tuple

class DocumentAwareChunker:
    """Chunk respecting document structure."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def identify_structure(self, text: str) -> List[Tuple[str, str, int]]:
        """Identify document structure elements."""
        lines = text.split('\n')
        structure = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for headers
            if re.match(r'^#{1,6}\s+', line_stripped):
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                structure.append(('header', line_stripped, level))
            elif re.match(r'^[A-Z][A-Z\s]{10,}$', line_stripped):
                structure.append(('header', line_stripped, 0))
            elif len(line_stripped) > 0:
                structure.append(('content', line_stripped, 0))
            else:
                structure.append(('blank', '', 0))
        
        return structure
    
    def chunk_by_sections(self, text: str) -> List[Dict]:
        """Chunk respecting section boundaries."""
        structure = self.identify_structure(text)
        chunks = []
        current_section = []
        current_header = None
        
        for elem_type, content, level in structure:
            if elem_type == 'header':
                # Save previous section if exists
                if current_section:
                    section_text = '\n'.join(current_section)
                    if len(section_text) > 0:
                        chunks.append({
                            "content": section_text,
                            "metadata": {
                                "header": current_header,
                                "type": "section"
                            }
                        })
                
                # Start new section
                current_header = content
                current_section = [content]
            
            elif elem_type == 'content':
                current_section.append(content)
            
            # If section gets too large, split it
            section_text = '\n'.join(current_section)
            if len(section_text) > self.max_chunk_size:
                # Split current section
                sub_chunks = self._split_large_section(section_text, current_header)
                chunks.extend(sub_chunks[:-1])  # All but last
                current_section = [sub_chunks[-1]["content"]]
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            if len(section_text) > 0:
                chunks.append({
                    "content": section_text,
                    "metadata": {
                        "header": current_header,
                        "type": "section"
                    }
                })
        
        return chunks
    
    def _split_large_section(self, text: str, header: str) -> List[Dict]:
        """Split large section into smaller chunks."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "content": '\n\n'.join(current_chunk),
                    "metadata": {"header": header, "type": "subsection"}
                })
                # Start new chunk with overlap
                overlap_text = '\n\n'.join(current_chunk[-self.overlap//50:])
                current_chunk = [overlap_text, para]
                current_size = len(overlap_text) + para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                "content": '\n\n'.join(current_chunk),
                "metadata": {"header": header, "type": "subsection"}
            })
        
        return chunks

# Example
doc_chunker = DocumentAwareChunker(max_chunk_size=1000)
structured_chunks = doc_chunker.chunk_by_sections(markdown_document)
```

### 3.5 Chunk Size Optimization

Optimal chunk size depends on use case, model, and content type.

```python
import time
from typing import Dict, List

class ChunkSizeOptimizer:
    """Optimize chunk size through experimentation."""
    
    def __init__(self, embedding_model, retriever):
        self.embedding_model = embedding_model
        self.retriever = retriever
    
    def benchmark_chunk_sizes(self, text: str, 
                             chunk_sizes: List[int],
                             test_queries: List[str]) -> Dict:
        """Benchmark different chunk sizes."""
        results = {}
        
        for chunk_size in chunk_sizes:
            print(f"Testing chunk size: {chunk_size}")
            
            # Chunk text
            chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_size//5)
            chunks = chunker.chunk(text)
            
            # Embed and store
            embeddings = self.embedding_model.encode(chunks)
            # Store in vector DB (simplified)
            
            # Test retrieval
            query_results = []
            for query in test_queries:
                query_embedding = self.embedding_model.encode([query])[0]
                
                # Calculate similarities
                similarities = cosine_similarity([query_embedding], embeddings)[0]
                top_k_indices = np.argsort(similarities)[-5:][::-1]
                
                query_results.append({
                    "query": query,
                    "top_similarity": float(similarities[top_k_indices[0]]),
                    "avg_top5_similarity": float(np.mean(similarities[top_k_indices]))
                })
            
            results[chunk_size] = {
                "num_chunks": len(chunks),
                "avg_chunk_size": np.mean([len(c) for c in chunks]),
                "query_results": query_results,
                "avg_top_similarity": np.mean([r["top_similarity"] for r in query_results])
            }
        
        return results
    
    def find_optimal_size(self, benchmark_results: Dict) -> int:
        """Find optimal chunk size from benchmark results."""
        best_size = None
        best_score = -1
        
        for chunk_size, results in benchmark_results.items():
            # Score based on similarity and chunk count
            similarity_score = results["avg_top_similarity"]
            chunk_count_penalty = 1.0 / (1.0 + results["num_chunks"] / 100)  # Prefer fewer chunks
            score = similarity_score * 0.7 + chunk_count_penalty * 0.3
            
            if score > best_score:
                best_score = score
                best_size = chunk_size
        
        return best_size

# Usage
optimizer = ChunkSizeOptimizer(embedding_model, retriever)
benchmark_results = optimizer.benchmark_chunk_sizes(
    text=document_text,
    chunk_sizes=[256, 512, 1024, 2048, 4096],
    test_queries=["What is the main topic?", "Explain the key concepts"]
)
optimal_size = optimizer.find_optimal_size(benchmark_results)
print(f"Optimal chunk size: {optimal_size}")
```

**Chunk Size Guidelines:**

| Content Type | Recommended Size | Reasoning |
|--------------|------------------|-----------|
| Code | 200-500 chars | Functions/classes are small units |
| Technical Docs | 500-1000 chars | Concepts fit in medium chunks |
| Articles/Blogs | 1000-2000 chars | Paragraphs are natural units |
| Books | 2000-4000 chars | Chapters/sections are large |
| Legal Docs | 500-1500 chars | Clauses need context |

### 3.6 Overlap Strategies

Overlap prevents losing context at chunk boundaries.

```python
class OverlapStrategy:
    """Different overlap strategies."""
    
    @staticmethod
    def fixed_overlap(chunk_size: int, overlap_size: int) -> int:
        """Fixed overlap size."""
        return overlap_size
    
    @staticmethod
    def percentage_overlap(chunk_size: int, overlap_percent: float = 0.2) -> int:
        """Overlap as percentage of chunk size."""
        return int(chunk_size * overlap_percent)
    
    @staticmethod
    def sentence_aware_overlap(text: str, chunk_end: int, 
                               target_overlap: int) -> int:
        """Adjust overlap to end at sentence boundary."""
        # Find last sentence end before target
        sentences = list(re.finditer(r'[.!?]\s+', text[:chunk_end]))
        if sentences:
            last_sentence_end = sentences[-1].end()
            if chunk_end - last_sentence_end <= target_overlap * 1.5:
                return chunk_end - last_sentence_end
        return target_overlap
    
    @staticmethod
    def paragraph_aware_overlap(text: str, chunk_end: int,
                                target_overlap: int) -> int:
        """Adjust overlap to end at paragraph boundary."""
        last_para_break = text[:chunk_end].rfind('\n\n')
        if last_para_break != -1 and chunk_end - last_para_break <= target_overlap * 2:
            return chunk_end - last_para_break
        return target_overlap

# Example
strategy = OverlapStrategy()
overlap = strategy.percentage_overlap(chunk_size=1000, overlap_percent=0.2)  # 200 chars
```

**Overlap Recommendations:**
- **Small chunks (<500)**: 10-20% overlap
- **Medium chunks (500-2000)**: 15-25% overlap
- **Large chunks (>2000)**: 10-15% overlap

### 3.7 Agentic Chunking

Agents decide chunking strategy based on content analysis.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class AgenticChunker:
    """AI agent decides chunking strategy."""
    
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.strategy_prompt = PromptTemplate(
            input_variables=["content_sample", "content_type"],
            template="""
Analyze this content sample and recommend chunking strategy.

Content Type: {content_type}
Content Sample (first 2000 chars):
{content_sample}

Provide:
1. Recommended chunk size (number)
2. Recommended overlap percentage (0-50)
3. Chunking method (fixed, recursive, semantic, document-aware)
4. Reasoning

Format as JSON:
{{
    "chunk_size": number,
    "overlap_percent": number,
    "method": "string",
    "reasoning": "string"
}}
"""
        )
    
    def analyze_and_chunk(self, text: str, content_type: str = "general") -> List[str]:
        """Agent analyzes content and chunks accordingly."""
        # Get sample for analysis
        sample = text[:2000]
        
        # Get strategy from LLM
        prompt = self.strategy_prompt.format(
            content_sample=sample,
            content_type=content_type
        )
        
        response = self.llm(prompt)
        # Parse JSON response (simplified)
        import json
        strategy = json.loads(response)
        
        # Apply strategy
        chunk_size = strategy["chunk_size"]
        overlap = int(chunk_size * strategy["overlap_percent"] / 100)
        method = strategy["method"]
        
        if method == "fixed":
            chunker = FixedSizeChunker(chunk_size, overlap)
        elif method == "recursive":
            chunker = RecursiveChunker(chunk_size, overlap)
        elif method == "semantic":
            chunker = SemanticChunker(similarity_threshold=0.7)
        else:  # document-aware
            chunker = DocumentAwareChunker(max_chunk_size=chunk_size, overlap=overlap)
        
        return chunker.chunk(text)

# Example
agentic_chunker = AgenticChunker()
chunks = agentic_chunker.analyze_and_chunk(long_document, content_type="technical_documentation")
```

---

## 4. Embedding Models

### 4.1 How Embeddings Work

Embeddings convert text into dense vector representations that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

```
┌─────────────────────────────────────────────────────────┐
│              EMBEDDING PROCESS                           │
└─────────────────────────────────────────────────────────┘

Text Input: "Machine learning is fascinating"
                ↓
        Tokenization
                ↓
    [Token IDs: 1234, 5678, 90, 12345]
                ↓
    Neural Network Processing
    (Transformer Layers)
                ↓
    Dense Vector (1536 dimensions)
    [0.023, -0.145, 0.892, ..., 0.234]
                ↓
    Normalized Embedding Vector
```

**Key Properties:**

1. **Semantic Similarity**: Similar meanings → Similar vectors
2. **Dimensionality**: Typically 384-1536 dimensions
3. **Normalization**: Often L2-normalized for cosine similarity
4. **Context-Aware**: Modern models understand context

### 4.2 OpenAI Embeddings

OpenAI provides high-quality embeddings optimized for various tasks.

```python
from openai import OpenAI
import numpy as np

class OpenAIEmbedder:
    """OpenAI embedding wrapper."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Args:
            model: "text-embedding-3-small" (1536 dims) or 
                   "text-embedding-3-large" (3072 dims) or
                   "text-embedding-ada-002" (1536 dims, legacy)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = self._get_dimensions(model)
    
    def _get_dimensions(self, model: str) -> int:
        """Get embedding dimensions for model."""
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dim_map.get(model, 1536)
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Embed multiple texts with batching."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_with_metadata(self, texts: List[str], 
                           metadata: List[Dict] = None) -> List[Dict]:
        """Embed with metadata preservation."""
        embeddings = self.embed_batch(texts)
        
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            result = {
                "text": text,
                "embedding": embedding,
                "metadata": metadata[i] if metadata else {}
            }
            results.append(result)
        
        return results

# Example Usage
embedder = OpenAIEmbedder(model="text-embedding-3-small")
embedding = embedder.embed("Machine learning algorithms")
print(f"Embedding shape: {embedding.shape}")  # (1536,)
```

**OpenAI Embedding Models Comparison:**

| Model | Dimensions | Cost per 1K tokens | Speed | Best For |
|-------|------------|-------------------|-------|----------|
| text-embedding-3-small | 1536 | $0.02 | Fast | General purpose, cost-effective |
| text-embedding-3-large | 3072 | $0.13 | Medium | High accuracy needs |
| text-embedding-ada-002 | 1536 | $0.10 | Fast | Legacy, still good |

### 4.3 Open-Source Alternatives

Open-source models offer free alternatives with good performance.

**Sentence Transformers:**

```python
from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerEmbedder:
    """Sentence Transformers embedding wrapper."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dims
        - all-mpnet-base-v2: Better quality, 768 dims
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """Embed batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_with_instructions(self, texts: List[str], 
                               instruction: str = "Represent this sentence for searching relevant passages:") -> np.ndarray:
        """Embed with instruction prefix (for some models)."""
        # Some models work better with instructions
        texts_with_instruction = [f"{instruction} {text}" for text in texts]
        return self.embed_batch(texts_with_instruction)

# Example Usage
st_embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")
embeddings = st_embedder.embed_batch(["Text 1", "Text 2", "Text 3"])
```

**BGE (BAAI General Embedding) Models:**

```python
class BGEEmbedder:
    """BGE embedding models - state-of-the-art open source."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        BGE Models:
        - BAAI/bge-small-en-v1.5: 384 dims, fast
        - BAAI/bge-base-en-v1.5: 768 dims, balanced
        - BAAI/bge-large-en-v1.5: 1024 dims, best quality
        - BAAI/bge-m3: Multilingual, 1024 dims
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed query (BGE uses instruction for queries)."""
        instruction = "Represent this sentence for searching relevant passages:"
        text = f"{instruction} {query}"
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed documents (no instruction needed)."""
        return self.model.encode(documents, convert_to_numpy=True)
    
    def embed_hybrid(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed with query/document distinction."""
        if is_query:
            return np.array([self.embed_query(text) for text in texts])
        else:
            return self.embed_documents(texts)

# Example
bge_embedder = BGEEmbedder("BAAI/bge-base-en-v1.5")
query_emb = bge_embedder.embed_query("What is machine learning?")
doc_embs = bge_embedder.embed_documents(["ML is a subset of AI...", "AI involves..."])
```

**E5 Models:**

```python
class E5Embedder:
    """E5 embedding models - instruction-tuned."""
    
    def __init__(self, model_name: str = "intfloat/e5-small-v2"):
        """
        E5 Models:
        - intfloat/e5-small-v2: 384 dims
        - intfloat/e5-base-v2: 768 dims
        - intfloat/e5-large-v2: 1024 dims
        - multilingual-e5-base: Multilingual
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str, task: str = "passage") -> np.ndarray:
        """
        Embed with task prefix.
        
        Tasks:
        - "query": For search queries
        - "passage": For documents/passages
        - "title": For titles
        """
        task_prefixes = {
            "query": "query: ",
            "passage": "passage: ",
            "title": "title: "
        }
        prefix = task_prefixes.get(task, "")
        text_with_prefix = f"{prefix}{text}"
        return self.model.encode(text_with_prefix, convert_to_numpy=True)

# Example
e5_embedder = E5Embedder("intfloat/e5-base-v2")
query_emb = e5_embedder.embed("machine learning", task="query")
doc_emb = e5_embedder.embed("Machine learning is a method...", task="passage")
```

### 4.4 Embedding Model Comparison Table

| Model | Provider | Dimensions | Speed | Quality | Cost | Best Use Case |
|-------|----------|------------|-------|---------|------|---------------|
| text-embedding-3-small | OpenAI | 1536 | Fast | High | $0.02/1K | Production, general purpose |
| text-embedding-3-large | OpenAI | 3072 | Medium | Very High | $0.13/1K | High accuracy needs |
| all-MiniLM-L6-v2 | Sentence Transformers | 384 | Very Fast | Good | Free | Fast prototyping |
| all-mpnet-base-v2 | Sentence Transformers | 768 | Medium | High | Free | Balanced quality/speed |
| BAAI/bge-large-en-v1.5 | BGE | 1024 | Medium | Very High | Free | Best open-source quality |
| intfloat/e5-large-v2 | E5 | 1024 | Medium | Very High | Free | Instruction-tuned tasks |
| multilingual-e5-base | E5 | 768 | Medium | High | Free | Multilingual applications |

### 4.5 Dimensionality and Performance Trade-offs

```python
import time
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingBenchmark:
    """Benchmark embedding models."""
    
    def __init__(self):
        self.test_texts = [
            "Machine learning is a subset of artificial intelligence",
            "AI involves creating intelligent machines",
            "Deep learning uses neural networks",
            "The weather today is sunny and warm",
            "Cooking recipes require precise measurements"
        ]
    
    def benchmark_model(self, embedder, name: str) -> Dict:
        """Benchmark a single model."""
        print(f"Benchmarking {name}...")
        
        # Time embedding
        start = time.time()
        embeddings = embedder.embed_batch(self.test_texts)
        embed_time = time.time() - start
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Expected: texts 0-2 should be similar (AI/ML related)
        # Texts 3-4 should be different
        ml_similarity = np.mean([
            similarity_matrix[0][1],
            similarity_matrix[0][2],
            similarity_matrix[1][2]
        ])
        
        cross_domain_similarity = np.mean([
            similarity_matrix[0][3],
            similarity_matrix[0][4],
            similarity_matrix[1][3]
        ])
        
        return {
            "name": name,
            "dimensions": embeddings.shape[1],
            "embed_time": embed_time,
            "ml_similarity": ml_similarity,
            "cross_domain_similarity": cross_domain_similarity,
            "discrimination_score": ml_similarity - cross_domain_similarity
        }
    
    def compare_models(self, embedders: List[tuple]) -> pd.DataFrame:
        """Compare multiple models."""
        results = []
        for embedder, name in embedders:
            result = self.benchmark_model(embedder, name)
            results.append(result)
        
        return pd.DataFrame(results)

# Example Usage
benchmark = EmbeddingBenchmark()
results = benchmark.compare_models([
    (OpenAIEmbedder("text-embedding-3-small"), "OpenAI Small"),
    (SentenceTransformerEmbedder("all-MiniLM-L6-v2"), "MiniLM"),
    (BGEEmbedder("BAAI/bge-base-en-v1.5"), "BGE Base")
])
print(results)
```

**Dimensionality Guidelines:**

- **< 400 dims**: Fast, good for simple tasks, limited semantic capture
- **400-800 dims**: Balanced, good for most RAG applications
- **800-1500 dims**: High quality, better for complex domains
- **> 1500 dims**: Diminishing returns, higher storage/compute costs

---

## 5. Vector Databases

### 5.1 ChromaDB

ChromaDB is a lightweight, embeddable vector database perfect for development and small-scale production.

```python
import chromadb
from chromadb.config import Settings

class ChromaDBStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(self, collection_name: str = "documents", 
                 persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        self.collection_name = collection_name
    
    def add_documents(self, texts: List[str], 
                     embeddings: List[List[float]],
                     metadatas: List[Dict] = None,
                     ids: List[str] = None):
        """Add documents to collection."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], 
              n_results: int = 5,
              where: Dict = None,
              where_document: Dict = None) -> Dict:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"category": "technical"})
            where_document: Document content filter (e.g., {"$contains": "Python"})
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def update_document(self, doc_id: str, 
                       text: str = None,
                       embedding: List[float] = None,
                       metadata: Dict = None):
        """Update a document."""
        update_data = {}
        if text:
            update_data["documents"] = text
        if embedding:
            update_data["embeddings"] = embedding
        if metadata:
            update_data["metadatas"] = metadata
        
        self.collection.update(
            ids=[doc_id],
            **update_data
        )
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count
        }

# Example Usage
chroma_store = ChromaDBStore(collection_name="knowledge_base")

# Add documents
texts = ["Document 1 text", "Document 2 text"]
embeddings = [[0.1] * 1536, [0.2] * 1536]  # Example embeddings
metadatas = [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
chroma_store.add_documents(texts, embeddings, metadatas)

# Search
query_emb = [0.15] * 1536
results = chroma_store.search(query_emb, n_results=3)
print(f"Found {len(results['documents'])} results")
```

**ChromaDB Features:**

- **Embeddable**: Runs in-process, no separate server
- **Persistent**: Can save to disk
- **Metadata Filtering**: Filter by metadata fields
- **Multiple Distance Metrics**: Cosine, L2, IP
- **Simple API**: Easy to use

**ChromaDB Limitations:**

- **Scale**: Best for < 1M vectors
- **Concurrency**: Limited concurrent writes
- **Advanced Features**: Fewer features than cloud solutions

### 5.2 Pinecone

Pinecone is a managed vector database service with excellent scalability.

```python
from pinecone import Pinecone, ServerlessSpec
import time

class PineconeStore:
    """Pinecone vector store wrapper."""
    
    def __init__(self, api_key: str, index_name: str = "rag-index",
                 dimension: int = 1536, metric: str = "cosine"):
        """
        Initialize Pinecone client and index.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the index
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(index_name)
    
    def upsert_vectors(self, vectors: List[Dict]):
        """
        Upsert vectors to index.
        
        Args:
            vectors: List of dicts with 'id', 'values', 'metadata'
        """
        self.index.upsert(vectors=vectors)
    
    def add_documents(self, texts: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict] = None,
                     ids: List[str] = None):
        """Add documents with embeddings."""
        if ids is None:
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        vectors = [
            {
                "id": doc_id,
                "values": embedding,
                "metadata": {**metadata, "text": text}
            }
            for doc_id, embedding, text, metadata in zip(ids, embeddings, texts, metadatas)
        ]
        
        self.upsert_vectors(vectors)
    
    def search(self, query_embedding: List[float],
              top_k: int = 5,
              filter: Dict = None,
              include_metadata: bool = True) -> Dict:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter: Metadata filter (e.g., {"category": {"$eq": "technical"}})
            include_metadata: Whether to include metadata
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )
        
        return {
            "matches": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        }
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by IDs."""
        self.index.delete(ids=ids)
    
    def delete_by_filter(self, filter: Dict):
        """Delete vectors matching filter."""
        self.index.delete(filter=filter)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }

# Example Usage
pinecone_store = PineconeStore(
    api_key="your-api-key",
    index_name="my-rag-index",
    dimension=1536
)

# Add documents
pinecone_store.add_documents(texts, embeddings, metadatas)

# Search
results = pinecone_store.search(query_embedding, top_k=5)
```

**Pinecone Features:**

- **Managed Service**: No infrastructure management
- **Scalability**: Handles millions of vectors
- **Metadata Filtering**: Complex filter expressions
- **Real-time Updates**: Low latency updates
- **Multiple Regions**: Global deployment

**Pinecone Pricing:**

- **Starter**: Free tier (1 index, 100K vectors)
- **Standard**: $70/month (1M vectors)
- **Enterprise**: Custom pricing

### 5.3 Weaviate Overview

Weaviate is an open-source vector database with GraphQL API.

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

class WeaviateStore:
    """Weaviate vector store wrapper."""
    
    def __init__(self, url: str = "http://localhost:8080",
                 api_key: str = None):
        """Initialize Weaviate client."""
        auth = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(url=url, auth_client_secret=auth)
    
    def create_schema(self, class_name: str = "Document"):
        """Create Weaviate schema."""
        schema = {
            "class": class_name,
            "description": "Document for RAG",
            "vectorizer": "none",  # We provide vectors
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Document text content"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Document source"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Chunk index in document"
                }
            ]
        }
        
        if not self.client.schema.exists(class_name):
            self.client.schema.create_class(schema)
    
    def add_documents(self, texts: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict] = None):
        """Add documents to Weaviate."""
        with self.client.batch as batch:
            batch.batch_size = 100
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                properties = {
                    "text": text,
                    **(metadatas[i] if metadatas else {})
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name="Document",
                    vector=embedding
                )
    
    def search(self, query_embedding: List[float],
              limit: int = 5,
              where: Dict = None) -> List[Dict]:
        """Search Weaviate."""
        query = (
            self.client.query
            .get("Document", ["text", "source", "chunk_index"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
        )
        
        if where:
            query = query.with_where(where)
        
        results = query.do()
        return results["data"]["Get"]["Document"]

# Example
weaviate_store = WeaviateStore()
weaviate_store.create_schema()
weaviate_store.add_documents(texts, embeddings)
results = weaviate_store.search(query_embedding)
```

### 5.4 Qdrant Overview

Qdrant is a high-performance vector database written in Rust.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantStore:
    """Qdrant vector store wrapper."""
    
    def __init__(self, url: str = "localhost", port: int = 6333):
        """Initialize Qdrant client."""
        self.client = QdrantClient(host=url, port=port)
        self.collection_name = "documents"
    
    def create_collection(self, dimension: int = 1536):
        """Create Qdrant collection."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
    
    def add_documents(self, texts: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict] = None,
                     ids: List[int] = None):
        """Add documents to Qdrant."""
        if ids is None:
            ids = list(range(len(texts)))
        
        points = [
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": text, **(metadata or {})}
            )
            for doc_id, embedding, text, metadata in zip(
                ids, embeddings, texts, metadatas or [{}] * len(texts)
            )
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_embedding: List[float],
              limit: int = 5,
              filter: Dict = None) -> List[Dict]:
        """Search Qdrant."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
```

### 5.5 Milvus Overview

Milvus is a cloud-native vector database built for scale.

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusStore:
    """Milvus vector store wrapper."""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        """Connect to Milvus."""
        connections.connect("default", host=host, port=port)
        self.collection_name = "documents"
    
    def create_collection(self, dimension: int = 1536):
        """Create Milvus collection."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        schema = CollectionSchema(fields, "Document collection for RAG")
        collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        """Add documents to Milvus."""
        collection = Collection(self.collection_name)
        entities = [
            texts,
            embeddings
        ]
        collection.insert(entities)
        collection.load()
    
    def search(self, query_embedding: List[float], top_k: int = 5):
        """Search Milvus."""
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        return results
```

### 5.6 FAISS for Local Development

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.

```python
import faiss
import numpy as np

class FAISSStore:
    """FAISS vector store for local development."""
    
    def __init__(self, dimension: int = 1536, metric: str = "cosine"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension
            metric: "cosine" or "L2"
        """
        self.dimension = dimension
        
        if metric == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = True
        else:  # L2
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        
        self.texts = []
        self.metadatas = []
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add_vectors(self, embeddings: np.ndarray,
                   texts: List[str],
                   metadatas: List[Dict] = None):
        """Add vectors to index."""
        embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.normalize:
            embeddings = self.normalize_vectors(embeddings)
        
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{}] * len(texts))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        if self.normalize:
            query_embedding = self.normalize_vectors(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(dist),
                    "score": float(dist) if self.normalize else 1.0 / (1.0 + dist)
                })
        
        return results
    
    def save(self, filepath: str):
        """Save index to disk."""
        faiss.write_index(self.index, filepath)
        # Save texts and metadata separately
        import pickle
        with open(filepath + ".meta", "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)
    
    def load(self, filepath: str):
        """Load index from disk."""
        self.index = faiss.read_index(filepath)
        import pickle
        with open(filepath + ".meta", "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]

# Example Usage
faiss_store = FAISSStore(dimension=1536, metric="cosine")
faiss_store.add_vectors(embeddings, texts, metadatas)
results = faiss_store.search(query_embedding, k=5)
faiss_store.save("./faiss_index")
```

### 5.7 Vector Database Comparison Table

| Database | Type | Scalability | Speed | Metadata Filtering | Cost | Best For |
|----------|------|-------------|-------|-------------------|------|----------|
| ChromaDB | Open-source | < 1M vectors | Fast | Basic | Free | Development, small projects |
| Pinecone | Managed | Millions | Very Fast | Advanced | $70+/month | Production, scale |
| Weaviate | Open-source/Cloud | Millions | Fast | GraphQL filters | Free/Paid | Graph + Vector search |
| Qdrant | Open-source/Cloud | Millions | Very Fast | Advanced | Free/Paid | High performance |
| Milvus | Open-source/Cloud | Billions | Fast | Advanced | Free/Paid | Large scale |
| FAISS | Library | Millions | Very Fast | None (manual) | Free | Research, local dev |

### 5.8 Indexing Strategies

**HNSW (Hierarchical Navigable Small World):**

```python
# HNSW index in FAISS
dimension = 1536
M = 32  # Number of connections
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = 200  # Construction time/quality trade-off
index.hnsw.efSearch = 50  # Search time/quality trade-off
```

**IVF (Inverted File Index):**

```python
# IVF index in FAISS
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.nprobe = 10  # Number of clusters to search
```

**Product Quantization (PQ):**

```python
# PQ for compression
m = 64  # Number of sub-vectors
bits = 8  # Bits per sub-vector
index = faiss.IndexPQ(dimension, m, bits)
```

**Combined Index (IVF + PQ + HNSW):**

```python
# Best of all worlds
quantizer = faiss.IndexHNSWFlat(dimension, M)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
```

---

## 6. Retrieval Methods

### 6.1 Dense Retrieval (Semantic Search)

Dense retrieval uses embeddings to find semantically similar documents.

```python
from sklearn.metrics.pairwise import cosine_similarity

class DenseRetriever:
    """Dense retrieval using embeddings."""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        """
        Args:
            embeddings: Document embeddings matrix (n_docs x dim)
            texts: List of document texts
        """
        self.embeddings = embeddings
        self.texts = texts
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve top-k similar documents."""
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.texts[idx],
                "score": float(similarities[idx]),
                "index": int(idx)
            })
        
        return results

# Example
retriever = DenseRetriever(document_embeddings, document_texts)
query_emb = embedder.embed("What is machine learning?")
results = retriever.retrieve(query_emb, k=5)
```

### 6.2 Sparse Retrieval (BM25, TF-IDF)

Sparse retrieval uses keyword matching and term frequency.

```python
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

class SparseRetriever:
    """Sparse retrieval using BM25 or TF-IDF."""
    
    def __init__(self, texts: List[str], method: str = "bm25"):
        """
        Args:
            texts: List of document texts
            method: "bm25" or "tfidf"
        """
        self.texts = texts
        self.method = method
        
        if method == "bm25":
            tokenized_texts = [text.lower().split() for text in texts]
            self.retriever = BM25Okapi(tokenized_texts)
        else:  # tfidf
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k documents."""
        if self.method == "bm25":
            tokenized_query = query.lower().split()
            scores = self.retriever.get_scores(tokenized_query)
        else:  # tfidf
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.texts[idx],
                "score": float(scores[idx]),
                "index": int(idx)
            })
        
        return results

# Example
sparse_retriever = SparseRetriever(document_texts, method="bm25")
results = sparse_retriever.retrieve("machine learning algorithms", k=5)
```

### 6.3 Hybrid Retrieval

Combine dense and sparse retrieval for better results.

```python
class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""
    
    def __init__(self, dense_retriever: DenseRetriever,
                 sparse_retriever: SparseRetriever,
                 dense_weight: float = 0.7):
        """
        Args:
            dense_retriever: Dense retrieval instance
            sparse_retriever: Sparse retrieval instance
            dense_weight: Weight for dense scores (0-1)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
    
    def retrieve(self, query: str, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Hybrid retrieval."""
        # Get results from both methods
        dense_results = self.dense_retriever.retrieve(query_embedding, k=k*2)
        sparse_results = self.sparse_retriever.retrieve(query, k=k*2)
        
        # Normalize scores to [0, 1]
        dense_scores = {r["index"]: r["score"] for r in dense_results}
        sparse_scores = {r["index"]: r["score"] for r in sparse_results}
        
        # Normalize
        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0
        
        # Combine scores
        combined_scores = {}
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0) / max_dense if max_dense > 0 else 0
            sparse_score = sparse_scores.get(idx, 0) / max_sparse if max_sparse > 0 else 0
            
            combined_score = (self.dense_weight * dense_score + 
                            self.sparse_weight * sparse_score)
            combined_scores[idx] = combined_score
        
        # Get top-k
        top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in top_indices:
            # Get text from either retriever
            text = self.dense_retriever.texts[idx] if idx < len(self.dense_retriever.texts) else self.sparse_retriever.texts[idx]
            results.append({
                "text": text,
                "score": score,
                "index": idx,
                "dense_score": dense_scores.get(idx, 0),
                "sparse_score": sparse_scores.get(idx, 0)
            })
        
        return results

# Example
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, dense_weight=0.7)
results = hybrid_retriever.retrieve("machine learning", query_embedding, k=5)
```

### 6.4 Re-ranking

Re-rank initial retrieval results using more sophisticated models.

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Re-rank retrieval results using cross-encoder."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Cross-encoder model for re-ranking
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Re-rank documents for query."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                "text": doc,
                "score": float(score)
            })
        
        return results

# Example Usage
reranker = Reranker()
initial_results = retriever.retrieve(query, k=20)  # Get more candidates
documents = [r["text"] for r in initial_results]
reranked = reranker.rerank(query, documents, top_k=5)
```

**Cohere Rerank API:**

```python
import cohere

class CohereReranker:
    """Re-rank using Cohere API."""
    
    def __init__(self, api_key: str):
        self.co = cohere.Client(api_key)
    
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        """Re-rank using Cohere."""
        results = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        return [
            {
                "text": results.results[i].document["text"],
                "score": results.results[i].relevance_score,
                "index": results.results[i].index
            }
            for i in range(len(results.results))
        ]
```

### 6.5 Maximal Marginal Relevance (MMR)

MMR balances relevance and diversity in retrieval.

```python
class MMRRetriever:
    """MMR retrieval for diverse results."""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str], lambda_param: float = 0.5):
        """
        Args:
            embeddings: Document embeddings
            texts: Document texts
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.embeddings = embeddings
        self.texts = texts
        self.lambda_param = lambda_param
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """MMR retrieval."""
        # Calculate similarities with query
        query_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        selected_indices = []
        remaining_indices = list(range(len(self.texts)))
        
        # Select first document (most relevant)
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        for _ in range(min(k - 1, len(remaining_indices))):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Max similarity to already selected
                if selected_indices:
                    max_sim = max([
                        cosine_similarity(
                            [self.embeddings[idx]],
                            [self.embeddings[sel_idx]]
                        )[0][0]
                        for sel_idx in selected_indices
                    ])
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return results
        results = []
        for idx in selected_indices:
            results.append({
                "text": self.texts[idx],
                "score": float(query_similarities[idx]),
                "index": idx
            })
        
        return results

# Example
mmr_retriever = MMRRetriever(document_embeddings, document_texts, lambda_param=0.7)
results = mmr_retriever.retrieve(query_embedding, k=5)
```

### 6.6 Metadata Filtering

Filter retrieval results using metadata.

```python
class MetadataFilteredRetriever:
    """Retriever with metadata filtering."""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict]):
        """
        Args:
            embeddings: Document embeddings
            texts: Document texts
            metadatas: List of metadata dicts for each document
        """
        self.embeddings = embeddings
        self.texts = texts
        self.metadatas = metadatas
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5,
                filters: Dict = None) -> List[Dict]:
        """
        Retrieve with metadata filters.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Dict of filters, e.g., {"category": "technical", "year": {"$gte": 2020}}
        """
        # Filter documents
        filtered_indices = list(range(len(self.texts)))
        
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle operators
                    if "$eq" in value:
                        filtered_indices = [
                            i for i in filtered_indices
                            if self.metadatas[i].get(key) == value["$eq"]
                        ]
                    elif "$gte" in value:
                        filtered_indices = [
                            i for i in filtered_indices
                            if self.metadatas[i].get(key, 0) >= value["$gte"]
                        ]
                    elif "$in" in value:
                        filtered_indices = [
                            i for i in filtered_indices
                            if self.metadatas[i].get(key) in value["$in"]
                        ]
                else:
                    # Simple equality
                    filtered_indices = [
                        i for i in filtered_indices
                        if self.metadatas[i].get(key) == value
                    ]
        
        # Retrieve from filtered set
        if not filtered_indices:
            return []
        
        filtered_embeddings = self.embeddings[filtered_indices]
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        top_k_local = np.argsort(similarities)[-k:][::-1]
        top_k_global = [filtered_indices[i] for i in top_k_local]
        
        results = []
        for idx in top_k_global:
            results.append({
                "text": self.texts[idx],
                "score": float(similarities[filtered_indices.index(idx)]),
                "metadata": self.metadatas[idx],
                "index": idx
            })
        
        return results

# Example
filtered_retriever = MetadataFilteredRetriever(embeddings, texts, metadatas)
results = filtered_retriever.retrieve(
    query_embedding,
    k=5,
    filters={"category": "technical", "year": {"$gte": 2020}}
)
```

### 6.7 Multi-Query Retrieval

Generate multiple query variations and combine results.

```python
from langchain.llms import OpenAI

class MultiQueryRetriever:
    """Generate multiple queries and combine results."""
    
    def __init__(self, retriever, llm=None):
        """
        Args:
            retriever: Base retriever instance
            llm: LLM for query generation
        """
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
    
    def generate_queries(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Generate query variations."""
        prompt = f"""Generate {n_queries} different ways to ask the following question.
Each query should be semantically similar but use different wording.

Original query: {original_query}

Generated queries (one per line):"""
        
        response = self.llm(prompt)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        queries = queries[:n_queries]
        
        # Always include original query
        if original_query not in queries:
            queries.insert(0, original_query)
        
        return queries
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve using multiple queries."""
        # Generate query variations
        queries = self.generate_queries(query, n_queries=3)
        
        # Retrieve for each query
        all_results = {}
        for q in queries:
            query_embedding = embedder.embed(q)
            results = self.retriever.retrieve(query_embedding, k=k*2)
            
            for result in results:
                doc_idx = result["index"]
                if doc_idx not in all_results:
                    all_results[doc_idx] = {
                        "text": result["text"],
                        "scores": [],
                        "index": doc_idx
                    }
                all_results[doc_idx]["scores"].append(result["score"])
        
        # Combine scores (average or max)
        for doc_idx in all_results:
            all_results[doc_idx]["score"] = np.mean(all_results[doc_idx]["scores"])
        
        # Sort and return top-k
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:k]

# Example
multi_query_retriever = MultiQueryRetriever(base_retriever)
results = multi_query_retriever.retrieve("What is machine learning?", k=5)
```

---

## 7. Advanced RAG Patterns

### 7.1 Corrective RAG (CRAG)

CRAG evaluates retrieval quality and corrects poor retrievals.

```python
from langchain.llms import OpenAI

class CorrectiveRAG:
    """Corrective RAG with retrieval quality assessment."""
    
    def __init__(self, retriever, llm=None):
        """
        Args:
            retriever: Base retriever
            llm: LLM for evaluation and correction
        """
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str]) -> Dict:
        """Evaluate quality of retrieved documents."""
        context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""Evaluate whether the following retrieved documents are relevant to the query.

Query: {query}

Retrieved Documents:
{context}

Rate the relevance on a scale of 0-1, where:
- 1.0: All documents are highly relevant
- 0.5: Some documents are relevant
- 0.0: No documents are relevant

Provide:
1. Relevance score (0-1)
2. Brief explanation
3. List of irrelevant document indices (if any)

Format as JSON:
{{
    "score": 0.0-1.0,
    "explanation": "string",
    "irrelevant_indices": [0, 1, ...]
}}"""
        
        response = self.llm(prompt)
        import json
        evaluation = json.loads(response)
        return evaluation
    
    def correct_retrieval(self, query: str, initial_docs: List[str], 
                         evaluation: Dict) -> List[str]:
        """Correct retrieval based on evaluation."""
        if evaluation["score"] >= 0.7:
            # Good retrieval, return as-is
            return initial_docs
        
        # Remove irrelevant documents
        relevant_docs = [
            doc for i, doc in enumerate(initial_docs)
            if i not in evaluation.get("irrelevant_indices", [])
        ]
        
        # If we removed too many, do web search or expand query
        if len(relevant_docs) < 2:
            # Generate alternative query
            correction_prompt = f"""The initial retrieval for this query was poor.

Query: {query}
Retrieved: {initial_docs[0][:200]}...

Generate an alternative search query that might retrieve better results:"""
            
            alternative_query = self.llm(correction_prompt).strip()
            # Re-retrieve with alternative query
            query_embedding = embedder.embed(alternative_query)
            new_results = self.retriever.retrieve(query_embedding, k=5)
            relevant_docs = [r["text"] for r in new_results]
        
        return relevant_docs
    
    def query(self, query: str) -> Dict:
        """Complete CRAG query pipeline."""
        # Initial retrieval
        query_embedding = embedder.embed(query)
        initial_results = self.retriever.retrieve(query_embedding, k=5)
        initial_docs = [r["text"] for r in initial_results]
        
        # Evaluate retrieval
        evaluation = self.evaluate_retrieval(query, initial_docs)
        
        # Correct if needed
        corrected_docs = self.correct_retrieval(query, initial_docs, evaluation)
        
        # Generate response
        context = "\n\n".join(corrected_docs)
        final_prompt = f"""Answer the question using the following context.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm(final_prompt)
        
        return {
            "query": query,
            "answer": answer,
            "retrieval_score": evaluation["score"],
            "correction_applied": evaluation["score"] < 0.7,
            "sources": corrected_docs
        }

# Example Usage
crag = CorrectiveRAG(retriever)
result = crag.query("What is machine learning?")
```

### 7.2 Self-RAG (Self-Reflective Retrieval)

Self-RAG uses the LLM to decide when to retrieve and how to use retrieved information.

```python
class SelfRAG:
    """Self-RAG with reflection and adaptive retrieval."""
    
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
    
    def should_retrieve(self, query: str, context: str = "") -> Dict:
        """Decide if retrieval is needed."""
        prompt = f"""Given this query and current context, decide if retrieval is needed.

Query: {query}
Current Context: {context if context else "None"}

Decide:
1. Is retrieval needed? (yes/no)
2. What information should be retrieved?
3. Confidence in current knowledge (0-1)

Format as JSON:
{{
    "retrieve": true/false,
    "retrieval_focus": "string",
    "confidence": 0.0-1.0
}}"""
        
        response = self.llm(prompt)
        import json
        return json.loads(response)
    
    def reflect_on_retrieval(self, query: str, retrieved_docs: List[str]) -> Dict:
        """Reflect on retrieved documents."""
        context = "\n\n".join([f"Doc {i+1}: {doc[:300]}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""Evaluate the retrieved documents for answering this query.

Query: {query}

Retrieved Documents:
{context}

Provide:
1. Relevance score (0-1)
2. Which documents are most useful
3. What information is missing (if any)

Format as JSON:
{{
    "relevance": 0.0-1.0,
    "useful_docs": [0, 1, ...],
    "missing_info": "string or null"
}}"""
        
        response = self.llm(prompt)
        import json
        return json.loads(response)
    
    def generate_with_reflection(self, query: str, retrieved_docs: List[str],
                                reflection: Dict) -> str:
        """Generate answer with reflection."""
        # Use only useful documents
        useful_docs = [retrieved_docs[i] for i in reflection["useful_docs"]]
        context = "\n\n".join(useful_docs)
        
        prompt = f"""Answer the question using the retrieved context. 
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Reflection: Relevance = {reflection['relevance']}, Missing: {reflection['missing_info']}

Answer:"""
        
        answer = self.llm(prompt)
        return answer
    
    def query(self, query: str, max_iterations: int = 3) -> Dict:
        """Self-RAG query with iterative refinement."""
        context = ""
        all_retrieved = []
        
        for iteration in range(max_iterations):
            # Decide if retrieval needed
            decision = self.should_retrieve(query, context)
            
            if not decision["retrieve"]:
                break
            
            # Retrieve
            if decision["retrieval_focus"]:
                retrieval_query = f"{query} {decision['retrieval_focus']}"
            else:
                retrieval_query = query
            
            query_embedding = embedder.embed(retrieval_query)
            results = self.retriever.retrieve(query_embedding, k=5)
            retrieved_docs = [r["text"] for r in results]
            all_retrieved.extend(retrieved_docs)
            
            # Reflect
            reflection = self.reflect_on_retrieval(query, retrieved_docs)
            
            # Generate with reflection
            answer = self.generate_with_reflection(query, retrieved_docs, reflection)
            context = answer
            
            # If relevance is high and no missing info, stop
            if reflection["relevance"] > 0.8 and not reflection["missing_info"]:
                break
        
        return {
            "query": query,
            "answer": context,
            "iterations": iteration + 1,
            "sources": list(set(all_retrieved))
        }

# Example
self_rag = SelfRAG(retriever)
result = self_rag.query("Explain machine learning in detail")
```

### 7.3 Agentic RAG

Agent-driven RAG where an agent orchestrates retrieval and generation.

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

class AgenticRAG:
    """Agent-driven RAG system."""
    
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
        self.setup_agent()
    
    def setup_agent(self):
        """Setup agent with retrieval tools."""
        def retrieve_tool(query: str) -> str:
            """Retrieve relevant documents."""
            query_embedding = embedder.embed(query)
            results = self.retriever.retrieve(query_embedding, k=5)
            return "\n\n".join([r["text"] for r in results])
        
        def search_specific_topic(topic: str) -> str:
            """Search for specific topic."""
            query_embedding = embedder.embed(f"information about {topic}")
            results = self.retriever.retrieve(query_embedding, k=3)
            return "\n\n".join([r["text"] for r in results])
        
        tools = [
            Tool(
                name="RetrieveDocuments",
                func=retrieve_tool,
                description="Retrieve relevant documents for a query"
            ),
            Tool(
                name="SearchTopic",
                func=search_specific_topic,
                description="Search for information about a specific topic"
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )
    
    def query(self, query: str) -> str:
        """Query using agent."""
        response = self.agent.run(f"""Answer this question using the available tools: {query}
        
Use RetrieveDocuments to get relevant context, then formulate your answer.""")
        return response

# Example
agentic_rag = AgenticRAG(retriever)
answer = agentic_rag.query("What are the main types of machine learning?")
```

### 7.4 Hypothetical Document Embeddings (HyDE)

HyDE generates a hypothetical answer first, then retrieves based on that.

```python
class HyDERAG:
    """Hypothetical Document Embeddings RAG."""
    
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate hypothetical answer to query."""
        prompt = f"""Generate a hypothetical answer to this question. 
This answer will be used to find similar documents, so make it detailed and informative.

Question: {query}

Hypothetical Answer:"""
        
        hypothetical = self.llm(prompt)
        return hypothetical
    
    def query(self, query: str) -> Dict:
        """HyDE query pipeline."""
        # Generate hypothetical answer
        hypothetical = self.generate_hypothetical_answer(query)
        
        # Retrieve using hypothetical answer
        hypothetical_embedding = embedder.embed(hypothetical)
        results = self.retriever.retrieve(hypothetical_embedding, k=5)
        retrieved_docs = [r["text"] for r in results]
        
        # Generate final answer using retrieved docs
        context = "\n\n".join(retrieved_docs)
        final_prompt = f"""Answer the question using the following context.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm(final_prompt)
        
        return {
            "query": query,
            "hypothetical_answer": hypothetical,
            "answer": answer,
            "sources": retrieved_docs
        }

# Example
hyde_rag = HyDERAG(retriever)
result = hyde_rag.query("How does neural network training work?")
```

### 7.5 Parent-Child Document Retrieval

Store documents in hierarchical structure for better retrieval.

```python
class ParentChildRetrieval:
    """Parent-child document retrieval."""
    
    def __init__(self):
        self.parent_docs = {}  # parent_id -> full document
        self.child_chunks = []  # List of child chunks with parent reference
    
    def index_document(self, parent_id: str, full_doc: str, chunks: List[str]):
        """Index document with parent-child structure."""
        self.parent_docs[parent_id] = full_doc
        
        for i, chunk in enumerate(chunks):
            self.child_chunks.append({
                "parent_id": parent_id,
                "chunk_index": i,
                "content": chunk,
                "embedding": embedder.embed(chunk)
            })
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve child chunks, return parent documents."""
        # Find similar child chunks
        child_embeddings = np.array([c["embedding"] for c in self.child_chunks])
        similarities = cosine_similarity([query_embedding], child_embeddings)[0]
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Get unique parent documents
        parent_ids = set()
        results = []
        
        for idx in top_k_indices:
            child = self.child_chunks[idx]
            parent_id = child["parent_id"]
            
            if parent_id not in parent_ids:
                parent_ids.add(parent_id)
                results.append({
                    "parent_id": parent_id,
                    "full_document": self.parent_docs[parent_id],
                    "matching_chunk": child["content"],
                    "similarity": float(similarities[idx])
                })
        
        return results

# Example
pc_retrieval = ParentChildRetrieval()
pc_retrieval.index_document("doc1", full_document, chunks)
results = pc_retrieval.retrieve(query_embedding, k=3)
```

### 7.6 Multi-Step Retrieval

Iterative retrieval with query refinement.

```python
class MultiStepRetrieval:
    """Multi-step retrieval with query refinement."""
    
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or OpenAI(temperature=0)
    
    def refine_query(self, original_query: str, retrieved_docs: List[str],
                    step: int) -> str:
        """Refine query based on retrieved documents."""
        context_summary = "\n".join([doc[:200] for doc in retrieved_docs])
        
        prompt = f"""Based on the initial query and retrieved information, refine the query for better retrieval.

Original Query: {original_query}

Retrieved Information (summary):
{context_summary}

Step: {step}

Generate a refined query that:
1. Focuses on missing information
2. Uses more specific terminology from the domain
3. Targets gaps in current retrieval

Refined Query:"""
        
        refined = self.llm(prompt).strip()
        return refined
    
    def query(self, query: str, steps: int = 3) -> Dict:
        """Multi-step retrieval."""
        all_retrieved = []
        current_query = query
        
        for step in range(steps):
            # Retrieve
            query_embedding = embedder.embed(current_query)
            results = self.retriever.retrieve(query_embedding, k=5)
            retrieved_docs = [r["text"] for r in results]
            all_retrieved.extend(retrieved_docs)
            
            # Refine query for next step
            if step < steps - 1:
                current_query = self.refine_query(query, retrieved_docs, step + 1)
        
        # Deduplicate
        unique_docs = []
        seen = set()
        for doc in all_retrieved:
            doc_hash = hash(doc[:100])
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
        
        return {
            "query": query,
            "retrieved_documents": unique_docs,
            "steps": steps
        }

# Example
multi_step = MultiStepRetrieval(retriever)
results = multi_step.query("Explain deep learning architectures", steps=3)
```

### 7.7 Graph RAG

Combine knowledge graphs with vector search.

```python
import networkx as nx

class GraphRAG:
    """Graph RAG combining knowledge graphs and vector search."""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.graph = nx.DiGraph()  # Knowledge graph
        self.node_embeddings = {}  # node_id -> embedding
    
    def build_graph_from_documents(self, documents: List[str]):
        """Build knowledge graph from documents."""
        from langchain.llms import OpenAI
        llm = OpenAI(temperature=0)
        
        for i, doc in enumerate(documents):
            # Extract entities and relations (simplified)
            prompt = f"""Extract entities and relationships from this text.

Text: {doc[:1000]}

Format as JSON:
{{
    "entities": ["entity1", "entity2", ...],
    "relations": [
        {{"source": "entity1", "relation": "relates_to", "target": "entity2"}},
        ...
    ]
}}"""
            
            response = llm(prompt)
            import json
            data = json.loads(response)
            
            # Add to graph
            for entity in data["entities"]:
                if entity not in self.graph:
                    self.graph.add_node(entity, doc_id=i)
                    self.node_embeddings[entity] = embedder.embed(entity)
            
            for rel in data["relations"]:
                self.graph.add_edge(
                    rel["source"],
                    rel["target"],
                    relation=rel["relation"]
                )
    
    def retrieve_with_graph(self, query: str, k: int = 5) -> Dict:
        """Retrieve using both vector search and graph traversal."""
        # Vector retrieval
        query_embedding = embedder.embed(query)
        vector_results = self.retriever.retrieve(query_embedding, k=k)
        
        # Find relevant entities in graph
        entity_similarities = {}
        for entity, emb in self.node_embeddings.items():
            sim = cosine_similarity([query_embedding], [emb])[0][0]
            entity_similarities[entity] = sim
        
        top_entities = sorted(entity_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get neighbors of top entities
        graph_docs = []
        for entity, score in top_entities:
            # Get connected entities
            neighbors = list(self.graph.neighbors(entity))
            for neighbor in neighbors[:3]:
                # Get documents containing these entities
                if neighbor in self.graph.nodes:
                    doc_id = self.graph.nodes[neighbor].get("doc_id")
                    if doc_id is not None:
                        graph_docs.append({
                            "entity": neighbor,
                            "score": score,
                            "doc_id": doc_id
                        })
        
        return {
            "vector_results": vector_results,
            "graph_results": graph_docs,
            "top_entities": [e[0] for e in top_entities]
        }

# Example
graph_rag = GraphRAG(retriever)
graph_rag.build_graph_from_documents(documents)
results = graph_rag.retrieve_with_graph("machine learning algorithms")
```

---

## 8. Knowledge Graphs

### 8.1 Graph Construction from Documents

Build knowledge graphs by extracting entities and relationships.

```python
import spacy
from spacy import displacy
import networkx as nx

class KnowledgeGraphBuilder:
    """Build knowledge graphs from documents."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = nx.DiGraph()
    
    def extract_entities_relations(self, text: str) -> Dict:
        """Extract entities and relations using spaCy."""
        doc = self.nlp(text)
        
        entities = {}
        relations = []
        
        # Extract named entities
        for ent in doc.ents:
            entities[ent.text] = {
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
        
        # Extract relations (simplified - using dependency parsing)
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                # Find relation
                head = token.head
                if head.pos_ == "VERB":
                    relation = head.text
                    subject = None
                    object_ = None
                    
                    # Find subject and object
                    for child in head.children:
                        if child.dep_ == "nsubj":
                            subject = child.text
                        elif child.dep_ in ["dobj", "pobj"]:
                            object_ = child.text
                    
                    if subject and object_:
                        relations.append({
                            "source": subject,
                            "relation": relation,
                            "target": object_
                        })
        
        return {
            "entities": entities,
            "relations": relations
        }
    
    def build_graph(self, documents: List[str]):
        """Build knowledge graph from multiple documents."""
        for doc_idx, text in enumerate(documents):
            extracted = self.extract_entities_relations(text)
            
            # Add entities as nodes
            for entity, info in extracted["entities"].items():
                if entity not in self.graph:
                    self.graph.add_node(entity, label=info["label"], docs=[doc_idx])
                else:
                    # Update document list
                    if "docs" in self.graph.nodes[entity]:
                        self.graph.nodes[entity]["docs"].append(doc_idx)
            
            # Add relations as edges
            for rel in extracted["relations"]:
                source = rel["source"]
                target = rel["target"]
                relation = rel["relation"]
                
                if source in self.graph and target in self.graph:
                    if self.graph.has_edge(source, target):
                        # Update edge weight
                        self.graph[source][target]["weight"] += 1
                        self.graph[source][target]["relations"].append(relation)
                    else:
                        self.graph.add_edge(source, target, weight=1, relations=[relation])
    
    def query_graph(self, entity: str, depth: int = 2) -> Dict:
        """Query graph starting from an entity."""
        if entity not in self.graph:
            return {"error": "Entity not found"}
        
        # Get subgraph
        nodes = [entity]
        for _ in range(depth):
            new_nodes = []
            for node in nodes:
                neighbors = list(self.graph.neighbors(node))
                new_nodes.extend(neighbors)
            nodes.extend(new_nodes)
            nodes = list(set(nodes))
        
        subgraph = self.graph.subgraph(nodes)
        
        return {
            "entity": entity,
            "neighbors": list(self.graph.neighbors(entity)),
            "subgraph_nodes": list(subgraph.nodes()),
            "subgraph_edges": list(subgraph.edges(data=True))
        }
    
    def visualize_graph(self, output_file: str = "graph.html"):
        """Visualize knowledge graph."""
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        plt.axis('off')
        plt.savefig(output_file.replace('.html', '.png'))
        plt.close()

# Example Usage
kg_builder = KnowledgeGraphBuilder()
kg_builder.build_graph(documents)
results = kg_builder.query_graph("machine learning", depth=2)
kg_builder.visualize_graph("knowledge_graph.png")
```

### 8.2 Entity Extraction and Relation Mapping

Advanced entity and relation extraction.

```python
from transformers import pipeline

class AdvancedEntityExtractor:
    """Advanced entity and relation extraction."""
    
    def __init__(self):
        # NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        
        # Relation extraction (simplified - would use specialized model)
        self.relation_model = None  # Would load relation extraction model
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using transformer model."""
        entities = self.ner_pipeline(text)
        return [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": ent["score"],
                "start": ent["start"],
                "end": ent["end"]
            }
            for ent in entities
        ]
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities."""
        # Simplified relation extraction
        # In practice, use a relation extraction model
        relations = []
        
        # Find co-occurring entities in sentences
        sentences = text.split('.')
        for sentence in sentences:
            sentence_entities = [
                e for e in entities
                if e["start"] >= text.find(sentence) and e["end"] <= text.find(sentence) + len(sentence)
            ]
            
            if len(sentence_entities) >= 2:
                # Create relations (simplified)
                for i in range(len(sentence_entities) - 1):
                    relations.append({
                        "source": sentence_entities[i]["text"],
                        "target": sentence_entities[i+1]["text"],
                        "relation": "related_to",
                        "context": sentence
                    })
        
        return relations
    
    def build_structured_graph(self, documents: List[str]) -> nx.DiGraph:
        """Build structured knowledge graph."""
        graph = nx.DiGraph()
        
        for doc_idx, text in enumerate(documents):
            entities = self.extract_entities(text)
            relations = self.extract_relations(text, entities)
            
            # Add entities
            for entity in entities:
                entity_id = f"{entity['text']}_{entity['label']}"
                if entity_id not in graph:
                    graph.add_node(entity_id, **entity, docs=[doc_idx])
                else:
                    graph.nodes[entity_id]["docs"].append(doc_idx)
            
            # Add relations
            for rel in relations:
                source_id = f"{rel['source']}_ENTITY"
                target_id = f"{rel['target']}_ENTITY"
                
                if source_id in graph and target_id in graph:
                    if graph.has_edge(source_id, target_id):
                        graph[source_id][target_id]["weight"] += 1
                    else:
                        graph.add_edge(
                            source_id,
                            target_id,
                            relation=rel["relation"],
                            weight=1,
                            context=rel["context"]
                        )
        
        return graph

# Example
extractor = AdvancedEntityExtractor()
graph = extractor.build_structured_graph(documents)
```

### 8.3 Neo4j Integration

Integrate knowledge graphs with Neo4j database.

```python
from neo4j import GraphDatabase

class Neo4jKnowledgeGraph:
    """Neo4j knowledge graph integration."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Connect to Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close connection."""
        self.driver.close()
    
    def create_entity(self, entity_name: str, entity_type: str, properties: Dict = None):
        """Create entity node."""
        with self.driver.session() as session:
            props = properties or {}
            props["name"] = entity_name
            props["type"] = entity_type
            
            query = f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e += $props
            RETURN e
            """
            session.run(query, name=entity_name, props=props)
    
    def create_relation(self, source: str, target: str, relation_type: str,
                       properties: Dict = None):
        """Create relationship between entities."""
        with self.driver.session() as session:
            props = properties or {}
            
            query = f"""
            MATCH (a), (b)
            WHERE a.name = $source AND b.name = $target
            MERGE (a)-[r:{relation_type}]->(b)
            SET r += $props
            RETURN r
            """
            session.run(query, source=source, target=target, props=props)
    
    def query_entities(self, entity_name: str, depth: int = 2) -> List[Dict]:
        """Query entities and their relationships."""
        with self.driver.session() as session:
            query = f"""
            MATCH path = (start {{name: $name}})-[*1..{depth}]-(connected)
            RETURN path
            LIMIT 100
            """
            result = session.run(query, name=entity_name)
            return [record["path"] for record in result]
    
    def vector_search_with_graph(self, query_embedding: List[float],
                                entity_name: str) -> List[Dict]:
        """Combine vector search with graph traversal."""
        # Find similar entities using vector search (would need vector index)
        # Then traverse graph from those entities
        
        with self.driver.session() as session:
            # Simplified - in practice would use vector index
            query = """
            MATCH (e)
            WHERE e.name CONTAINS $query_term
            MATCH path = (e)-[*1..2]-(connected)
            RETURN path
            LIMIT 20
            """
            result = session.run(query, query_term=entity_name)
            return [record["path"] for record in result]

# Example Usage
neo4j_kg = Neo4jKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
neo4j_kg.create_entity("Machine Learning", "Concept", {"description": "AI subset"})
neo4j_kg.create_relation("Machine Learning", "Neural Networks", "USES")
results = neo4j_kg.query_entities("Machine Learning", depth=2)
```

### 8.4 Combining KG with Vector Search

Hybrid approach combining knowledge graphs and vector embeddings.

```python
class HybridKGVectorSearch:
    """Combine knowledge graph and vector search."""
    
    def __init__(self, graph: nx.DiGraph, vector_store, entity_embeddings: Dict):
        """
        Args:
            graph: Knowledge graph
            vector_store: Vector database/store
            entity_embeddings: Dict mapping entity names to embeddings
        """
        self.graph = graph
        self.vector_store = vector_store
        self.entity_embeddings = entity_embeddings
    
    def search(self, query: str, query_embedding: np.ndarray, k: int = 5) -> Dict:
        """Hybrid search."""
        # Vector search for documents
        vector_results = self.vector_store.search(query_embedding, k=k*2)
        
        # Find relevant entities
        entity_scores = {}
        for entity, emb in self.entity_embeddings.items():
            score = cosine_similarity([query_embedding], [emb])[0][0]
            entity_scores[entity] = score
        
        top_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Graph expansion from top entities
        graph_docs = []
        for entity, score in top_entities:
            if entity in self.graph:
                # Get neighbors
                neighbors = list(self.graph.neighbors(entity))
                # Get documents containing these entities
                for neighbor in neighbors:
                    if neighbor in self.graph.nodes:
                        doc_ids = self.graph.nodes[neighbor].get("docs", [])
                        graph_docs.extend([
                            {"entity": neighbor, "doc_id": doc_id, "score": score}
                            for doc_id in doc_ids
                        ])
        
        # Combine results
        doc_scores = {}
        for result in vector_results:
            doc_id = result.get("doc_id", hash(result["text"]))
            doc_scores[doc_id] = result["score"] * 0.7  # Weight vector results
        
        for graph_doc in graph_docs:
            doc_id = graph_doc["doc_id"]
            if doc_id in doc_scores:
                doc_scores[doc_id] += graph_doc["score"] * 0.3  # Add graph boost
            else:
                doc_scores[doc_id] = graph_doc["score"] * 0.3
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return {
            "results": sorted_docs,
            "top_entities": [e[0] for e in top_entities],
            "graph_expansion": len(graph_docs)
        }

# Example
hybrid_search = HybridKGVectorSearch(graph, vector_store, entity_embeddings)
results = hybrid_search.search("machine learning", query_embedding, k=5)
```

---

## 9. Evaluation

### 9.1 Retrieval Metrics

Evaluate retrieval quality using standard metrics.

```python
from typing import List, Set

class RetrievalMetrics:
    """Calculate retrieval evaluation metrics."""
    
    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Precision@K: Fraction of retrieved docs that are relevant."""
        retrieved_k = set(retrieved[:k])
        if len(retrieved_k) == 0:
            return 0.0
        return len(retrieved_k & relevant) / len(retrieved_k)
    
    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Recall@K: Fraction of relevant docs that were retrieved."""
        retrieved_k = set(retrieved[:k])
        if len(relevant) == 0:
            return 0.0
        return len(retrieved_k & relevant) / len(relevant)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[int]], 
                            relevant_sets: List[Set[int]]) -> float:
        """MRR: Average reciprocal rank of first relevant document."""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: Set[int], 
                 relevance_scores: Dict[int, float], k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain."""
        # Get relevance scores for retrieved docs
        retrieved_relevance = [
            relevance_scores.get(doc_id, 0.0) for doc_id in retrieved[:k]
        ]
        
        # Calculate DCG
        dcg = sum(
            rel / np.log2(idx + 2)
            for idx, rel in enumerate(retrieved_relevance)
        )
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(
            rel / np.log2(idx + 2)
            for idx, rel in enumerate(ideal_relevance)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(self, retrieved_lists: List[List[int]],
                          relevant_sets: List[Set[int]],
                          relevance_scores: List[Dict[int, float]] = None,
                          k_values: List[int] = [1, 5, 10]) -> Dict:
        """Comprehensive retrieval evaluation."""
        results = {}
        
        for k in k_values:
            precisions = [
                self.precision_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(retrieved_lists, relevant_sets)
            ]
            recalls = [
                self.recall_at_k(retrieved, relevant, k)
                for retrieved, relevant in zip(retrieved_lists, relevant_sets)
            ]
            
            results[f"precision@{k}"] = np.mean(precisions)
            results[f"recall@{k}"] = np.mean(recalls)
            
            if relevance_scores:
                ndcgs = [
                    self.ndcg_at_k(retrieved, relevant, scores, k)
                    for retrieved, relevant, scores in zip(
                        retrieved_lists, relevant_sets, relevance_scores
                    )
                ]
                results[f"ndcg@{k}"] = np.mean(ndcgs)
        
        results["mrr"] = self.mean_reciprocal_rank(retrieved_lists, relevant_sets)
        
        return results

# Example Usage
metrics = RetrievalMetrics()

# Example: 3 queries
retrieved_lists = [
    [1, 3, 5, 7, 9],  # Query 1 results
    [2, 4, 6, 8, 10],  # Query 2 results
    [1, 2, 3, 4, 5]   # Query 3 results
]

relevant_sets = [
    {1, 3, 5},  # Relevant for query 1
    {2, 4},     # Relevant for query 2
    {1, 2, 6}   # Relevant for query 3
]

evaluation = metrics.evaluate_retrieval(retrieved_lists, relevant_sets, k_values=[1, 5, 10])
print(evaluation)
```

### 9.2 Generation Metrics

Evaluate generated answers.

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score

class GenerationMetrics:
    """Evaluate generation quality."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def rouge_score(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    def bert_score(self, generated: List[str], reference: List[str]) -> Dict:
        """Calculate BERTScore."""
        P, R, F1 = bert_score(generated, reference, lang='en', verbose=False)
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    
    def faithfulness(self, generated: str, source_docs: List[str], llm=None) -> float:
        """Evaluate if generated answer is faithful to sources."""
        if llm is None:
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=0)
        
        sources_text = "\n\n".join([f"Source {i+1}: {doc[:500]}" for i, doc in enumerate(source_docs)])
        
        prompt = f"""Evaluate if the generated answer is faithful to the source documents.
Rate from 0.0 (not faithful, contains unsupported claims) to 1.0 (fully faithful, all claims supported).

Generated Answer:
{generated}

Source Documents:
{sources_text}

Provide a faithfulness score (0.0-1.0) and brief explanation."""
        
        response = llm(prompt)
        # Extract score (simplified)
        import re
        score_match = re.search(r'(\d+\.?\d*)', response)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return min(max(score, 0.0), 1.0)
    
    def answer_relevance(self, query: str, answer: str, llm=None) -> float:
        """Evaluate if answer is relevant to query."""
        if llm is None:
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=0)
        
        prompt = f"""Rate how well this answer addresses the query.
Score from 0.0 (not relevant) to 1.0 (highly relevant).

Query: {query}
Answer: {answer}

Provide a relevance score (0.0-1.0):"""
        
        response = llm(prompt)
        import re
        score_match = re.search(r'(\d+\.?\d*)', response)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return min(max(score, 0.0), 1.0)
    
    def answer_correctness(self, answer: str, reference_answer: str, llm=None) -> float:
        """Evaluate correctness compared to reference."""
        if llm is None:
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=0)
        
        prompt = f"""Compare the generated answer with the reference answer.
Rate correctness from 0.0 (incorrect) to 1.0 (fully correct).

Generated Answer:
{answer}

Reference Answer:
{reference_answer}

Provide a correctness score (0.0-1.0):"""
        
        response = llm(prompt)
        import re
        score_match = re.search(r'(\d+\.?\d*)', response)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return min(max(score, 0.0), 1.0)

# Example Usage
gen_metrics = GenerationMetrics()
rouge = gen_metrics.rouge_score(generated_answer, reference_answer)
faithfulness = gen_metrics.faithfulness(generated_answer, source_documents)
relevance = gen_metrics.answer_relevance(query, generated_answer)
```

### 9.3 RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) provides comprehensive evaluation.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

class RAGASEvaluator:
    """RAGAS evaluation framework."""
    
    def evaluate_rag(self, questions: List[str], 
                     ground_truths: List[str],
                     answers: List[str],
                     contexts: List[List[str]],
                     retrieval_scores: List[List[float]] = None) -> Dict:
        """
        Evaluate RAG system using RAGAS.
        
        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            answers: List of generated answers
            contexts: List of retrieved context lists
            retrieval_scores: Optional retrieval similarity scores
        """
        # Prepare dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        if retrieval_scores:
            data["retrieval_scores"] = retrieval_scores
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        return result
    
    def evaluate_retrieval_only(self, questions: List[str],
                               contexts: List[List[str]],
                               ground_truth_contexts: List[List[str]],
                               retrieval_scores: List[List[float]] = None) -> Dict:
        """Evaluate only retrieval component."""
        # Create dummy answers for retrieval-only evaluation
        answers = ["dummy"] * len(questions)
        ground_truths = ["dummy"] * len(questions)
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        if retrieval_scores:
            data["retrieval_scores"] = retrieval_scores
        
        dataset = Dataset.from_dict(data)
        
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall
            ]
        )
        
        return result

# Example Usage
ragas_evaluator = RAGASEvaluator()

questions = ["What is machine learning?", "How do neural networks work?"]
ground_truths = ["ML is a subset of AI...", "Neural networks are..."]
answers = ["Machine learning is...", "Neural networks consist of..."]
contexts = [["Doc1 about ML", "Doc2 about AI"], ["Doc3 about NN", "Doc4 about DL"]]

results = ragas_evaluator.evaluate_rag(
    questions=questions,
    ground_truths=ground_truths,
    answers=answers,
    contexts=contexts
)
print(results)
```

### 9.4 End-to-End Evaluation Pipeline

Complete evaluation pipeline for RAG systems.

```python
class RAGEvaluationPipeline:
    """End-to-end RAG evaluation pipeline."""
    
    def __init__(self, rag_system, retrieval_metrics: RetrievalMetrics,
                 generation_metrics: GenerationMetrics):
        """
        Args:
            rag_system: RAG system to evaluate
            retrieval_metrics: RetrievalMetrics instance
            generation_metrics: GenerationMetrics instance
        """
        self.rag_system = rag_system
        self.retrieval_metrics = retrieval_metrics
        self.generation_metrics = generation_metrics
    
    def evaluate_on_dataset(self, test_dataset: List[Dict]) -> Dict:
        """
        Evaluate on test dataset.
        
        Args:
            test_dataset: List of dicts with keys:
                - "question": str
                - "ground_truth_answer": str
                - "relevant_doc_ids": List[int]
                - "relevance_scores": Dict[int, float] (optional)
        """
        all_retrieved = []
        all_relevant = []
        all_answers = []
        all_references = []
        all_contexts = []
        
        for example in test_dataset:
            # Run RAG system
            result = self.rag_system.query(example["question"])
            
            # Collect retrieval results
            retrieved_ids = [r.get("index", i) for i, r in enumerate(result["sources"])]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(set(example["relevant_doc_ids"]))
            
            # Collect generation results
            all_answers.append(result["answer"])
            all_references.append(example["ground_truth_answer"])
            all_contexts.append([r["text"] if isinstance(r, dict) else r for r in result["sources"]])
        
        # Evaluate retrieval
        retrieval_eval = self.retrieval_metrics.evaluate_retrieval(
            retrieved_lists=all_retrieved,
            relevant_sets=all_relevant,
            k_values=[1, 5, 10]
        )
        
        # Evaluate generation
        rouge_scores = []
        faithfulness_scores = []
        relevance_scores = []
        
        for answer, reference, contexts in zip(all_answers, all_references, all_contexts):
            rouge = self.generation_metrics.rouge_score(answer, reference)
            rouge_scores.append(rouge)
            
            faithfulness = self.generation_metrics.faithfulness(answer, contexts)
            faithfulness_scores.append(faithfulness)
            
            # Note: would need original questions for relevance
            # relevance = self.generation_metrics.answer_relevance(question, answer)
            # relevance_scores.append(relevance)
        
        generation_eval = {
            "rouge1": np.mean([r["rouge1"] for r in rouge_scores]),
            "rouge2": np.mean([r["rouge2"] for r in rouge_scores]),
            "rougeL": np.mean([r["rougeL"] for r in rouge_scores]),
            "faithfulness": np.mean(faithfulness_scores)
        }
        
        return {
            "retrieval": retrieval_eval,
            "generation": generation_eval,
            "overall": {
                "retrieval_score": np.mean([
                    retrieval_eval["precision@5"],
                    retrieval_eval["recall@5"],
                    retrieval_eval.get("ndcg@5", 0)
                ]),
                "generation_score": np.mean([
                    generation_eval["rouge1"],
                    generation_eval["faithfulness"]
                ])
            }
        }
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate human-readable evaluation report."""
        report = "# RAG System Evaluation Report\n\n"
        
        report += "## Retrieval Metrics\n\n"
        retrieval = evaluation_results["retrieval"]
        report += f"- Precision@5: {retrieval['precision@5']:.3f}\n"
        report += f"- Recall@5: {retrieval['recall@5']:.3f}\n"
        report += f"- MRR: {retrieval['mrr']:.3f}\n"
        if "ndcg@5" in retrieval:
            report += f"- NDCG@5: {retrieval['ndcg@5']:.3f}\n"
        
        report += "\n## Generation Metrics\n\n"
        generation = evaluation_results["generation"]
        report += f"- ROUGE-1: {generation['rouge1']:.3f}\n"
        report += f"- ROUGE-2: {generation['rouge2']:.3f}\n"
        report += f"- ROUGE-L: {generation['rougeL']:.3f}\n"
        report += f"- Faithfulness: {generation['faithfulness']:.3f}\n"
        
        report += "\n## Overall Scores\n\n"
        overall = evaluation_results["overall"]
        report += f"- Retrieval Score: {overall['retrieval_score']:.3f}\n"
        report += f"- Generation Score: {overall['generation_score']:.3f}\n"
        
        return report

# Example Usage
eval_pipeline = RAGEvaluationPipeline(rag_system, retrieval_metrics, generation_metrics)

test_data = [
    {
        "question": "What is machine learning?",
        "ground_truth_answer": "Machine learning is a subset of artificial intelligence...",
        "relevant_doc_ids": [0, 1, 2],
        "relevance_scores": {0: 1.0, 1: 0.8, 2: 0.6}
    },
    # ... more examples
]

results = eval_pipeline.evaluate_on_dataset(test_data)
report = eval_pipeline.generate_report(results)
print(report)
```

---

## Conclusion

This comprehensive guide covered RAG and Knowledge Systems from fundamentals to advanced patterns. Key takeaways:

1. **RAG Fundamentals**: Understand when and why to use RAG vs alternatives
2. **Document Processing**: Proper ingestion and cleaning are crucial
3. **Chunking**: Strategy significantly impacts retrieval quality
4. **Embeddings**: Choose models based on quality/speed/cost trade-offs
5. **Vector Databases**: Select based on scale and feature requirements
6. **Retrieval Methods**: Combine multiple approaches for best results
7. **Advanced Patterns**: Use CRAG, Self-RAG, Agentic RAG for complex scenarios
8. **Knowledge Graphs**: Enhance RAG with structured knowledge
9. **Evaluation**: Comprehensive evaluation ensures system quality

Remember: RAG is an iterative process. Continuously evaluate, refine chunking strategies, experiment with retrieval methods, and optimize based on your specific use case and data.
