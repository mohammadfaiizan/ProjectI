#!/usr/bin/env python3
"""
Basic RAG Implementation: Foundation of Knowledge-Augmented Generation
=====================================================================

WHAT IS THE PROBLEM?
==================
Large Language Models have knowledge limitations:
- Training data has a cutoff date
- Can't access real-time information
- May hallucinate facts not in training data
- Limited ability to cite sources
- No access to private/proprietary knowledge

Example: Medical Diagnosis Failure
PURE LLM APPROACH (Dangerous):
- Doctor asks LLM about rare disease symptoms
- LLM generates plausible but outdated treatment
- New research published after training cutoff
- Patient receives incorrect medical advice
- No way to verify or update the information

REAL WORLD EXAMPLE:
=================
How does ChatGPT with web browsing provide current information?

CHATGPT RAG PROCESS:
When you ask about today's news:
1. RETRIEVE: Search current web sources for relevant information
2. AUGMENT: Add retrieved context to the user's question
3. GENERATE: LLM generates response using both its training and retrieved data
4. CITE: Provides sources and links for verification
5. UPDATE: Always accesses the most current information available

BENEFITS:
- Combines LLM reasoning with current factual data
- Provides source attribution and verification
- Enables access to private knowledge bases
- Reduces hallucinations through grounding
- Allows real-time information integration

THE ALGORITHM:
=============
1. ENCODE: Convert query to dense vector representation
2. RETRIEVE: Find most relevant documents using similarity search
3. RANK: Score and select top-k most relevant passages
4. AUGMENT: Combine query with retrieved context
5. GENERATE: LLM produces response using augmented prompt
6. CITE: Extract and format source attributions
7. VALIDATE: Optionally verify generated content against sources

RAG COMPONENTS:
- Document Store: Vector database of embedded knowledge
- Retriever: Similarity search engine
- Generator: Language model for response generation
- Orchestrator: Coordinates retrieval and generation

WHY IS THIS REVOLUTIONARY?
========================
- Extends LLM capabilities beyond training data
- Enables dynamic knowledge updates without retraining
- Provides factual grounding and reduces hallucinations
- Supports enterprise knowledge integration
- Powers next-generation AI assistants and search
- Foundation for agentic knowledge systems
"""

import asyncio
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import sqlite3
import pickle
from pathlib import Path
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class Document:
    """Document in the knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Content structure
    title: str = ""
    source: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Processing metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    processed: bool = False
    
    # Embeddings
    embedding: Optional[np.ndarray] = None
    embedding_model: str = ""
    
    def __post_init__(self):
        """Initialize document"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.title and self.content:
            # Extract title from first line or first 50 chars
            lines = self.content.split('\n')
            self.title = lines[0][:50] if lines else self.content[:50]
    
    def get_content_hash(self) -> str:
        """Get hash of document content"""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'title': self.title,
            'source': self.source,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'processed': self.processed,
            'embedding_model': self.embedding_model
        }
        
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary"""
        embedding = None
        if 'embedding' in data and data['embedding']:
            embedding = np.array(data['embedding'])
        
        doc = cls(
            id=data['id'],
            content=data['content'],
            metadata=data.get('metadata', {}),
            title=data.get('title', ''),
            source=data.get('source', ''),
            chunk_index=data.get('chunk_index', 0),
            total_chunks=data.get('total_chunks', 1),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            processed=data.get('processed', False),
            embedding_model=data.get('embedding_model', '')
        )
        
        doc.embedding = embedding
        return doc

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    rank: int
    
    def __post_init__(self):
        """Ensure score is valid"""
        if self.score < 0:
            self.score = 0.0
        elif self.score > 1:
            self.score = 1.0

@dataclass
class RAGQuery:
    """Query for RAG system"""
    id: str
    query_text: str
    
    # Query metadata
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Retrieval parameters
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Generation parameters
    max_length: int = 500
    temperature: float = 0.7
    
    def __post_init__(self):
        """Initialize query"""
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class RAGResponse:
    """Response from RAG system"""
    query_id: str
    generated_text: str
    retrieved_documents: List[RetrievalResult]
    
    # Response metadata
    response_time: float = 0.0
    total_tokens: int = 0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    
    # Quality metrics
    confidence_score: float = 0.0
    citation_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    
    def add_citation(self, source: str) -> None:
        """Add source citation"""
        if source not in self.sources_used:
            self.sources_used.append(source)
            self.citation_count += 1

class TextEmbedder(ABC):
    """Abstract text embedder"""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        pass

class SimpleEmbedder(TextEmbedder):
    """Simple TF-IDF based embedder for demonstration"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        return [token for token in text.split() if len(token) > 2]
    
    def fit(self, documents: List[str]) -> None:
        """Fit embedder on documents"""
        # Count term frequencies across documents
        doc_freq = {}
        total_docs = len(documents)
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # Build vocabulary (most frequent terms)
        sorted_terms = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {
            term: idx for idx, (term, _) in enumerate(sorted_terms[:self.vocab_size])
        }
        
        # Calculate IDF scores
        for term, freq in doc_freq.items():
            self.idf_scores[term] = np.log(total_docs / (1 + freq))
        
        self.fitted = True
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate TF-IDF embedding"""
        if not self.fitted:
            raise ValueError("Embedder not fitted. Call fit() first.")
        
        tokens = self._tokenize(text)
        
        # Count term frequencies
        tf_scores = {}
        for token in tokens:
            tf_scores[token] = tf_scores.get(token, 0) + 1
        
        # Normalize by document length
        doc_length = len(tokens)
        if doc_length > 0:
            for token in tf_scores:
                tf_scores[token] /= doc_length
        
        # Create embedding vector
        embedding = np.zeros(len(self.vocabulary))
        for token, tf in tf_scores.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                embedding[idx] = tf * idf
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch"""
        return [self.embed_text(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return len(self.vocabulary)

class VectorStore:
    """Simple vector store for document embeddings"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.embeddings: Dict[str, np.ndarray] = {}
        
    async def initialize(self) -> None:
        """Initialize vector store"""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                title TEXT,
                source TEXT,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                created_at REAL,
                updated_at REAL,
                processed BOOLEAN DEFAULT FALSE,
                embedding_model TEXT,
                content_hash TEXT
            )
        """)
        
        self.conn.commit()
    
    async def add_document(self, document: Document) -> None:
        """Add document to store"""
        if not self.conn:
            await self.initialize()
        
        # Store document data
        self.conn.execute("""
            INSERT OR REPLACE INTO documents 
            (id, content, metadata, title, source, chunk_index, total_chunks,
             created_at, updated_at, processed, embedding_model, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.id,
            document.content,
            json.dumps(document.metadata),
            document.title,
            document.source,
            document.chunk_index,
            document.total_chunks,
            document.created_at,
            document.updated_at,
            document.processed,
            document.embedding_model,
            document.get_content_hash()
        ))
        
        # Store embedding separately
        if document.embedding is not None:
            self.embeddings[document.id] = document.embedding
        
        self.conn.commit()
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        if not self.conn:
            return None
        
        cursor = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Reconstruct document
        doc_data = {
            'id': row[0],
            'content': row[1],
            'metadata': json.loads(row[2]) if row[2] else {},
            'title': row[3] or '',
            'source': row[4] or '',
            'chunk_index': row[5] or 0,
            'total_chunks': row[6] or 1,
            'created_at': row[7] or time.time(),
            'updated_at': row[8] or time.time(),
            'processed': bool(row[9]),
            'embedding_model': row[10] or ''
        }
        
        document = Document.from_dict(doc_data)
        
        # Add embedding if available
        if doc_id in self.embeddings:
            document.embedding = self.embeddings[doc_id]
        
        return document
    
    async def search_similar(self, query_embedding: np.ndarray, 
                           top_k: int = 5, threshold: float = 0.0) -> List[RetrievalResult]:
        """Search for similar documents"""
        if not self.conn:
            return []
        
        # Get all document IDs
        cursor = self.conn.execute("SELECT id FROM documents WHERE processed = TRUE")
        doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Calculate similarities
        similarities = []
        for doc_id in doc_ids:
            if doc_id in self.embeddings:
                doc_embedding = self.embeddings[doc_id]
                
                # Cosine similarity
                dot_product = np.dot(query_embedding, doc_embedding)
                norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and limit
        filtered_similarities = [
            (doc_id, score) for doc_id, score in similarities 
            if score >= threshold
        ][:top_k]
        
        # Get documents and create results
        results = []
        for rank, (doc_id, score) in enumerate(filtered_similarities):
            document = await self.get_document(doc_id)
            if document:
                result = RetrievalResult(
                    document=document,
                    score=score,
                    rank=rank + 1
                )
                results.append(result)
        
        return results
    
    async def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        if not self.conn:
            return []
        
        cursor = self.conn.execute("SELECT id FROM documents")
        doc_ids = [row[0] for row in cursor.fetchall()]
        
        documents = []
        for doc_id in doc_ids:
            doc = await self.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        return documents
    
    async def close(self) -> None:
        """Close vector store"""
        if self.conn:
            self.conn.close()
            self.conn = None

class SimpleGenerator:
    """Simple text generator for demonstration"""
    
    def __init__(self):
        self.max_length = 500
        
    async def generate_response(self, query: str, context_docs: List[Document],
                              max_length: int = None, temperature: float = 0.7) -> str:
        """Generate response using query and context"""
        
        # In a real implementation, this would use a language model
        # For demonstration, we'll create a template-based response
        
        max_len = max_length or self.max_length
        
        # Extract key information from context documents
        context_snippets = []
        sources = []
        
        for doc in context_docs[:3]:  # Use top 3 documents
            # Extract relevant snippet (first 200 chars)
            snippet = doc.content[:200].strip()
            if snippet:
                context_snippets.append(snippet)
                if doc.source:
                    sources.append(doc.source)
        
        # Create response
        if context_snippets:
            response = f"Based on the available information, here's what I found regarding your query:\n\n"
            
            for i, snippet in enumerate(context_snippets, 1):
                response += f"{i}. {snippet}...\n\n"
            
            response += "This information is based on the following sources:\n"
            for i, source in enumerate(sources, 1):
                response += f"- Source {i}: {source}\n"
        else:
            response = "I couldn't find specific information to answer your query in the knowledge base."
        
        # Truncate if too long
        if len(response) > max_len:
            response = response[:max_len-3] + "..."
        
        return response

class BasicRAGSystem:
    """
    Basic RAG (Retrieval-Augmented Generation) System
    
    EXAMPLE USAGE:
    =============
    # Create RAG system
    rag = BasicRAGSystem()
    await rag.initialize()
    
    # Add documents to knowledge base
    docs = [
        Document(id="", content="Python is a programming language...", source="Python docs"),
        Document(id="", content="Machine learning is a subset of AI...", source="ML textbook")
    ]
    
    for doc in docs:
        await rag.add_document(doc)
    
    # Query the system
    query = RAGQuery(id="", query_text="What is Python used for?")
    response = await rag.query(query)
    
    print(response.generated_text)
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.vector_store = VectorStore(db_path)
        self.embedder = SimpleEmbedder()
        self.generator = SimpleGenerator()
        
        # System state
        self.initialized = False
        self.total_documents = 0
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_documents': 0,
            'average_retrieval_time': 0.0,
            'average_generation_time': 0.0,
            'average_response_time': 0.0
        }
        
        self.logger = logging.getLogger("BasicRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize RAG system"""
        await self.vector_store.initialize()
        self.initialized = True
        self.logger.info("RAG system initialized")
    
    async def add_document(self, document: Document) -> None:
        """Add document to knowledge base"""
        if not self.initialized:
            await self.initialize()
        
        # Check if we need to fit embedder
        if not self.embedder.fitted:
            # For first document, we'll fit on just this document
            # In practice, you'd collect a corpus first
            self.embedder.fit([document.content])
        
        # Generate embedding
        embedding = self.embedder.embed_text(document.content)
        document.embedding = embedding
        document.embedding_model = "simple_tfidf"
        document.processed = True
        
        # Store document
        await self.vector_store.add_document(document)
        self.total_documents += 1
        self.stats['total_documents'] = self.total_documents
        
        self.logger.info(f"Added document: {document.title[:50]}...")
    
    async def add_documents_batch(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently"""
        if not documents:
            return
        
        # Fit embedder on all documents
        all_texts = [doc.content for doc in documents]
        self.embedder.fit(all_texts)
        
        # Process documents
        for document in documents:
            await self.add_document(document)
        
        self.logger.info(f"Added {len(documents)} documents to knowledge base")
    
    async def query(self, query: RAGQuery) -> RAGResponse:
        """Process RAG query"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query.query_text)
        
        # Search for similar documents
        retrieved_docs = await self.vector_store.search_similar(
            query_embedding,
            top_k=query.top_k,
            threshold=query.similarity_threshold
        )
        
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generate response
        generation_start = time.time()
        
        context_documents = [result.document for result in retrieved_docs]
        generated_text = await self.generator.generate_response(
            query.query_text,
            context_documents,
            max_length=query.max_length,
            temperature=query.temperature
        )
        
        generation_time = time.time() - generation_start
        
        # Step 3: Create response
        total_time = time.time() - start_time
        
        response = RAGResponse(
            query_id=query.id,
            generated_text=generated_text,
            retrieved_documents=retrieved_docs,
            response_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=len(generated_text.split()),
            confidence_score=self._calculate_confidence(retrieved_docs),
            citation_count=len(retrieved_docs),
            sources_used=[doc.document.source for doc in retrieved_docs if doc.document.source]
        )
        
        # Update statistics
        self._update_stats(total_time, retrieval_time, generation_time)
        
        self.logger.info(f"Processed query: {query.query_text[:50]}... "
                        f"(retrieved {len(retrieved_docs)} docs, {total_time:.3f}s)")
        
        return response
    
    def _calculate_confidence(self, retrieved_docs: List[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not retrieved_docs:
            return 0.0
        
        # Simple confidence based on top score and number of results
        top_score = retrieved_docs[0].score if retrieved_docs else 0.0
        doc_count_factor = min(len(retrieved_docs) / 5.0, 1.0)  # Normalize to max 5 docs
        
        confidence = (top_score * 0.7) + (doc_count_factor * 0.3)
        return min(confidence, 1.0)
    
    def _update_stats(self, total_time: float, retrieval_time: float, generation_time: float) -> None:
        """Update running statistics"""
        query_count = self.stats['total_queries']
        
        # Running averages
        self.stats['average_response_time'] = (
            (self.stats['average_response_time'] * (query_count - 1) + total_time) / query_count
        )
        
        self.stats['average_retrieval_time'] = (
            (self.stats['average_retrieval_time'] * (query_count - 1) + retrieval_time) / query_count
        )
        
        self.stats['average_generation_time'] = (
            (self.stats['average_generation_time'] * (query_count - 1) + generation_time) / query_count
        )
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return await self.vector_store.get_document(doc_id)
    
    async def search_documents(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search documents without generation"""
        if not self.initialized:
            await self.initialize()
        
        query_embedding = self.embedder.embed_text(query_text)
        return await self.vector_store.search_similar(query_embedding, top_k=top_k)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'system_stats': self.stats.copy(),
            'embedder_vocab_size': len(self.embedder.vocabulary) if self.embedder.fitted else 0,
            'embedder_fitted': self.embedder.fitted,
            'embedding_dimension': self.embedder.get_embedding_dimension() if self.embedder.fitted else 0,
            'initialized': self.initialized
        }
    
    async def close(self) -> None:
        """Close RAG system"""
        await self.vector_store.close()
        self.logger.info("RAG system closed")

# Utility functions for document processing
class DocumentProcessor:
    """Utility class for processing documents"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending in the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = -1
                
                for pos in range(end, search_start, -1):
                    if text[pos] in '.!?':
                        sentence_end = pos + 1
                        break
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    @staticmethod
    def create_documents_from_text(text: str, source: str = "", 
                                 chunk_size: int = 1000) -> List[Document]:
        """Create document objects from text"""
        chunks = DocumentProcessor.chunk_text(text, chunk_size)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                id="",
                content=chunk,
                source=source,
                chunk_index=i,
                total_chunks=len(chunks)
            )
            documents.append(doc)
        
        return documents

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_rag():
    """Demo: Basic RAG system functionality"""
    print("\nDEMO 1: BASIC RAG SYSTEM")
    print("=" * 50)
    
    # Create RAG system
    rag = BasicRAGSystem()
    await rag.initialize()
    
    # Sample knowledge base documents
    knowledge_docs = [
        Document(
            id="",
            content="Python is a high-level programming language known for its simplicity and readability. "
                   "It was created by Guido van Rossum and first released in 1991. Python supports multiple "
                   "programming paradigms including procedural, object-oriented, and functional programming. "
                   "It's widely used in web development, data science, artificial intelligence, and automation.",
            source="Python Programming Guide",
            title="Introduction to Python"
        ),
        Document(
            id="",
            content="Machine Learning is a subset of artificial intelligence that enables computers to learn "
                   "and make decisions from data without being explicitly programmed. Common ML algorithms "
                   "include linear regression, decision trees, neural networks, and support vector machines. "
                   "ML is used in applications like image recognition, natural language processing, and recommendation systems.",
            source="AI & ML Textbook",
            title="Machine Learning Fundamentals"
        ),
        Document(
            id="",
            content="Data Science is an interdisciplinary field that combines statistics, computer science, "
                   "and domain expertise to extract insights from data. Data scientists use tools like Python, "
                   "R, SQL, and various visualization libraries. The data science process includes data collection, "
                   "cleaning, analysis, modeling, and interpretation of results.",
            source="Data Science Handbook",
            title="What is Data Science"
        ),
        Document(
            id="",
            content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, "
                   "interpret, and manipulate human language. NLP techniques include tokenization, part-of-speech "
                   "tagging, named entity recognition, sentiment analysis, and machine translation. Modern NLP "
                   "uses transformer models like BERT and GPT for advanced language understanding.",
            source="NLP Research Papers",
            title="Natural Language Processing Overview"
        ),
        Document(
            id="",
            content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
                   "RAG systems first retrieve relevant documents from a knowledge base, then use these documents "
                   "as context for generating responses. This approach helps reduce hallucinations and provides "
                   "source attribution for generated content.",
            source="RAG Technical Documentation",
            title="Understanding RAG Systems"
        )
    ]
    
    print(f"Adding {len(knowledge_docs)} documents to knowledge base...")
    await rag.add_documents_batch(knowledge_docs)
    
    # Test queries
    test_queries = [
        "What is Python programming language?",
        "How does machine learning work?",
        "What tools do data scientists use?",
        "Explain natural language processing",
        "What is RAG and how does it work?"
    ]
    
    print(f"\nTesting RAG system with {len(test_queries)} queries:")
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query_text} ---")
        
        query = RAGQuery(
            id="",
            query_text=query_text,
            top_k=3,
            similarity_threshold=0.1
        )
        
        response = await rag.query(query)
        
        print(f"Retrieved {len(response.retrieved_documents)} documents:")
        for j, result in enumerate(response.retrieved_documents, 1):
            print(f"  {j}. {result.document.title} (score: {result.score:.3f})")
        
        print(f"\nGenerated Response:")
        print(response.generated_text)
        
        print(f"Response Time: {response.response_time:.3f}s "
              f"(Retrieval: {response.retrieval_time:.3f}s, "
              f"Generation: {response.generation_time:.3f}s)")
    
    # Show system statistics
    stats = rag.get_system_stats()
    print(f"\nSystem Statistics:")
    for key, value in stats['system_stats'].items():
        print(f"  {key}: {value}")
    
    await rag.close()

async def demo_document_chunking():
    """Demo: Document chunking and processing"""
    print("\nDEMO 2: DOCUMENT CHUNKING AND PROCESSING")
    print("=" * 50)
    
    # Long document for chunking
    long_document = """
    Artificial Intelligence (AI) is transforming industries across the globe. From healthcare to finance, 
    AI technologies are being deployed to solve complex problems and automate processes.
    
    In healthcare, AI is being used for medical imaging analysis, drug discovery, and personalized treatment 
    plans. Machine learning algorithms can analyze medical scans to detect diseases like cancer at early stages.
    
    The financial sector leverages AI for fraud detection, algorithmic trading, and risk assessment. 
    Banks use AI to analyze transaction patterns and identify suspicious activities in real-time.
    
    Transportation is another area where AI is making significant impact. Autonomous vehicles use computer 
    vision and sensor fusion to navigate roads safely. Ride-sharing companies optimize routes using AI algorithms.
    
    In retail, AI powers recommendation systems that suggest products based on customer behavior. 
    Inventory management systems use AI to predict demand and optimize stock levels.
    
    The future of AI looks promising with advancements in quantum computing, neuromorphic chips, and 
    brain-computer interfaces. These technologies will enable more powerful and efficient AI systems.
    
    However, AI also raises important ethical considerations around privacy, bias, and job displacement. 
    It's crucial to develop AI responsibly with proper governance and regulations.
    """
    
    print("Original document length:", len(long_document))
    
    # Chunk the document
    chunks = DocumentProcessor.chunk_text(long_document, chunk_size=300, overlap=50)
    
    print(f"Document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    # Create documents from chunks
    documents = DocumentProcessor.create_documents_from_text(
        long_document, 
        source="AI Industry Report",
        chunk_size=300
    )
    
    print(f"\nCreated {len(documents)} document objects:")
    for doc in documents:
        print(f"  Doc {doc.chunk_index + 1}/{doc.total_chunks}: {doc.title[:50]}...")
    
    # Test with RAG system
    rag = BasicRAGSystem()
    await rag.initialize()
    
    print(f"\nAdding chunked documents to RAG system...")
    await rag.add_documents_batch(documents)
    
    # Query the chunked content
    query = RAGQuery(
        id="",
        query_text="How is AI used in healthcare?",
        top_k=2
    )
    
    response = await rag.query(query)
    
    print(f"\nQuery: {query.query_text}")
    print(f"Retrieved chunks:")
    for result in response.retrieved_documents:
        print(f"  - Chunk {result.document.chunk_index + 1}: score {result.score:.3f}")
        print(f"    Content: {result.document.content[:100]}...")
    
    print(f"\nGenerated Response:")
    print(response.generated_text)
    
    await rag.close()

async def demo_similarity_search():
    """Demo: Document similarity search"""
    print("\nDEMO 3: DOCUMENT SIMILARITY SEARCH")
    print("=" * 50)
    
    rag = BasicRAGSystem()
    await rag.initialize()
    
    # Add diverse documents
    tech_docs = [
        Document(id="", content="JavaScript is a programming language for web development", 
                source="Web Dev Guide", title="JavaScript Basics"),
        Document(id="", content="React is a JavaScript library for building user interfaces", 
                source="React Documentation", title="React Overview"),
        Document(id="", content="Node.js allows JavaScript to run on the server side", 
                source="Node.js Guide", title="Server-side JavaScript"),
        Document(id="", content="CSS is used for styling web pages and applications", 
                source="CSS Tutorial", title="CSS Fundamentals"),
        Document(id="", content="HTML is the markup language for creating web content", 
                source="HTML Reference", title="HTML Basics"),
        Document(id="", content="Docker containers help package and deploy applications", 
                source="DevOps Guide", title="Container Technology"),
        Document(id="", content="Kubernetes orchestrates containerized applications at scale", 
                source="K8s Documentation", title="Container Orchestration"),
        Document(id="", content="Git is a version control system for tracking code changes", 
                source="Git Handbook", title="Version Control"),
    ]
    
    await rag.add_documents_batch(tech_docs)
    
    # Test similarity searches
    search_queries = [
        "web development programming",
        "JavaScript frameworks and libraries",
        "container deployment tools",
        "version control systems"
    ]
    
    print("Testing similarity search:")
    
    for query in search_queries:
        print(f"\n--- Search: '{query}' ---")
        
        results = await rag.search_documents(query, top_k=3)
        
        print(f"Found {len(results)} similar documents:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.document.title}")
            print(f"     Similarity: {result.score:.3f}")
            print(f"     Content: {result.document.content[:80]}...")
    
    await rag.close()

async def demo_rag_with_metadata():
    """Demo: RAG with document metadata filtering"""
    print("\nDEMO 4: RAG WITH METADATA FILTERING")
    print("=" * 50)
    
    rag = BasicRAGSystem()
    await rag.initialize()
    
    # Add documents with rich metadata
    research_papers = [
        Document(
            id="",
            content="Deep learning neural networks have revolutionized computer vision tasks. "
                   "Convolutional Neural Networks (CNNs) are particularly effective for image classification.",
            source="Computer Vision Journal 2023",
            title="Deep Learning in Computer Vision",
            metadata={
                "category": "computer_vision",
                "year": 2023,
                "authors": ["Smith, J.", "Johnson, A."],
                "difficulty": "advanced",
                "field": "AI"
            }
        ),
        Document(
            id="",
            content="Natural Language Processing has advanced significantly with transformer models. "
                   "BERT and GPT have shown remarkable performance in various NLP tasks.",
            source="NLP Research Quarterly 2023",
            title="Transformer Models in NLP",
            metadata={
                "category": "nlp",
                "year": 2023,
                "authors": ["Brown, K.", "Davis, M."],
                "difficulty": "intermediate",
                "field": "AI"
            }
        ),
        Document(
            id="",
            content="Basic machine learning algorithms include linear regression, decision trees, and k-means clustering. "
                   "These foundational techniques are essential for understanding more complex models.",
            source="ML Fundamentals Textbook",
            title="Introduction to ML Algorithms",
            metadata={
                "category": "machine_learning",
                "year": 2022,
                "authors": ["Wilson, R."],
                "difficulty": "beginner",
                "field": "AI"
            }
        ),
        Document(
            id="",
            content="Quantum computing principles include superposition, entanglement, and quantum gates. "
                   "Quantum algorithms like Shor's algorithm show potential for cryptography applications.",
            source="Quantum Computing Review",
            title="Quantum Computing Fundamentals",
            metadata={
                "category": "quantum_computing",
                "year": 2023,
                "authors": ["Taylor, S.", "Anderson, L."],
                "difficulty": "advanced",
                "field": "Physics"
            }
        )
    ]
    
    await rag.add_documents_batch(research_papers)
    
    # Query with different contexts
    queries = [
        ("What are neural networks?", "Looking for computer vision information"),
        ("Explain machine learning basics", "Need beginner-friendly content"),
        ("How do transformers work?", "Interested in NLP research"),
        ("What is quantum computing?", "Exploring physics applications")
    ]
    
    print("Testing RAG with metadata-rich documents:")
    
    for query_text, context in queries:
        print(f"\n--- Query: {query_text} ---")
        print(f"Context: {context}")
        
        query = RAGQuery(
            id="",
            query_text=query_text,
            top_k=2,
            context={"user_context": context}
        )
        
        response = await rag.query(query)
        
        print(f"Retrieved documents:")
        for result in response.retrieved_documents:
            doc = result.document
            print(f"  - {doc.title} (score: {result.score:.3f})")
            print(f"    Category: {doc.metadata.get('category', 'N/A')}")
            print(f"    Difficulty: {doc.metadata.get('difficulty', 'N/A')}")
            print(f"    Year: {doc.metadata.get('year', 'N/A')}")
        
        print(f"Response: {response.generated_text[:200]}...")
    
    await rag.close()

async def demo_rag_performance():
    """Demo: RAG system performance analysis"""
    print("\nDEMO 5: RAG PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    rag = BasicRAGSystem()
    await rag.initialize()
    
    # Create larger knowledge base for performance testing
    import random
    
    topics = [
        "artificial intelligence", "machine learning", "data science", 
        "web development", "mobile apps", "cloud computing",
        "cybersecurity", "blockchain", "quantum computing", "robotics"
    ]
    
    print("Creating large knowledge base for performance testing...")
    
    large_kb = []
    for i in range(50):  # Create 50 documents
        topic = random.choice(topics)
        content = f"""
        This document discusses {topic} and its applications in modern technology.
        {topic.title()} has become increasingly important in recent years.
        Key concepts include implementation strategies, best practices, and future trends.
        Industries are adopting {topic} to improve efficiency and innovation.
        Research in {topic} continues to advance rapidly with new discoveries.
        The impact of {topic} extends across multiple sectors and applications.
        """
        
        doc = Document(
            id="",
            content=content,
            source=f"{topic.title()} Research Vol. {i+1}",
            title=f"Advanced {topic.title()} Concepts",
            metadata={"topic": topic, "doc_number": i+1}
        )
        large_kb.append(doc)
    
    # Batch add documents and measure time
    start_time = time.time()
    await rag.add_documents_batch(large_kb)
    indexing_time = time.time() - start_time
    
    print(f"Indexed {len(large_kb)} documents in {indexing_time:.3f}s")
    print(f"Average indexing time per document: {indexing_time/len(large_kb):.4f}s")
    
    # Performance test queries
    test_queries = [
        "artificial intelligence applications",
        "machine learning algorithms",
        "web development frameworks",
        "cloud computing benefits",
        "cybersecurity best practices"
    ]
    
    print(f"\nRunning performance tests with {len(test_queries)} queries...")
    
    total_response_time = 0
    all_retrieval_times = []
    all_generation_times = []
    
    for i, query_text in enumerate(test_queries, 1):
        query = RAGQuery(
            id="",
            query_text=query_text,
            top_k=5
        )
        
        response = await rag.query(query)
        
        total_response_time += response.response_time
        all_retrieval_times.append(response.retrieval_time)
        all_generation_times.append(response.generation_time)
        
        print(f"  Query {i}: {response.response_time:.3f}s "
              f"(retrieval: {response.retrieval_time:.3f}s, "
              f"generation: {response.generation_time:.3f}s)")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Average response time: {total_response_time/len(test_queries):.3f}s")
    print(f"  Average retrieval time: {sum(all_retrieval_times)/len(all_retrieval_times):.3f}s")
    print(f"  Average generation time: {sum(all_generation_times)/len(all_generation_times):.3f}s")
    print(f"  Knowledge base size: {len(large_kb)} documents")
    
    # System statistics
    stats = rag.get_system_stats()
    print(f"\nFinal System Statistics:")
    for key, value in stats['system_stats'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    await rag.close()

async def main():
    """
    Demonstrate Basic RAG Implementation for knowledge-augmented generation
    
    WHAT YOU'LL LEARN:
    ================
    1. How to build a complete RAG system from scratch
    2. How to implement document embedding and similarity search
    3. How to combine retrieval with text generation
    4. How to handle document chunking and metadata
    5. How to optimize RAG system performance
    
    REAL WORLD APPLICATIONS:
    =======================
    - Enterprise knowledge management systems
    - Intelligent document search and Q&A
    - Customer support chatbots with knowledge bases
    - Research paper analysis and summarization
    - Legal document analysis and case law research
    - Medical information systems and diagnosis support
    """
    
    print("BASIC RAG IMPLEMENTATION DEMONSTRATION")
    print("Showing how to build knowledge-augmented generation from the ground up!")
    
    await demo_basic_rag()
    await demo_document_chunking()
    await demo_similarity_search()
    await demo_rag_with_metadata()
    await demo_rag_performance()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ RAG combines retrieval with generation for factual responses")
    print("✓ Document embeddings enable semantic similarity search")
    print("✓ Chunking long documents improves retrieval granularity")
    print("✓ Metadata enhances document filtering and ranking")
    print("✓ Performance optimization is crucial for production systems")
    print("✓ Source attribution reduces hallucinations and improves trust")
    print("\nTHE POWER OF RAG SYSTEMS:")
    print("- Extends LLM knowledge beyond training data")
    print("- Provides real-time access to updated information")
    print("- Enables enterprise knowledge integration")
    print("- Reduces AI hallucinations through factual grounding")

if __name__ == "__main__":
    asyncio.run(main())
