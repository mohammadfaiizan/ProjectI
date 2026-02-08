"""
RAG Chatbot Implementation
A complete Retrieval-Augmented Generation chatbot using OpenAI and ChromaDB.
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from openai import OpenAI
import chromadb
from chromadb.config import Settings


class Document_Loader:
    """Loads documents from various sources: text files, PDFs, web pages."""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.text'}
    
    def load_text_file(self, file_path: str) -> str:
        """Load content from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def load_pdf_mock(self, file_path: str) -> str:
        """Mock PDF loader - simulates PDF text extraction."""
        # In production, use libraries like PyPDF2, pdfplumber, or pymupdf
        # This is a placeholder that reads text files as if they were PDFs
        print(f"[Mock] Loading PDF: {file_path}")
        try:
            # Simulate PDF extraction by reading as text
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Mock: Remove some characters to simulate PDF extraction artifacts
            content = content.replace('\x00', '')
            return content
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")
    
    def load_web_page(self, url: str) -> str:
        """Fetch and extract text content from a web page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Basic HTML text extraction (remove tags)
            html_content = response.text
            # Simple tag removal - in production, use BeautifulSoup
            text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
        except requests.RequestException as e:
            raise Exception(f"Error fetching URL {url}: {str(e)}")
    
    def load_document(self, source: str) -> Tuple[str, str]:
        """
        Load a document from various sources.
        Returns: (content, source_type)
        """
        source_lower = source.lower()
        
        # Check if it's a URL
        if source.startswith('http://') or source.startswith('https://'):
            content = self.load_web_page(source)
            return content, 'web'
        
        # Check file extension
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            content = self.load_pdf_mock(source)
            return content, 'pdf'
        elif extension in self.supported_extensions:
            content = self.load_text_file(source)
            return content, 'text'
        else:
            # Try as text file anyway
            try:
                content = self.load_text_file(source)
                return content, 'text'
            except Exception:
                raise ValueError(f"Unsupported file type: {extension}")


class Text_Splitter:
    """Recursive character-based text splitter with overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a separator."""
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        return [s for s in splits if s.strip()]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits back together, respecting chunk size."""
        separator_len = len(separator) if separator else 0
        docs = []
        current_doc = []
        current_len = 0
        
        for split in splits:
            split_len = len(split)
            if current_len + split_len + separator_len > self.chunk_size and current_doc:
                # Save current doc
                doc = separator.join(current_doc)
                if doc.strip():
                    docs.append(doc)
                # Start new doc with overlap
                if self.chunk_overlap > 0 and current_doc:
                    # Take last part of current_doc for overlap
                    overlap_text = separator.join(current_doc[-1:])
                    if len(overlap_text) > self.chunk_overlap:
                        overlap_text = overlap_text[-self.chunk_overlap:]
                    current_doc = [overlap_text] if overlap_text.strip() else []
                    current_len = len(overlap_text) + separator_len
                else:
                    current_doc = []
                    current_len = 0
            
            current_doc.append(split)
            current_len += split_len + separator_len
        
        # Add final doc
        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
        
        return docs
    
    def split_text(self, text: str) -> List[str]:
        """
        Recursively split text into chunks.
        Tries: paragraphs -> sentences -> characters
        """
        # Try splitting by paragraphs first
        paragraphs = self._split_by_separator(text, '\n\n')
        if len(paragraphs) > 1:
            chunks = []
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    chunks.append(para)
                else:
                    # Paragraph too large, split by sentences
                    sentences = self._split_by_separator(para, '. ')
                    if len(sentences) > 1:
                        merged = self._merge_splits(sentences, '. ')
                        chunks.extend(merged)
                    else:
                        # Fall back to character splitting
                        chunks.extend(self._split_by_characters(para))
            return chunks
        
        # Try splitting by sentences
        sentences = self._split_by_separator(text, '. ')
        if len(sentences) > 1:
            return self._merge_splits(sentences, '. ')
        
        # Fall back to character splitting
        return self._split_by_characters(text)
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by characters when other methods fail."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """
        Split multiple documents into chunks with metadata.
        
        Args:
            documents: List of (content, source) tuples
        
        Returns:
            List of dicts with 'text', 'source', 'chunk_index' keys
        """
        all_chunks = []
        
        for doc_idx, (content, source) in enumerate(documents):
            chunks = self.split_text(content)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_index': chunk_idx,
                    'document_index': doc_idx
                })
        
        return all_chunks


class Embedding_Manager:
    """Manages OpenAI embeddings for text chunks."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding manager.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dimension = 1536  # Default for text-embedding-3-small
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
        
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise Exception(f"Error generating batch embeddings: {str(e)}")
        
        return all_embeddings


class Vector_Store:
    """ChromaDB wrapper for vector storage and retrieval."""
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist data (None for in-memory)
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts for each chunk
        """
        ids = [f"chunk_{i}" for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of dicts with 'text', 'source', 'distance', 'metadata' keys
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'unknown'),
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'metadata': results['metadatas'][0][i]
                })
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection.name)
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()


class Chat_Memory:
    """Manages conversation history with sliding window."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat memory.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_exchange(self, query: str, response: str) -> None:
        """Add a query-response pair to history."""
        self.history.append({
            'query': query,
            'response': response
        })
        
        # Enforce sliding window
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Format history as context string for LLM."""
        if not self.history:
            return ""
        
        context_parts = []
        for exchange in self.history:
            context_parts.append(f"User: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response']}")
        
        return "\n".join(context_parts)
    
    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """Get the most recent n queries."""
        return [ex['query'] for ex in self.history[-n:]]


class RAG_Chatbot:
    """Main RAG chatbot class that orchestrates all components."""
    
    def __init__(self, collection_name: str = "rag_documents", 
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-3.5-turbo",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG chatbot.
        
        Args:
            collection_name: ChromaDB collection name
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
        """
        self.document_loader = Document_Loader()
        self.text_splitter = Text_Splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_manager = Embedding_Manager(model=embedding_model)
        self.vector_store = Vector_Store(collection_name=collection_name)
        self.chat_memory = Chat_Memory()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.llm_client = OpenAI(api_key=api_key)
        self.llm_model = llm_model
    
    def ingest_documents(self, document_sources: List[str]) -> None:
        """
        Process and store documents in the vector store.
        
        Args:
            document_sources: List of file paths or URLs
        """
        print(f"Loading {len(document_sources)} documents...")
        
        # Load documents
        documents = []
        for source in document_sources:
            try:
                print(f"Processing document: {source}")
                content, source_type = self.document_loader.load_document(source)
                documents.append((content, source))
                print(f"  Loaded {len(content)} characters from {source_type}")
            except Exception as e:
                print(f"  Error loading {source}: {str(e)}")
                continue
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
        
        # Split documents
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_manager.embed_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        print("Storing in vector database...")
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                'document_index': chunk['document_index']
            }
            for chunk in chunks
        ]
        self.vector_store.add_documents(texts, embeddings, metadatas)
        print(f"Stored {len(chunks)} chunks in vector database")
    
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, any]:
        """
        Answer a single query using RAG.
        
        Args:
            user_query: User's question
            top_k: Number of document chunks to retrieve
        
        Returns:
            Dict with 'answer', 'sources', 'retrieved_chunks' keys
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(user_query)
        
        # Retrieve relevant chunks
        retrieved = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not retrieved:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'retrieved_chunks': []
            }
        
        # Format context
        context_parts = []
        sources = []
        for i, chunk in enumerate(retrieved):
            context_parts.append(f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}")
            sources.append(chunk['source'])
        
        context = "\n\n".join(context_parts)
        
        # Get conversation history
        history_context = self.chat_memory.get_context()
        
        # Generate answer
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from documents.
Always cite your sources using [Source X] notation when referencing information from the documents.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but thorough."""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if history_context:
            messages.append({
                "role": "system",
                "content": f"Previous conversation:\n{history_context}"
            })
        
        messages.append({
            "role": "system",
            "content": f"Context from documents:\n{context}"
        })
        
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Update memory
            self.chat_memory.add_exchange(user_query, answer)
            
            return {
                'answer': answer,
                'sources': list(set(sources)),
                'retrieved_chunks': retrieved
            }
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def chat(self) -> None:
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("RAG Chatbot - Interactive Mode")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'reset' to clear conversation history")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'reset':
                    self.chat_memory.reset()
                    print("Conversation history cleared.")
                    continue
                
                # Get answer
                result = self.query(user_input)
                
                print(f"\nAssistant: {result['answer']}")
                
                if result['sources']:
                    print(f"\nSources: {', '.join(set(result['sources']))}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


def main():
    """Main function demonstrating the RAG chatbot."""
    print("Initializing RAG Chatbot...")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize chatbot
    chatbot = RAG_Chatbot(
        collection_name="demo_rag",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Sample documents (create these files or modify paths)
    sample_documents = [
        "sample_document1.txt",
        "sample_document2.txt"
    ]
    
    # Check if sample documents exist, if not create them
    for doc_path in sample_documents:
        if not os.path.exists(doc_path):
            print(f"Creating sample document: {doc_path}")
            with open(doc_path, 'w', encoding='utf-8') as f:
                if "document1" in doc_path:
                    f.write("""Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Instead of being explicitly programmed, ML systems learn patterns from data.

Deep Learning is a further subset of machine learning that uses neural networks with multiple layers to learn representations of data. Deep learning has been particularly successful in areas like image recognition, natural language processing, and game playing.

Key applications of AI include:
- Natural language processing for chatbots and translation
- Computer vision for image and video analysis
- Recommendation systems for e-commerce and content platforms
- Autonomous vehicles and robotics
- Healthcare diagnostics and drug discovery""")
                else:
                    f.write("""Large Language Models and RAG Systems

Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text. Examples include GPT-4, Claude, and LLaMA. These models can perform a wide range of language tasks including translation, summarization, question answering, and creative writing.

Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by combining them with external knowledge retrieval. Instead of relying solely on pre-trained knowledge, RAG systems:

1. Retrieve relevant information from external sources (documents, databases, web)
2. Augment the LLM's context with this retrieved information
3. Generate responses that are grounded in the retrieved content

RAG systems are particularly useful for:
- Answering questions about specific documents or knowledge bases
- Providing up-to-date information beyond the LLM's training cutoff
- Reducing hallucinations by grounding responses in retrieved facts
- Enabling domain-specific applications without fine-tuning

The typical RAG pipeline involves document ingestion, chunking, embedding, vector storage, retrieval, and generation.""")
    
    try:
        # Ingest documents
        chatbot.ingest_documents(sample_documents)
        print("\nDocuments ingested successfully!\n")
        
        # Enter interactive chat
        chatbot.chat()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
