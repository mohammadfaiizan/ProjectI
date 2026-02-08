# RAG Chatbot Project Description

## Problem Statement

The RAG (Retrieval-Augmented Generation) Chatbot project addresses the challenge of building an intelligent question-answering system that can answer questions based on user-uploaded documents. Traditional chatbots rely solely on their pre-trained knowledge, which limits their ability to answer questions about specific documents, recent information, or proprietary content. This project implements a RAG system that combines document retrieval with large language model generation to provide accurate, context-aware answers grounded in the provided documents.

The core problem is enabling a chatbot to:
- Process and understand multiple document formats (text files, PDFs, web pages)
- Store document content in a searchable format
- Retrieve relevant document chunks when answering questions
- Generate coherent answers that cite specific sources
- Maintain conversation context across multiple turns
- Handle queries that may require information from multiple documents

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Query Input / Chat History)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG_CHATBOT (Main Class)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    CHAT_MEMORY                            │  │
│  │  - Stores conversation history                            │  │
│  │  - Sliding window for context management                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  QUERY()       │
                    └────────┬───────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │   RETRIEVER      │      │   LLM (OpenAI)   │
    │  (Vector Store)  │      │   GPT-4 / GPT-3.5│
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             │                         │
             ▼                         │
    ┌──────────────────┐               │
    │  VECTOR_STORE    │               │
    │   (ChromaDB)     │               │
    └────────┬─────────┘               │
             │                         │
             │                         │
             ▼                         │
    ┌──────────────────┐               │
    │ EMBEDDING_MANAGER│               │
    │  (OpenAI API)    │               │
    └──────────────────┘               │
                                       │
                                       ▼
                            ┌──────────────────┐
                            │  GENERATED       │
                            │  RESPONSE        │
                            │  + SOURCES       │
                            └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ DOCUMENT_LOADER   │      │  TEXT_SPLITTER   │
    │ - Text files      │      │ - Recursive      │
    │ - PDF (mock)      │─────▶│   character      │
    │ - Web pages       │      │ - Overlap        │
    └──────────────────┘      └────────┬─────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │ EMBEDDING_MANAGER│
                            │  (Batch process) │
                            └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  VECTOR_STORE    │
                            │  (ChromaDB)      │
                            └──────────────────┘
```

## Component Breakdown

### Document_Loader
The Document_Loader class handles loading documents from various sources. It supports:
- Plain text files (.txt): Direct file reading
- PDF files: Mock implementation that simulates PDF text extraction
- Web pages: URL fetching and HTML content extraction using requests

This component normalizes different input formats into a consistent text representation that can be processed by downstream components.

### Text_Splitter
The Text_Splitter class implements recursive character-based text splitting with configurable chunk size and overlap. Key features:
- Recursive splitting: Attempts to split on paragraph boundaries first, then sentences, then characters
- Overlap management: Maintains context between chunks through overlapping text
- Chunk size control: Configurable maximum chunk size to balance retrieval granularity and context preservation

This ensures that related information stays together while allowing fine-grained retrieval of specific sections.

### Embedding_Manager
The Embedding_Manager class handles the conversion of text chunks into vector embeddings using OpenAI's embedding API. Features:
- Batch processing: Efficiently processes multiple chunks in a single API call
- Error handling: Manages API rate limits and failures gracefully
- Caching: Can cache embeddings to avoid redundant API calls

Embeddings enable semantic search where queries can find relevant content even if exact keywords don't match.

### Vector_Store
The Vector_Store class provides a wrapper around ChromaDB for storing and searching document embeddings. Capabilities:
- Document storage: Stores embeddings with associated metadata (source, chunk index, text)
- Similarity search: Finds most relevant chunks for a given query embedding
- Collection management: Creates and manages separate collections for different document sets
- Deletion: Supports removing documents or entire collections

This component enables fast similarity search over large document collections.

### Chat_Memory
The Chat_Memory class manages conversation history for multi-turn dialogues. Features:
- History storage: Maintains a list of user queries and assistant responses
- Sliding window: Limits context size to prevent token overflow
- Context formatting: Formats history for LLM context injection
- Reset capability: Allows clearing conversation history

This enables the chatbot to maintain context across multiple questions in a conversation.

### RAG_Chatbot
The RAG_Chatbot class orchestrates all components to provide the complete RAG functionality. Main methods:
- Ingest_Documents(): Processes documents through the full pipeline (load -> split -> embed -> store)
- Query(): Single-turn question answering with retrieval and generation
- Chat(): Multi-turn conversation with memory management

This is the main interface for users to interact with the RAG system.

## Data Flow

### Document Ingestion Flow

1. **Document Loading**: User provides document path or URL
   - Document_Loader reads the file or fetches web content
   - Raw text is extracted and normalized

2. **Text Splitting**: Raw text is divided into chunks
   - Text_Splitter applies recursive splitting strategy
   - Chunks are created with specified size and overlap
   - Each chunk is tagged with metadata (source, position)

3. **Embedding Generation**: Text chunks are converted to vectors
   - Embedding_Manager sends chunks to OpenAI embedding API
   - Vector embeddings are generated for each chunk
   - Embeddings are returned as numerical arrays

4. **Vector Storage**: Embeddings are stored in ChromaDB
   - Vector_Store creates or accesses a collection
   - Embeddings are stored with metadata (text, source, chunk_id)
   - Collection is persisted for future queries

### Query Processing Flow

1. **User Query**: User submits a question
   - Query text is received by RAG_Chatbot
   - Chat_Memory retrieves recent conversation history

2. **Query Embedding**: Query is converted to embedding vector
   - Embedding_Manager generates embedding for the query
   - Same embedding model is used as for documents

3. **Retrieval**: Relevant document chunks are found
   - Vector_Store performs similarity search using query embedding
   - Top-k most similar chunks are retrieved (default k=3-5)
   - Retrieved chunks include source metadata

4. **Context Assembly**: Retrieved chunks are formatted for LLM
   - Chunks are combined with conversation history
   - Context is formatted with clear source citations
   - Total context length is checked against model limits

5. **Generation**: LLM generates answer based on context
   - OpenAI API is called with system prompt, context, and query
   - Model generates response grounded in retrieved context
   - Response includes reasoning and source references

6. **Response Processing**: Answer is formatted and returned
   - Response text is extracted from API response
   - Sources are extracted from metadata
   - Chat_Memory stores query and response for future context

7. **User Display**: Formatted answer is presented
   - Answer text is displayed
   - Source citations are shown
   - Conversation history is updated

## Design Decisions

### Why ChromaDB?

ChromaDB was chosen as the vector database for several reasons:
- **Simplicity**: Easy to set up and use with minimal configuration
- **Python-native**: Built specifically for Python applications
- **Lightweight**: No external dependencies like separate database servers
- **Persistence**: Supports both in-memory and persistent storage
- **Metadata filtering**: Allows filtering by document metadata
- **Open source**: Free and actively maintained

Alternatives considered: Pinecone (cloud-based, requires API key), Weaviate (more complex setup), FAISS (no persistence by default).

### Why Recursive Splitting?

Recursive character splitting was chosen over fixed-size splitting because:
- **Semantic coherence**: Attempts to keep sentences and paragraphs together
- **Flexibility**: Falls back gracefully when natural boundaries aren't found
- **Context preservation**: Maintains better context than arbitrary character cuts
- **Overlap support**: Easily implements overlapping chunks for context continuity

The recursive approach tries paragraph boundaries first, then sentence boundaries, then falls back to character boundaries, ensuring chunks respect natural text structure when possible.

### Chunk Size Rationale

The default chunk size of 500-1000 characters was chosen based on:
- **Embedding model limits**: OpenAI embeddings work well with chunks of this size
- **Retrieval precision**: Smaller chunks allow more precise retrieval of specific information
- **Context window**: Balances between having enough context and avoiding irrelevant information
- **Overlap**: 100-200 character overlap ensures continuity between chunks

Larger chunks (2000+ characters) risk including irrelevant information, while smaller chunks (100-200 characters) may lose important context.

### Embedding Model Choice

OpenAI's text-embedding-ada-002 or text-embedding-3-small was chosen because:
- **Quality**: State-of-the-art semantic understanding
- **Consistency**: Same API as the generation model simplifies integration
- **Dimension**: 1536 dimensions provide good semantic granularity
- **Cost**: Reasonable pricing for production use
- **Reliability**: Well-maintained and widely used

Alternative embedding models (sentence-transformers, Cohere) were considered but OpenAI provides the best integration with the generation model.

### LLM Choice

OpenAI GPT-3.5-turbo or GPT-4 was selected for generation because:
- **Quality**: Excellent instruction following and reasoning
- **Context handling**: Can effectively use retrieved context
- **Citation ability**: Can naturally incorporate source references
- **API reliability**: Stable and well-documented API
- **Token limits**: Sufficient context window for RAG applications

## Prerequisites

### Required Packages

Install the following Python packages:

```bash
pip install openai chromadb requests
```

### Package Versions

- **openai**: >= 1.0.0 (for modern API compatibility)
- **chromadb**: >= 0.4.0 (for vector storage)
- **requests**: >= 2.28.0 (for web content fetching)

### API Keys

You will need an OpenAI API key:
1. Sign up at https://platform.openai.com/
2. Create an API key in your account settings
3. Set the environment variable: `export OPENAI_API_KEY="your-key-here"`
   - On Windows: `set OPENAI_API_KEY=your-key-here`
   - Or use a `.env` file with python-dotenv

### System Requirements

- Python 3.8 or higher
- Internet connection (for API calls and web content)
- 2GB+ RAM (for embedding processing)
- Disk space for ChromaDB persistence (varies by document collection size)

## How to Run

### Step 1: Install Dependencies

```bash
pip install openai chromadb requests
```

### Step 2: Set Up API Key

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here
```

### Step 3: Prepare Documents

Create a directory with sample documents:
- `documents/example1.txt` - Plain text file
- `documents/example2.txt` - Another text file
- Or use URLs for web content

### Step 4: Run the Implementation

```bash
python Implementation.py
```

### Step 5: Interactive Usage

The script will:
1. Initialize the RAG chatbot
2. Load sample documents (or you can specify your own)
3. Process documents through the ingestion pipeline
4. Enter interactive chat mode where you can ask questions
5. Type "quit" or "exit" to end the session

### Example Session

```
Loading documents...
Processing document: example1.txt
Processing document: example2.txt
Documents ingested successfully!

You can now ask questions. Type 'quit' to exit.
> What is the main topic of the documents?
[Answer with sources]

> Can you provide more details about X?
[Answer with conversation context]

> quit
Goodbye!
```

## Possible Extensions

### Multi-Modal Support

Extend the system to handle images, tables, and other non-text content:
- Add image embedding models (CLIP, OpenAI's vision models)
- Implement OCR for scanned documents
- Support structured data extraction (tables, forms)
- Multi-modal retrieval combining text and image embeddings

### Streaming Responses

Implement streaming for better user experience:
- Use OpenAI's streaming API for token-by-token generation
- Display partial responses as they're generated
- Reduce perceived latency for long responses
- Better handling of long-form answers

### Evaluation Framework

Add metrics to measure system performance:
- Retrieval accuracy (precision, recall)
- Answer quality (BLEU, ROUGE, semantic similarity)
- Latency metrics (retrieval time, generation time)
- User satisfaction tracking
- A/B testing framework for different configurations

### Advanced Retrieval

Improve retrieval quality:
- Hybrid search (combining keyword and semantic search)
- Re-ranking retrieved chunks with cross-encoder models
- Query expansion and reformulation
- Multi-query retrieval (generate multiple query variations)
- Filtering by document metadata (date, author, category)

### Production Features

Add features for production deployment:
- Authentication and user management
- Rate limiting and API quota management
- Caching for frequent queries
- Logging and monitoring
- Error handling and retry logic
- Database persistence for chat history
- Web interface (Flask/FastAPI + React)
- Docker containerization
- CI/CD pipeline

### Advanced Memory

Enhance conversation memory:
- Long-term memory storage in database
- Memory summarization for very long conversations
- User-specific memory (personalization)
- Memory search and retrieval
- Context compression techniques

### Document Management

Improve document handling:
- Incremental updates (add/remove documents without full re-indexing)
- Document versioning
- Access control and permissions
- Document preview and visualization
- Batch processing for large document collections
- Support for more file formats (Word, Excel, PowerPoint)
