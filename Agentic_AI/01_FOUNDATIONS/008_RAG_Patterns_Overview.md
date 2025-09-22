# RAG Patterns Overview: Retrieval-Augmented Generation Architectures

## Quick Reference: RAG Pattern Types

| Pattern | Use Case | Complexity | Best For |
|---------|----------|------------|----------|
| **Basic RAG** | Simple Q&A | Low | Documentation, FAQ |
| **Multi-Step RAG** | Complex queries | Medium | Research, analysis |
| **Agentic RAG** | Interactive exploration | High | Investigation, discovery |
| **Self-RAG** | Self-correcting | High | High accuracy needs |
| **Corrective RAG** | Error correction | Medium | Quality assurance |
| **Adaptive RAG** | Dynamic routing | High | Multi-domain systems |

---

## Core RAG Components

### **1. Basic RAG Architecture**
```python
class BasicRAG:
    def __init__(self):
        self.retriever = VectorRetriever()
        self.generator = LLMGenerator()
    
    def query(self, question):
        docs = self.retriever.retrieve(question)
        context = self.format_context(docs)
        return self.generator.generate(question, context)
```

### **2. Multi-Step RAG**
```python
class MultiStepRAG:
    def query(self, complex_question):
        sub_questions = self.decompose_question(complex_question)
        sub_answers = []
        for sub_q in sub_questions:
            docs = self.retriever.retrieve(sub_q)
            answer = self.generator.generate(sub_q, docs)
            sub_answers.append(answer)
        return self.synthesize_final_answer(sub_answers)
```

### **3. Agentic RAG**
```python
class AgenticRAG:
    def query(self, question):
        agent_response = self.agent.process(question)
        if agent_response.needs_retrieval:
            docs = self.retriever.retrieve(agent_response.search_query)
            return self.agent.generate_with_docs(question, docs)
        return agent_response.direct_answer
```

---

## RAG Enhancement Patterns

### **Self-RAG (Self-Reflective)**
- **Concept**: Agent evaluates its own responses and retrieves additional information if needed
- **Use Case**: High-accuracy requirements, fact-checking
- **Implementation**: Self-evaluation loops with retrieval triggers

### **Corrective RAG (CRAG)**
- **Concept**: Corrects retrieved information based on relevance scoring
- **Use Case**: Noisy or low-quality knowledge bases
- **Implementation**: Relevance filtering and web search fallback

### **Adaptive RAG**
- **Concept**: Dynamically chooses retrieval strategy based on query type
- **Use Case**: Multi-domain systems with varying query complexity
- **Implementation**: Query classification and strategy routing

---

## Vector Database Options

| Database | Strengths | Best For | Deployment |
|----------|-----------|----------|------------|
| **Pinecone** | Managed, scalable | Production, cloud | Cloud-only |
| **Weaviate** | Open source, flexible | Custom deployments | Self-hosted/Cloud |
| **Chroma** | Simple, lightweight | Development, small scale | Local/Self-hosted |
| **Qdrant** | Fast, Rust-based | Performance-critical | Self-hosted/Cloud |
| **FAISS** | Facebook, CPU/GPU | Research, experimentation | Local |

---

## Embedding Model Comparison

| Model | Dimensions | Performance | Use Case |
|-------|------------|-------------|----------|
| **OpenAI Ada-002** | 1536 | High | General purpose |
| **Sentence-BERT** | 384-768 | Medium | Lightweight applications |
| **E5-Large** | 1024 | High | Multilingual |
| **BGE-Large** | 1024 | High | Chinese + English |
| **Instructor** | Variable | High | Task-specific |

---

## Chunking Strategies

### **Fixed-Size Chunking**
```python
def fixed_chunk(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

### **Semantic Chunking**
```python
def semantic_chunk(text, model):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if should_start_new_chunk(current_chunk, sentence, model):
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)
    
    return chunks
```

### **Document Structure Chunking**
```python
def structure_chunk(document):
    chunks = []
    for section in document.sections:
        chunks.append({
            'content': section.text,
            'metadata': {
                'section': section.title,
                'level': section.level
            }
        })
    return chunks
```

---

## Query Enhancement Techniques

### **Query Expansion**
```python
def expand_query(original_query, expansion_model):
    similar_queries = expansion_model.generate_similar(original_query)
    expanded = f"{original_query} {' '.join(similar_queries)}"
    return expanded
```

### **Query Rewriting**
```python
def rewrite_query(query, rewriter_model):
    rewritten_variants = rewriter_model.rewrite(query, num_variants=3)
    return rewritten_variants
```

### **Hypothetical Document Embeddings (HyDE)**
```python
def hyde_query(question, llm):
    hypothetical_doc = llm.generate_document(question)
    query_embedding = embed(hypothetical_doc)
    return query_embedding
```

---

## Retrieval Strategies

### **Dense Retrieval**
- **Method**: Vector similarity search
- **Pros**: Semantic understanding, good for conceptual queries
- **Cons**: May miss exact keyword matches

### **Sparse Retrieval (BM25)**
- **Method**: Keyword-based scoring
- **Pros**: Excellent for exact matches, interpretable
- **Cons**: Limited semantic understanding

### **Hybrid Retrieval**
```python
class HybridRetriever:
    def retrieve(self, query, k=10):
        dense_results = self.dense_retriever.retrieve(query, k//2)
        sparse_results = self.sparse_retriever.retrieve(query, k//2)
        return self.combine_and_rank(dense_results, sparse_results)
```

### **Multi-Vector Retrieval**
```python
class MultiVectorRetriever:
    def retrieve(self, query):
        # Create multiple query representations
        embeddings = [
            self.summary_embedder.embed(query),
            self.question_embedder.embed(query),
            self.keyword_embedder.embed(query)
        ]
        
        all_results = []
        for embedding in embeddings:
            results = self.vector_store.search(embedding)
            all_results.extend(results)
        
        return self.deduplicate_and_rank(all_results)
```

---

## Context Management

### **Context Window Optimization**
```python
def optimize_context(retrieved_docs, max_tokens=4000):
    # Rank by relevance
    ranked_docs = rank_by_relevance(retrieved_docs)
    
    # Fit within token limit
    context = ""
    for doc in ranked_docs:
        if count_tokens(context + doc.content) <= max_tokens:
            context += doc.content + "\n\n"
        else:
            break
    
    return context
```

### **Contextual Compression**
```python
def compress_context(docs, query, compressor_model):
    compressed_docs = []
    for doc in docs:
        relevant_parts = compressor_model.extract_relevant(doc, query)
        compressed_docs.append(relevant_parts)
    return compressed_docs
```

---

## RAG Evaluation Metrics

### **Retrieval Metrics**
- **Recall@k**: Percentage of relevant documents retrieved
- **Precision@k**: Percentage of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG**: Normalized Discounted Cumulative Gain

### **Generation Metrics**
- **Faithfulness**: How well the answer is supported by retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Relevancy**: How relevant retrieved context is to the question

### **Implementation Example**
```python
class RAGEvaluator:
    def evaluate_retrieval(self, queries, retrieved_docs, ground_truth):
        metrics = {}
        metrics['recall@5'] = self.calculate_recall(retrieved_docs, ground_truth, k=5)
        metrics['precision@5'] = self.calculate_precision(retrieved_docs, ground_truth, k=5)
        metrics['mrr'] = self.calculate_mrr(retrieved_docs, ground_truth)
        return metrics
    
    def evaluate_generation(self, questions, answers, contexts):
        metrics = {}
        metrics['faithfulness'] = self.assess_faithfulness(answers, contexts)
        metrics['relevancy'] = self.assess_relevancy(answers, questions)
        return metrics
```

---

## Advanced RAG Patterns

### **Graph RAG**
- **Concept**: Uses knowledge graphs for enhanced retrieval
- **Benefits**: Relationship-aware retrieval, better context understanding
- **Implementation**: Graph databases + vector search

### **Temporal RAG**
- **Concept**: Time-aware retrieval and generation
- **Benefits**: Handles temporal queries, evolving knowledge
- **Implementation**: Time-stamped embeddings, temporal indexing

### **Multimodal RAG**
- **Concept**: Retrieval across text, images, audio, video
- **Benefits**: Rich, diverse knowledge sources
- **Implementation**: Multi-modal embeddings, cross-modal search

---

## RAG Pipeline Optimization

### **Indexing Optimization**
```python
# Hierarchical indexing for large datasets
class HierarchicalIndex:
    def __init__(self):
        self.coarse_index = CoarseGrainedIndex()
        self.fine_index = FineGrainedIndex()
    
    def search(self, query):
        coarse_results = self.coarse_index.search(query)
        fine_results = self.fine_index.search_within(query, coarse_results)
        return fine_results
```

### **Caching Strategy**
```python
class RAGCache:
    def __init__(self):
        self.query_cache = {}
        self.doc_cache = {}
    
    def get_or_retrieve(self, query):
        if query in self.query_cache:
            return self.query_cache[query]
        
        results = self.retriever.retrieve(query)
        self.query_cache[query] = results
        return results
```

### **Streaming RAG**
```python
async def streaming_rag_query(query):
    # Start retrieval
    retrieval_task = asyncio.create_task(retrieve_docs(query))
    
    # Begin generation as soon as first docs arrive
    async for doc_batch in retrieval_task:
        async for token in generate_with_docs(query, doc_batch):
            yield token
```

---

## Common Pitfalls and Solutions

### **1. Chunk Boundary Issues**
- **Problem**: Important information split across chunks
- **Solution**: Overlapping chunks, semantic chunking

### **2. Retrieved Context Not Used**
- **Problem**: LLM ignores retrieved information
- **Solution**: Better prompting, context highlighting

### **3. Hallucination Despite Retrieval**
- **Problem**: LLM generates false information
- **Solution**: Self-RAG, fact-checking layers

### **4. Poor Retrieval Quality**
- **Problem**: Irrelevant documents retrieved
- **Solution**: Better embeddings, query enhancement, hybrid retrieval

### **5. Context Window Overflow**
- **Problem**: Too much retrieved context
- **Solution**: Context compression, relevance ranking

---

## Quick Setup Guide

### **Basic RAG in 10 Lines**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Setup
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Query
result = qa_chain.run("Your question here")
```

This overview provides the essential patterns and considerations for implementing effective RAG systems across various use cases and complexity levels.
