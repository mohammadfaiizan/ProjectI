"""
Sample input module for RAG Chatbot.
Contains sample documents, queries, and demonstration functions.
"""

from Config import LLM_Config, Vector_Store_Config, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from Tools import Document_Processor, Vector_Store_Manager
from Agent import RAG_Agent
from langchain_core.documents import Document
from typing import List, Dict, Any


# Sample documents about AI topics
SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "Large Language Models (LLMs) represent a significant breakthrough "
            "in artificial intelligence and natural language processing. These models "
            "are trained on vast amounts of text data using transformer architectures, "
            "enabling them to understand and generate human-like text. The transformer "
            "architecture, introduced in 2017, uses self-attention mechanisms to process "
            "sequences of tokens in parallel rather than sequentially. This innovation "
            "allows models to capture long-range dependencies in text more effectively "
            "than previous recurrent neural network approaches. Modern LLMs like GPT-4, "
            "Claude, and LLaMA contain billions or even trillions of parameters, making "
            "them capable of performing a wide range of language tasks including text "
            "generation, translation, summarization, question answering, and code generation. "
            "The training process involves predicting the next token in a sequence given "
            "the previous context, which helps the model learn statistical patterns and "
            "semantic relationships in language. Fine-tuning techniques allow these models "
            "to be adapted for specific domains or tasks, improving their performance on "
            "targeted applications. However, LLMs also face challenges including hallucination, "
            "where they generate plausible but incorrect information, and the need for "
            "substantial computational resources for training and inference."
        ),
        metadata={"source": "document_llm.txt", "topic": "LLMs"}
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) is a powerful technique that combines "
            "the capabilities of large language models with external knowledge retrieval "
            "systems. The RAG architecture addresses a key limitation of LLMs: their "
            "knowledge is frozen at training time and may not include the most recent "
            "information or domain-specific details. In a RAG system, when a user asks "
            "a question, the system first retrieves relevant documents or passages from "
            "a knowledge base using semantic search. These retrieved documents are then "
            "provided as context to the LLM, which generates an answer based on both "
            "its pre-trained knowledge and the retrieved information. The retrieval process "
            "typically uses dense vector embeddings to find semantically similar content, "
            "often implemented with vector databases like ChromaDB, Pinecone, or Weaviate. "
            "The generation step uses the retrieved context along with the user's question "
            "to produce accurate, up-to-date responses. RAG systems offer several advantages: "
            "they can access current information beyond the model's training cutoff, reduce "
            "hallucination by grounding responses in retrieved documents, and enable domain-specific "
            "applications without expensive fine-tuning. Common RAG architectures include "
            "naive RAG, which retrieves once and generates, and advanced RAG with query "
            "rewriting, multi-step retrieval, and re-ranking. The effectiveness of RAG "
            "depends on the quality of the knowledge base, the retrieval mechanism, and "
            "the prompt engineering used to combine context with queries."
        ),
        metadata={"source": "document_rag.txt", "topic": "RAG"}
    ),
    Document(
        page_content=(
            "LangChain is a comprehensive framework designed to simplify the development "
            "of applications powered by large language models. It provides abstractions "
            "and tools that make it easier to build complex LLM applications including "
            "RAG systems, agents, and chains. The framework consists of several core "
            "components: LangChain Core, which provides the foundational abstractions "
            "and interfaces; LangChain Community, containing community-contributed "
            "integrations with various tools and services; and LangChain LangGraph, "
            "which enables building stateful, multi-actor applications with graphs. "
            "Key concepts in LangChain include chains, which combine multiple components "
            "to create workflows, and agents, which can use tools and make decisions "
            "dynamically. The framework supports various document loaders for ingesting "
            "data from different sources like PDFs, web pages, and databases. LangChain "
            "also provides text splitters for chunking documents, vector store integrations "
            "for semantic search, and retrieval mechanisms. The LangChain Expression Language "
            "(LCEL) offers a declarative way to compose chains using pipe operators, making "
            "it intuitive to build complex workflows. LangGraph extends LangChain by providing "
            "a graph-based approach to building agents, allowing developers to define nodes "
            "and edges that represent different steps in an agent's decision-making process. "
            "This enables building sophisticated multi-step reasoning systems, conversational "
            "agents, and workflows with cycles and conditional logic. LangChain's modular "
            "design allows developers to mix and match components to suit their specific needs."
        ),
        metadata={"source": "document_langchain.txt", "topic": "LangChain"}
    ),
    Document(
        page_content=(
            "Vector databases are specialized storage systems designed to efficiently store "
            "and query high-dimensional vector embeddings. These databases are essential "
            "for semantic search, recommendation systems, and RAG applications. Unlike "
            "traditional relational databases that store structured data, vector databases "
            "optimize for similarity search operations, allowing users to find vectors "
            "that are most similar to a query vector based on distance metrics like cosine "
            "similarity, Euclidean distance, or dot product. Popular vector databases include "
            "ChromaDB, which is lightweight and easy to use; Pinecone, a managed cloud service "
            "with high performance; Weaviate, which combines vector search with traditional "
            "filtering; and Milvus, an open-source system designed for large-scale deployments. "
            "Vector databases use various indexing techniques such as approximate nearest neighbor "
            "(ANN) algorithms including HNSW (Hierarchical Navigable Small World), IVF (Inverted "
            "File Index), and LSH (Locality Sensitive Hashing) to enable fast similarity search "
            "even with millions or billions of vectors. The embedding process converts text, "
            "images, or other data into dense vector representations using models like OpenAI's "
            "text-embedding-ada-002 or open-source alternatives. These embeddings capture semantic "
            "meaning, allowing the database to find conceptually similar content even when the "
            "exact words differ. Vector databases typically support metadata filtering, enabling "
            "hybrid search that combines semantic similarity with traditional attribute-based "
            "queries. They are crucial for building production RAG systems that need to quickly "
            "retrieve relevant context from large knowledge bases."
        ),
        metadata={"source": "document_vectordb.txt", "topic": "Vector Databases"}
    )
]


# Sample queries covering different topics
SAMPLE_QUERIES = [
    "What are Large Language Models and how do they work?",
    "Explain the transformer architecture used in LLMs.",
    "What is Retrieval-Augmented Generation and what are its benefits?",
    "How does RAG address the limitations of LLMs?",
    "What is LangChain and what are its main components?",
    "How does LangGraph differ from traditional LangChain chains?",
    "What are vector databases and why are they important for RAG?",
    "Compare different vector database options like ChromaDB and Pinecone."
]


# Multi-turn conversation that builds on previous context
MULTI_TURN_CONVERSATION = [
    "What is RAG?",
    "How does it use vector databases?",
    "Can you explain how LangChain helps build RAG systems?",
    "What are the main advantages of using RAG over fine-tuning?",
    "How would I implement a RAG system using LangChain and ChromaDB?"
]


def Run_Samples() -> None:
    """
    Run sample demonstrations with sample documents and queries.
    """
    print("="*70)
    print("RAG Chatbot - Sample Demonstration")
    print("="*70)
    
    # Setup system
    print("\n1. Setting up RAG system...")
    llm_config = LLM_Config(model_name="gpt-4o-mini", temperature=0.0)
    llm = llm_config.Get_LLM()
    embeddings = llm_config.Get_Embeddings()
    
    vector_config = Vector_Store_Config(
        collection_name="sample_rag_documents",
        persist_directory="./sample_chroma_db"
    )
    
    doc_processor = Document_Processor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process sample documents
    print("\n2. Processing sample documents...")
    chunks = doc_processor.Split_Documents(SAMPLE_DOCUMENTS)
    print(f"   Created {len(chunks)} chunks from {len(SAMPLE_DOCUMENTS)} documents")
    
    # Create vector store
    print("\n3. Creating vector store...")
    vector_manager = Vector_Store_Manager(
        collection_name=vector_config.Get_Collection_Name(),
        persist_directory=vector_config.Get_Persist_Directory(),
        embeddings=embeddings
    )
    vector_manager.Create_Store(chunks)
    
    # Create retriever
    print("\n4. Creating retriever...")
    retriever = vector_manager.Get_Retriever(search_kwargs={"k": TOP_K})
    
    # Create agent
    print("\n5. Building RAG agent...")
    agent = RAG_Agent(llm=llm, retriever=retriever)
    agent.Build_Graph()
    
    # Run sample queries
    print("\n" + "="*70)
    print("Running Sample Queries")
    print("="*70 + "\n")
    
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"Query {i}: {query}")
        print("-"*70)
        try:
            result = agent.Query(query)
            print(f"Answer:\n{result['answer']}\n")
            if result.get("sources"):
                unique_sources = list(set(result["sources"]))
                print(f"Sources: {', '.join(unique_sources)}")
            print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f"Error: {e}\n")
            print("="*70 + "\n")
    
    # Run multi-turn conversation
    print("\n" + "="*70)
    print("Multi-Turn Conversation Demo")
    print("="*70 + "\n")
    
    agent.Reset_History()
    
    for i, query in enumerate(MULTI_TURN_CONVERSATION, 1):
        print(f"Turn {i} - User: {query}")
        print("-"*70)
        try:
            result = agent.Query(query, agent.chat_history)
            print(f"Assistant: {result['answer']}\n")
            if result.get("sources"):
                unique_sources = list(set(result["sources"]))
                print(f"Sources: {', '.join(unique_sources)}")
            print("\n" + "="*70 + "\n")
            
            # Update history for next turn
            agent.chat_history.append({"human": query})
            agent.chat_history.append({"ai": result["answer"]})
        except Exception as e:
            print(f"Error: {e}\n")
            print("="*70 + "\n")
    
    print("\nSample demonstration complete!")
    print(f"Vector store persisted at: {vector_config.Get_Persist_Directory()}")


if __name__ == "__main__":
    Run_Samples()
