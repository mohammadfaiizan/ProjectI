"""
LlamaIndex RAG Agent Examples
Demonstrates various RAG (Retrieval-Augmented Generation) patterns using LlamaIndex.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext,
    PromptTemplate,
    ServiceContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def Create_Sample_Documents() -> str:
    """
    Create sample documents for testing RAG functionality.
    Returns the directory path where documents are created.
    """
    temp_dir = tempfile.mkdtemp()
    doc_dir = Path(temp_dir) / "sample_docs"
    doc_dir.mkdir(exist_ok=True)
    
    # Create multiple sample documents
    documents = {
        "ai_history.txt": """
Artificial Intelligence (AI) has a rich history dating back to the 1950s.
The term was first coined by John McCarthy in 1956 at the Dartmouth Conference.
Early AI research focused on symbolic reasoning and problem-solving.
The field experienced several "AI winters" where funding and interest declined.
Modern AI has been revolutionized by machine learning and deep learning techniques.
Neural networks, inspired by biological neurons, form the foundation of deep learning.
Large language models like GPT have transformed natural language processing.
""",
        "machine_learning.txt": """
Machine Learning is a subset of artificial intelligence.
It enables systems to learn from data without explicit programming.
Supervised learning uses labeled training data to make predictions.
Unsupervised learning finds patterns in unlabeled data.
Reinforcement learning learns through trial and error with rewards.
Deep learning uses neural networks with multiple layers.
Common algorithms include linear regression, decision trees, and neural networks.
""",
        "llm_technology.txt": """
Large Language Models (LLMs) are transformer-based neural networks.
They are trained on vast amounts of text data using self-supervised learning.
LLMs can generate human-like text, answer questions, and perform various NLP tasks.
Popular models include GPT-4, Claude, Llama, and PaLM.
Fine-tuning adapts pre-trained models for specific tasks or domains.
Retrieval-Augmented Generation (RAG) combines LLMs with external knowledge bases.
RAG improves accuracy by grounding responses in retrieved documents.
""",
        "vector_databases.txt": """
Vector databases store and query high-dimensional vector embeddings.
They enable efficient similarity search for semantic retrieval.
Common vector databases include Pinecone, Weaviate, and Chroma.
Embeddings represent text as dense numerical vectors.
Similarity is measured using cosine similarity or Euclidean distance.
Vector databases are essential for RAG and semantic search applications.
Indexing strategies include HNSW, IVF, and flat indexes for different use cases.
"""
    }
    
    for filename, content in documents.items():
        file_path = doc_dir / filename
        file_path.write_text(content.strip())
    
    return str(doc_dir)


def Basic_RAG(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Basic RAG implementation using SimpleDirectoryReader and VectorStoreIndex.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional, can use environment variable)
    """
    print("\n" + "="*80)
    print("BASIC RAG EXAMPLE")
    print("="*80)
    
    # Set up LLM and embeddings
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    print(f"\nLoading documents from: {documents_dir}")
    documents = SimpleDirectoryReader(documents_dir).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Create vector store index
    print("\nCreating vector store index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Query examples
    queries = [
        "What is artificial intelligence?",
        "How do vector databases work?",
        "What are the main types of machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Custom_Prompt_RAG(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    RAG with custom prompt templates for QA and refinement.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("CUSTOM PROMPT RAG EXAMPLE")
    print("="*80)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Define custom QA prompt template
    qa_prompt_template = PromptTemplate(
        """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the question.
Question: {query_str}
Answer: Provide a detailed, well-structured answer based solely on the context provided.
If the context does not contain enough information, state that clearly.
"""
    )
    
    # Define custom refine prompt template
    refine_prompt_template = PromptTemplate(
        """The original question is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the question.
If the context is not useful, return the original answer.
Refined Answer: """
    )
    
    # Create query engine with custom prompts
    query_engine = index.as_query_engine(
        text_qa_template=qa_prompt_template,
        refine_template=refine_prompt_template,
    )
    
    # Query examples
    queries = [
        "Explain the history of AI and its major developments.",
        "Compare supervised and unsupervised learning approaches."
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Chat_Engine_RAG(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Conversational RAG using CondensePlusContextChatEngine for multi-turn conversations.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("CHAT ENGINE RAG EXAMPLE")
    print("="*80)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create chat engine
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=3),
        llm=llm,
        verbose=True
    )
    
    # Simulate multi-turn conversation
    conversation_turns = [
        "What is machine learning?",
        "What are its main types?",
        "How does deep learning relate to it?",
        "Can you give examples of deep learning applications?"
    ]
    
    print("\nStarting conversational RAG session...")
    for i, query in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {query}")
        response = chat_engine.chat(query)
        print(f"Assistant: {response}")
        print("-" * 80)


def Sub_Question_Query(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Decompose complex queries into sub-questions using SubQuestionQueryEngine.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("SUB-QUESTION QUERY ENGINE EXAMPLE")
    print("="*80)
    
    try:
        from llama_index.core.query_engine import SubQuestionQueryEngine
        from llama_index.core.tools import QueryEngineTool
    except ImportError:
        print("SubQuestionQueryEngine not available in this version of LlamaIndex")
        return
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create base query engine
    base_query_engine = index.as_query_engine()
    
    # Create query engine tool
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=base_query_engine,
        description="Useful for answering questions about AI, machine learning, LLMs, and vector databases."
    )
    
    # Create sub-question query engine
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[query_engine_tool],
        llm=llm,
        verbose=True
    )
    
    # Complex queries that benefit from decomposition
    complex_queries = [
        "What is the relationship between AI, machine learning, and deep learning?",
        "How do vector databases support RAG systems and what are their key features?",
        "Compare the history of AI with modern LLM technology and their applications."
    ]
    
    for query in complex_queries:
        print(f"\nComplex Query: {query}")
        response = sub_question_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Agent_With_RAG_Tool(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    FunctionCallingAgentWorker with QueryEngineTool wrapping a RAG index.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("AGENT WITH RAG TOOL EXAMPLE")
    print("="*80)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Create query engine tool
    rag_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=(
            "Use this tool to answer questions about artificial intelligence, "
            "machine learning, large language models, and vector databases. "
            "Input should be a complete question."
        )
    )
    
    # Create agent worker
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        system_prompt="You are a helpful AI assistant that answers questions using retrieved knowledge."
    )
    
    # Create agent
    agent = agent_worker.as_agent()
    
    # Test queries
    queries = [
        "What is the difference between AI and machine learning?",
        "Explain how RAG improves LLM responses.",
        "What are the advantages of using vector databases?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.chat(query)
        print(f"Agent Response: {response}")
        print("-" * 80)


def Multi_Document_Agent(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Agent that queries across multiple separate document indices.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("MULTI-DOCUMENT AGENT EXAMPLE")
    print("="*80)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load all documents
    all_documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Separate documents by topic (simulate multiple document collections)
    ai_docs = [doc for doc in all_documents if "ai_history" in doc.metadata.get("file_path", "").lower()]
    ml_docs = [doc for doc in all_documents if "machine_learning" in doc.metadata.get("file_path", "").lower()]
    llm_docs = [doc for doc in all_documents if "llm_technology" in doc.metadata.get("file_path", "").lower()]
    vector_docs = [doc for doc in all_documents if "vector_databases" in doc.metadata.get("file_path", "").lower()]
    
    # Create separate indices
    print("\nCreating separate indices for different topics...")
    ai_index = VectorStoreIndex.from_documents(ai_docs) if ai_docs else None
    ml_index = VectorStoreIndex.from_documents(ml_docs) if ml_docs else None
    llm_index = VectorStoreIndex.from_documents(llm_docs) if llm_docs else None
    vector_index = VectorStoreIndex.from_documents(vector_docs) if vector_docs else None
    
    # Create query engine tools for each index
    tools = []
    
    if ai_index:
        ai_tool = QueryEngineTool.from_defaults(
            query_engine=ai_index.as_query_engine(),
            description="Useful for questions about AI history, development, and early research."
        )
        tools.append(ai_tool)
    
    if ml_index:
        ml_tool = QueryEngineTool.from_defaults(
            query_engine=ml_index.as_query_engine(),
            description="Useful for questions about machine learning algorithms, types, and techniques."
        )
        tools.append(ml_tool)
    
    if llm_index:
        llm_tool = QueryEngineTool.from_defaults(
            query_engine=llm_index.as_query_engine(),
            description="Useful for questions about large language models, transformers, and RAG."
        )
        tools.append(llm_tool)
    
    if vector_index:
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_index.as_query_engine(),
            description="Useful for questions about vector databases, embeddings, and similarity search."
        )
        tools.append(vector_tool)
    
    # Create agent with multiple tools
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are an expert AI assistant with access to multiple knowledge bases. "
            "Use the appropriate tool(s) to answer questions accurately. "
            "You can combine information from multiple sources when needed."
        )
    )
    
    agent = agent_worker.as_agent()
    
    # Test queries that may require multiple indices
    queries = [
        "What is the history of AI and how has it evolved to include modern LLMs?",
        "How do machine learning and vector databases work together in RAG systems?",
        "Explain the complete pipeline from neural networks to RAG applications."
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.chat(query)
        print(f"Agent Response: {response}")
        print("-" * 80)


def main():
    """
    Main function to run all RAG examples.
    """
    print("LlamaIndex RAG Agent Examples")
    print("="*80)
    
    # Get API key from environment or use None
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\nWarning: OPENAI_API_KEY not set in environment.")
        print("Please set it or pass it as a parameter to the functions.")
        print("Continuing with examples (will fail if API key is required)...\n")
    
    # Create sample documents
    print("Creating sample documents...")
    documents_dir = Create_Sample_Documents()
    print(f"Sample documents created at: {documents_dir}\n")
    
    try:
        # Run all examples
        Basic_RAG(documents_dir, api_key)
        Custom_Prompt_RAG(documents_dir, api_key)
        Chat_Engine_RAG(documents_dir, api_key)
        Sub_Question_Query(documents_dir, api_key)
        Agent_With_RAG_Tool(documents_dir, api_key)
        Multi_Document_Agent(documents_dir, api_key)
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(documents_dir)
            print(f"\nCleaned up temporary directory: {documents_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")


if __name__ == "__main__":
    main()
