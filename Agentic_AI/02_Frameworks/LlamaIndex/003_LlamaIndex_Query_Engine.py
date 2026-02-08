"""
LlamaIndex Query Engine Examples
Demonstrates various query engine patterns and configurations using LlamaIndex.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Any

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext,
    QueryBundle,
)
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RouterQueryEngine,
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import (
    NodeWithScore,
    MetadataMode,
    TextNode,
    QueryType,
)
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    EvaluationResult,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def Create_Sample_Documents() -> str:
    """
    Create sample documents for testing query engine functionality.
    Returns the directory path where documents are created.
    """
    temp_dir = tempfile.mkdtemp()
    doc_dir = Path(temp_dir) / "sample_docs"
    doc_dir.mkdir(exist_ok=True)
    
    # Create multiple sample documents
    documents = {
        "python_basics.txt": """
Python is a high-level programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
The language uses dynamic typing and garbage collection.
Python has a large standard library and extensive ecosystem of third-party packages.
It is widely used in web development, data science, artificial intelligence, and automation.
Python's syntax emphasizes code readability with significant whitespace.
""",
        "data_structures.txt": """
Data structures are ways of organizing and storing data in computer memory.
Lists are ordered, mutable collections that can contain elements of different types.
Dictionaries are key-value pairs that provide fast lookup by key.
Tuples are immutable ordered sequences, useful for fixed collections.
Sets are unordered collections of unique elements.
Arrays and lists differ in that arrays typically store homogeneous data types.
Trees and graphs are hierarchical data structures for representing relationships.
Hash tables provide O(1) average-case lookup time using hash functions.
""",
        "algorithms.txt": """
Algorithms are step-by-step procedures for solving problems.
Sorting algorithms arrange elements in a specific order (ascending or descending).
Common sorting algorithms include quicksort, mergesort, and heapsort.
Search algorithms find elements within data structures.
Binary search works on sorted arrays and has O(log n) time complexity.
Graph algorithms include depth-first search (DFS) and breadth-first search (BFS).
Dynamic programming solves problems by breaking them into overlapping subproblems.
Greedy algorithms make locally optimal choices at each step.
""",
        "software_engineering.txt": """
Software engineering is the application of engineering principles to software development.
The software development lifecycle includes requirements, design, implementation, testing, and maintenance.
Version control systems like Git track changes to code over time.
Testing ensures software quality through unit tests, integration tests, and system tests.
Code reviews help maintain code quality and share knowledge among team members.
Design patterns provide reusable solutions to common software design problems.
Agile methodologies emphasize iterative development and customer collaboration.
Documentation is crucial for maintaining and understanding software systems.
"""
    }
    
    for filename, content in documents.items():
        file_path = doc_dir / filename
        file_path.write_text(content.strip())
    
    return str(doc_dir)


def Vector_Store_Query_Engine(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Basic vector search QA with VectorStoreIndex.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("VECTOR STORE QUERY ENGINE EXAMPLE")
    print("="*80)
    
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
        "What is Python?",
        "Explain data structures in programming.",
        "What are the main sorting algorithms?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Summary_Index_Engine(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    SummaryIndex to summarize all documents.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("SUMMARY INDEX ENGINE EXAMPLE")
    print("="*80)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create summary index
    print("\nCreating summary index...")
    summary_index = SummaryIndex.from_documents(documents)
    
    # Create query engine
    query_engine = summary_index.as_query_engine(
        response_mode=ResponseMode.TREE_SUMMARIZE
    )
    
    # Query examples
    queries = [
        "Summarize all the documents.",
        "What are the main topics covered across all documents?",
        "Provide an overview of programming concepts discussed."
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Knowledge_Graph_Engine(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    KnowledgeGraphIndex for entity/relation extraction and graph queries.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH ENGINE EXAMPLE")
    print("="*80)
    
    try:
        from llama_index.core import KnowledgeGraphIndex
        from llama_index.core.graph_stores import SimpleGraphStore
        from llama_index.core.storage.storage_context import StorageContext
    except ImportError:
        print("KnowledgeGraphIndex not available in this version of LlamaIndex")
        return
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Create graph store
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # Create knowledge graph index
    print("\nCreating knowledge graph index...")
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        llm=llm,
    )
    
    # Create query engine
    query_engine = kg_index.as_query_engine(
        include_text=True,
        retriever_mode="hybrid",
        similarity_top_k=3,
    )
    
    # Query examples
    queries = [
        "What entities are related to Python?",
        "What are the relationships between algorithms and data structures?",
        "Extract key concepts and their relationships from the documents."
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Router_Query_Engine(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    RouterQueryEngine routing queries to appropriate sub-engines.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("ROUTER QUERY ENGINE EXAMPLE")
    print("="*80)
    
    try:
        from llama_index.core.selectors import LLMSingleSelector
    except ImportError:
        print("RouterQueryEngine components not available in this version")
        return
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    embed_model = OpenAIEmbedding()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load documents
    all_documents = SimpleDirectoryReader(documents_dir).load_data()
    
    # Separate documents by topic
    python_docs = [doc for doc in all_documents if "python" in doc.metadata.get("file_path", "").lower()]
    ds_docs = [doc for doc in all_documents if "data_structures" in doc.metadata.get("file_path", "").lower()]
    algo_docs = [doc for doc in all_documents if "algorithms" in doc.metadata.get("file_path", "").lower()]
    
    # Create separate indices
    print("\nCreating separate indices for routing...")
    python_index = VectorStoreIndex.from_documents(python_docs) if python_docs else None
    ds_index = VectorStoreIndex.from_documents(ds_docs) if ds_docs else None
    algo_index = VectorStoreIndex.from_documents(algo_docs) if algo_docs else None
    
    # Create query engines with descriptions
    query_engines = {}
    
    if python_index:
        query_engines["python"] = python_index.as_query_engine()
    
    if ds_index:
        query_engines["data_structures"] = ds_index.as_query_engine()
    
    if algo_index:
        query_engines["algorithms"] = algo_index.as_query_engine()
    
    # Create router query engine
    router_query_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=[
            {
                "query_engine": engine,
                "description": f"Useful for questions about {topic}"
            }
            for topic, engine in query_engines.items()
        ],
        selector=LLMSingleSelector.from_defaults(llm=llm),
        verbose=True
    )
    
    # Query examples
    queries = [
        "What is Python?",
        "Explain hash tables.",
        "How does quicksort work?",
        "What are the differences between lists and arrays?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = router_query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


class Custom_Retriever(BaseRetriever):
    """
    Custom retriever class inheriting from BaseRetriever.
    Implements a hybrid retrieval strategy combining vector search with keyword matching.
    """
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_threshold: float = 0.5,
    ):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._keyword_threshold = keyword_threshold
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid vector and keyword search.
        
        Args:
            query_bundle: Query bundle containing the query string
            
        Returns:
            List of nodes with scores
        """
        # Get vector retrieval results
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        
        # Simple keyword matching boost
        query_lower = query_bundle.query_str.lower()
        query_words = set(query_lower.split())
        
        # Boost scores for nodes containing query keywords
        enhanced_nodes = []
        for node in vector_nodes:
            node_text_lower = node.node.get_content(metadata_mode=MetadataMode.NONE).lower()
            node_words = set(node_text_lower.split())
            
            # Calculate keyword overlap
            keyword_overlap = len(query_words.intersection(node_words)) / max(len(query_words), 1)
            
            # Enhance score if keyword overlap is significant
            if keyword_overlap >= self._keyword_threshold:
                enhanced_score = node.score * (1.0 + keyword_overlap * 0.3)
                enhanced_nodes.append(
                    NodeWithScore(node=node.node, score=enhanced_score)
                )
            else:
                enhanced_nodes.append(node)
        
        # Sort by enhanced score
        enhanced_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return enhanced_nodes


def Custom_Retriever_Engine(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Implement a custom retriever class inheriting from BaseRetriever.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("CUSTOM RETRIEVER ENGINE EXAMPLE")
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
    
    # Create base vector retriever
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )
    
    # Create custom retriever
    custom_retriever = Custom_Retriever(
        vector_retriever=vector_retriever,
        keyword_threshold=0.3,
    )
    
    # Create query engine with custom retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=custom_retriever,
        llm=llm,
    )
    
    # Query examples
    queries = [
        "What is Python programming?",
        "Explain data structures and algorithms.",
        "How do sorting algorithms work?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")
        print("-" * 80)


def Response_Evaluator(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Evaluate response relevance and faithfulness using LLM.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("RESPONSE EVALUATOR EXAMPLE")
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
    
    # Create evaluators
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    
    # Test queries
    test_cases = [
        {
            "query": "What is Python?",
            "expected_answer": "Python is a programming language"
        },
        {
            "query": "Explain data structures.",
            "expected_answer": "Data structures organize data in memory"
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        print(f"\nQuery: {query}")
        
        # Get response
        response = query_engine.query(query)
        print(f"Response: {response}")
        
        # Evaluate faithfulness
        faithfulness_result = faithfulness_evaluator.evaluate_response(
            query=query,
            response=response,
            contexts=[node.get_content() for node in response.source_nodes]
        )
        print(f"Faithfulness Score: {faithfulness_result.score}")
        print(f"Faithfulness Feedback: {faithfulness_result.feedback}")
        
        # Evaluate relevancy
        relevancy_result = relevancy_evaluator.evaluate_response(
            query=query,
            response=response,
        )
        print(f"Relevancy Score: {relevancy_result.score}")
        print(f"Relevancy Feedback: {relevancy_result.feedback}")
        
        print("-" * 80)


def Advanced_Query_Config(documents_dir: str, api_key: Optional[str] = None) -> None:
    """
    Configure similarity_top_k, response_mode, and streaming.
    
    Args:
        documents_dir: Directory containing documents to index
        api_key: OpenAI API key (optional)
    """
    print("\n" + "="*80)
    print("ADVANCED QUERY CONFIG EXAMPLE")
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
    
    # Configuration 1: High similarity_top_k with compact response
    print("\n--- Configuration 1: High retrieval, compact response ---")
    query_engine_1 = index.as_query_engine(
        similarity_top_k=10,
        response_mode=ResponseMode.COMPACT,
    )
    
    response_1 = query_engine_1.query("What is Python?")
    print(f"Query: What is Python?")
    print(f"Response: {response_1}")
    print(f"Retrieved nodes: {len(response_1.source_nodes)}")
    
    # Configuration 2: Low similarity_top_k with tree summarize
    print("\n--- Configuration 2: Low retrieval, tree summarize ---")
    query_engine_2 = index.as_query_engine(
        similarity_top_k=2,
        response_mode=ResponseMode.TREE_SUMMARIZE,
    )
    
    response_2 = query_engine_2.query("Explain programming concepts.")
    print(f"Query: Explain programming concepts.")
    print(f"Response: {response_2}")
    print(f"Retrieved nodes: {len(response_2.source_nodes)}")
    
    # Configuration 3: Streaming response
    print("\n--- Configuration 3: Streaming response ---")
    query_engine_3 = index.as_query_engine(
        similarity_top_k=5,
        response_mode=ResponseMode.COMPACT,
        streaming=True,
    )
    
    query_str = "What are the main data structures?"
    print(f"Query: {query_str}")
    print("Streaming response:")
    
    streaming_response = query_engine_3.query(query_str)
    for token in streaming_response.response_gen:
        print(token, end="", flush=True)
    print("\n")
    
    # Configuration 4: Custom retriever with specific top_k
    print("\n--- Configuration 4: Custom retriever configuration ---")
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=7,
    )
    
    query_engine_4 = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_mode=ResponseMode.REFINE,
    )
    
    response_4 = query_engine_4.query("Compare different algorithms.")
    print(f"Query: Compare different algorithms.")
    print(f"Response: {response_4}")
    print(f"Retrieved nodes: {len(response_4.source_nodes)}")
    
    print("-" * 80)


def main():
    """
    Main function to run all query engine examples.
    """
    print("LlamaIndex Query Engine Examples")
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
        Vector_Store_Query_Engine(documents_dir, api_key)
        Summary_Index_Engine(documents_dir, api_key)
        Knowledge_Graph_Engine(documents_dir, api_key)
        Router_Query_Engine(documents_dir, api_key)
        Custom_Retriever_Engine(documents_dir, api_key)
        Response_Evaluator(documents_dir, api_key)
        Advanced_Query_Config(documents_dir, api_key)
        
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
