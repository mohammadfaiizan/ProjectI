"""
Main module for RAG Chatbot.
Provides setup and demo functions for running the chatbot.
"""

from Config import LLM_Config, Vector_Store_Config, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from Tools import Document_Processor, Vector_Store_Manager
from Agent import RAG_Agent
from typing import List, Dict, Any, Optional
import os


def Setup_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    collection_name: str = "rag_documents",
    persist_directory: str = "./chroma_db",
    documents_directory: Optional[str] = None,
    web_urls: Optional[List[str]] = None
) -> RAG_Agent:
    """
    Set up the complete RAG system with all components.
    
    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for LLM generation
        collection_name: ChromaDB collection name
        persist_directory: Directory to persist vector store
        documents_directory: Optional directory containing .txt files
        web_urls: Optional list of URLs to load documents from
        
    Returns:
        Configured RAG_Agent instance
    """
    print("Setting up RAG system...")
    
    # Initialize configurations
    print("1. Initializing LLM and embeddings...")
    llm_config = LLM_Config(model_name=model_name, temperature=temperature)
    llm = llm_config.Get_LLM()
    embeddings = llm_config.Get_Embeddings()
    
    print("2. Configuring vector store...")
    vector_config = Vector_Store_Config(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Initialize document processor
    print("3. Initializing document processor...")
    doc_processor = Document_Processor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Load documents
    all_documents = []
    
    if documents_directory and os.path.isdir(documents_directory):
        print(f"4. Loading documents from {documents_directory}...")
        try:
            docs = doc_processor.Load_Text_Documents(documents_directory)
            all_documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not load documents from directory: {e}")
    
    if web_urls:
        print(f"5. Loading documents from {len(web_urls)} URL(s)...")
        for url in web_urls:
            try:
                docs = doc_processor.Load_Web_Page(url)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load URL {url}: {e}")
    
    if not all_documents:
        print("Warning: No documents loaded. Creating empty vector store.")
        # Create empty store - user can add documents later
        vector_manager = Vector_Store_Manager(
            collection_name=vector_config.Get_Collection_Name(),
            persist_directory=vector_config.Get_Persist_Directory(),
            embeddings=embeddings
        )
        # Create a dummy document to initialize the store
        from langchain_core.documents import Document
        dummy_doc = Document(page_content="Initial document", metadata={"source": "dummy"})
        vector_manager.Create_Store([dummy_doc])
    else:
        # Split documents
        print("6. Splitting documents into chunks...")
        chunks = doc_processor.Split_Documents(all_documents)
        
        # Create vector store
        print("7. Creating vector store...")
        vector_manager = Vector_Store_Manager(
            collection_name=vector_config.Get_Collection_Name(),
            persist_directory=vector_config.Get_Persist_Directory(),
            embeddings=embeddings
        )
        vector_manager.Create_Store(chunks)
    
    # Get retriever
    print("8. Creating retriever...")
    retriever = vector_manager.Get_Retriever(search_kwargs={"k": TOP_K})
    
    # Create agent
    print("9. Building RAG agent...")
    agent = RAG_Agent(llm=llm, retriever=retriever)
    agent.Build_Graph()
    
    print("Setup complete!")
    return agent


def Run_Demo(agent: Optional[RAG_Agent] = None) -> None:
    """
    Run interactive demo loop.
    
    Args:
        agent: Optional pre-configured agent. If None, calls Setup_System()
    """
    if agent is None:
        print("No agent provided. Setting up system...")
        agent = Setup_System()
    
    print("\n" + "="*60)
    print("RAG Chatbot Demo")
    print("="*60)
    print("Type your questions (or 'quit'/'exit' to end)")
    print("-"*60 + "\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            result = agent.Query(question)
            
            print(f"\nAssistant: {result['answer']}")
            
            if result.get("sources"):
                print(f"\nSources: {', '.join(set(result['sources']))}")
            
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def Run_Batch_Queries(
    queries: List[str],
    agent: Optional[RAG_Agent] = None
) -> List[Dict[str, Any]]:
    """
    Run a batch of queries and return results.
    
    Args:
        queries: List of questions to ask
        agent: Optional pre-configured agent. If None, calls Setup_System()
        
    Returns:
        List of result dictionaries
    """
    if agent is None:
        print("No agent provided. Setting up system...")
        agent = Setup_System()
    
    results = []
    
    print("\n" + "="*60)
    print(f"Running {len(queries)} batch queries")
    print("="*60 + "\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}/{len(queries)}: {query}")
        try:
            result = agent.Query(query)
            results.append({
                "query": query,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "context": result.get("context", [])
            })
            print(f"Answer: {result['answer'][:200]}...")
            if result.get("sources"):
                print(f"Sources: {', '.join(set(result['sources']))}")
            print("-"*60 + "\n")
        except Exception as e:
            print(f"Error processing query: {e}\n")
            results.append({
                "query": query,
                "answer": f"Error: {e}",
                "sources": [],
                "context": []
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Initializing RAG Chatbot...")
    
    # Setup system (you can customize parameters here)
    agent = Setup_System(
        model_name="gpt-4o-mini",
        temperature=0.0
    )
    
    # Run interactive demo
    Run_Demo(agent)
