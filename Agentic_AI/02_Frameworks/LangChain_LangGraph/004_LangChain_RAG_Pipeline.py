"""
Comprehensive RAG (Retrieval-Augmented Generation) Pipeline Implementation using LangChain.

This module provides multiple RAG implementations:
1. Basic RAG Pipeline
2. Conversational RAG with chat history
3. Advanced Retrieval (multi-query, ensemble, contextual compression, self-query)
4. Custom RAG Chain with LCEL
5. Agentic RAG
6. RAG Evaluation
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage


def basic_rag_pipeline(
    documents_path: Optional[str] = None,
    web_urls: Optional[List[str]] = None,
    query: str = "What is the main topic?",
    persist_directory: str = "./chroma_db"
) -> Dict[str, Any]:
    """
    Basic RAG Pipeline implementation.
    
    Loads documents, splits them, creates embeddings, stores in vector database,
    and retrieves relevant context for answering questions.
    
    Args:
        documents_path: Path to text file(s) to load. If None, uses web URLs.
        web_urls: List of URLs to load documents from. Used if documents_path is None.
        query: The question to answer.
        persist_directory: Directory to persist the Chroma vector store.
        
    Returns:
        Dictionary containing the answer and source documents.
    """
    try:
        # Initialize LLM and embeddings
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        embeddings = OpenAIEmbeddings()
        
        # Load documents
        documents = []
        if documents_path:
            loader = TextLoader(documents_path)
            documents.extend(loader.load())
        elif web_urls:
            for url in web_urls:
                loader = WebBaseLoader(url)
                documents.extend(loader.load())
        else:
            raise ValueError("Either documents_path or web_urls must be provided")
        
        if not documents:
            raise ValueError("No documents were loaded")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the question. "
                      "If you don't know the answer, say you don't know.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        # Create chain
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Get answer
        answer = chain.invoke(query)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def conversational_rag(
    vectorstore: Chroma,
    chat_history: List[Tuple[str, str]],
    query: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Conversational RAG with chat history support.
    
    Contextualizes questions based on conversation history and retrieves
    relevant documents for answering.
    
    Args:
        vectorstore: Chroma vector store instance.
        chat_history: List of (question, answer) tuples representing conversation history.
        query: Current question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer, source documents, and contextualized question.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Contextualize question based on history
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                      "which might reference context in the chat history, "
                      "formulate a standalone question which can be understood "
                      "without the chat history. Do NOT answer the question, "
                      "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create question answering chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context "
                      "to answer the question. If you don't know the answer, "
                      "say you don't know.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        
        # Convert chat history to LangChain message format
        chat_history_messages = []
        for human_msg, ai_msg in chat_history:
            chat_history_messages.append(HumanMessage(content=human_msg))
            chat_history_messages.append(AIMessage(content=ai_msg))
        
        # Invoke chain
        response = rag_chain.invoke({
            "chat_history": chat_history_messages,
            "question": query
        })
        
        return {
            "answer": response["answer"],
            "source_documents": response.get("context", []),
            "question": response.get("question", query),
            "num_sources": len(response.get("context", []))
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def multi_query_retriever_rag(
    vectorstore: Chroma,
    query: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Multi-query retriever that generates multiple search queries.
    
    Generates multiple queries from different perspectives and retrieves
    documents for each, combining results.
    
    Args:
        vectorstore: Chroma vector store instance.
        query: The question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer and retrieved documents.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create multi-query retriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Create prompt and chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context "
                      "to answer the question. If you don't know the answer, "
                      "say you don't know.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(retrieved_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def ensemble_retriever_rag(
    documents: List[Document],
    query: str,
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None
) -> Dict[str, Any]:
    """
    Ensemble retriever combining BM25 (keyword) and vector search.
    
    Combines BM25 keyword-based retrieval with vector similarity search
    for better retrieval performance.
    
    Args:
        documents: List of Document objects to search.
        query: The question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        embeddings: Optional OpenAIEmbeddings instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer and retrieved documents.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        
        # Create vector store and retriever
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # Retrieve documents
        retrieved_docs = ensemble_retriever.get_relevant_documents(query)
        
        # Create prompt and chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context "
                      "to answer the question. If you don't know the answer, "
                      "say you don't know.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(retrieved_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def contextual_compression_rag(
    vectorstore: Chroma,
    query: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    RAG with contextual compression to reduce noise in retrieved documents.
    
    Uses compression to filter and re-rank retrieved documents based on
    relevance to the query.
    
    Args:
        vectorstore: Chroma vector store instance.
        query: The question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer and compressed documents.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        from langchain.retrievers.document_compressors import LLMChainExtractor
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # Create compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Retrieve compressed documents
        compressed_docs = compression_retriever.get_relevant_documents(query)
        
        # Create prompt and chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context "
                      "to answer the question. If you don't know the answer, "
                      "say you don't know.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(compressed_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        
        return {
            "answer": answer,
            "source_documents": compressed_docs,
            "num_sources": len(compressed_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def self_query_retriever_rag(
    documents: List[Document],
    query: str,
    metadata_field_info: List[Dict[str, Any]],
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None
) -> Dict[str, Any]:
    """
    Self-query retriever with metadata filtering.
    
    Allows natural language queries that include metadata filters,
    automatically extracting and applying filters.
    
    Args:
        documents: List of Document objects with metadata.
        query: Natural language query that may include metadata filters.
        metadata_field_info: List of metadata field descriptions.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        embeddings: Optional OpenAIEmbeddings instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer and filtered documents.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        
        from langchain.chains.query_constructor.base import AttributeInfo
        
        # Create vector store
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
        
        # Convert metadata field info to AttributeInfo objects
        attribute_info = [
            AttributeInfo(**info) for info in metadata_field_info
        ]
        
        # Create self-query retriever
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents="Document content",
            metadata_field_info=attribute_info,
            verbose=True
        )
        
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Create prompt and chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context "
                      "to answer the question. If you don't know the answer, "
                      "say you don't know.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(retrieved_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(query)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def custom_rag_chain_lcel(
    vectorstore: Chroma,
    query: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Custom RAG chain built from scratch using LangChain Expression Language (LCEL).
    
    Demonstrates building a RAG pipeline using LCEL with custom prompt,
    document formatting, and source tracking.
    
    Args:
        vectorstore: Chroma vector store instance.
        query: The question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer, sources, and formatted response.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Custom prompt template
        template = """You are a helpful assistant. Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Format documents with source tracking
        def format_documents(docs: List[Document]) -> str:
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                formatted.append(f"[Source {i} - {source}]\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        # Extract sources
        def extract_sources(docs: List[Document]) -> List[str]:
            return [doc.metadata.get("source", "Unknown") for doc in docs]
        
        # Build RAG chain
        rag_chain = (
            {
                "context": retriever | format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Retrieve documents for source tracking
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Get answer
        answer = rag_chain.invoke(query)
        
        # Extract sources
        sources = extract_sources(retrieved_docs)
        
        return {
            "answer": answer,
            "sources": sources,
            "source_documents": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None, "source_documents": []}


def agentic_rag(
    vectorstore: Chroma,
    query: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Agentic RAG where an agent decides when to retrieve information.
    
    The agent can decide whether to retrieve documents, search, or answer
    directly based on the question.
    
    Args:
        vectorstore: Chroma vector store instance.
        query: The question to answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary containing answer, tool calls, and agent steps.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "document_search",
            "Search through documents to find information. Use this tool when you need to "
            "find specific information from the document collection. Input should be a search query."
        )
        
        # Create agent prompt
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. You have access to a document search tool. "
                      "Use the tool when you need to find information from documents. "
                      "If you don't know the answer after searching, say you don't know."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create agent
        tools = [retriever_tool]
        agent = create_openai_tools_agent(llm, tools, agent_prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Execute agent
        response = agent_executor.invoke({"input": query})
        
        return {
            "answer": response.get("output", ""),
            "intermediate_steps": response.get("intermediate_steps", []),
            "tool_calls": [
                step[0].tool for step in response.get("intermediate_steps", [])
            ]
        }
        
    except Exception as e:
        return {"error": str(e), "answer": None}


def evaluate_retrieval_quality(
    retriever: Any,
    queries: List[str],
    ground_truth_docs: Dict[str, List[str]],
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate retrieval quality using precision and recall.
    
    Args:
        retriever: Retriever instance to evaluate.
        queries: List of test queries.
        ground_truth_docs: Dictionary mapping query to list of relevant document IDs.
        k: Number of documents to retrieve.
        
    Returns:
        Dictionary with precision, recall, and F1 scores.
    """
    try:
        total_precision = 0.0
        total_recall = 0.0
        
        for query in queries:
            retrieved_docs = retriever.get_relevant_documents(query)[:k]
            retrieved_ids = [
                doc.metadata.get("id", str(i)) for i, doc in enumerate(retrieved_docs)
            ]
            
            relevant_ids = set(ground_truth_docs.get(query, []))
            retrieved_set = set(retrieved_ids)
            
            if len(retrieved_set) > 0:
                precision = len(relevant_ids & retrieved_set) / len(retrieved_set)
            else:
                precision = 0.0
            
            if len(relevant_ids) > 0:
                recall = len(relevant_ids & retrieved_set) / len(relevant_ids)
            else:
                recall = 0.0
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(queries) if queries else 0.0
        avg_recall = total_recall / len(queries) if queries else 0.0
        
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": f1,
            "num_queries": len(queries)
        }
        
    except Exception as e:
        return {"error": str(e), "precision": 0.0, "recall": 0.0, "f1_score": 0.0}


def evaluate_answer_quality(
    answer: str,
    reference_answer: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Evaluate answer quality using LLM-based evaluation.
    
    Args:
        answer: Generated answer to evaluate.
        reference_answer: Reference or ground truth answer.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary with evaluation scores and feedback.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator. Rate the quality of the answer on a scale of 1-10. "
                      "Consider accuracy, completeness, and relevance. "
                      "Provide a score and brief feedback."),
            ("human", "Reference Answer: {reference}\n\nGenerated Answer: {answer}\n\n"
                     "Provide a score (1-10) and feedback.")
        ])
        
        chain = evaluation_prompt | llm | StrOutputParser()
        
        evaluation = chain.invoke({
            "reference": reference_answer,
            "answer": answer
        })
        
        return {
            "evaluation": evaluation,
            "answer": answer,
            "reference": reference_answer
        }
        
    except Exception as e:
        return {"error": str(e), "evaluation": None}


def check_answer_faithfulness(
    answer: str,
    source_documents: List[Document],
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Check if the answer is faithful to the source documents.
    
    Args:
        answer: Generated answer to check.
        source_documents: List of source documents used.
        llm: Optional ChatOpenAI instance. Creates new one if not provided.
        
    Returns:
        Dictionary with faithfulness check results.
    """
    try:
        if llm is None:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Combine source documents
        source_text = "\n\n".join([doc.page_content for doc in source_documents])
        
        faithfulness_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checker. Determine if the answer is supported by the source documents. "
                      "Respond with 'FAITHFUL' if the answer is supported, 'UNFAITHFUL' if it contains "
                      "unsupported claims, or 'PARTIAL' if partially supported. Provide reasoning."),
            ("human", "Source Documents:\n{source}\n\nAnswer: {answer}\n\n"
                     "Is the answer faithful to the sources?")
        ])
        
        chain = faithfulness_prompt | llm | StrOutputParser()
        
        result = chain.invoke({
            "source": source_text,
            "answer": answer
        })
        
        is_faithful = "FAITHFUL" in result.upper()
        is_unfaithful = "UNFAITHFUL" in result.upper()
        
        return {
            "faithfulness_check": result,
            "is_faithful": is_faithful,
            "is_unfaithful": is_unfaithful,
            "answer": answer,
            "num_sources": len(source_documents)
        }
        
    except Exception as e:
        return {"error": str(e), "is_faithful": False}


def main():
    """
    Main function demonstrating usage of various RAG pipeline implementations.
    """
    print("=" * 80)
    print("LangChain RAG Pipeline Demonstrations")
    print("=" * 80)
    
    # Set up OpenAI API key (should be set as environment variable)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some functions may fail.")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Example 1: Basic RAG Pipeline
    print("\n1. Basic RAG Pipeline")
    print("-" * 80)
    try:
        # Create sample document
        sample_text = """
        Artificial Intelligence (AI) is transforming the way we work and live.
        Machine learning, a subset of AI, enables computers to learn from data.
        Deep learning uses neural networks with multiple layers to process complex patterns.
        Natural Language Processing (NLP) allows machines to understand human language.
        Computer vision enables machines to interpret and understand visual information.
        """
        
        # Save to temporary file
        temp_file = Path("./temp_sample.txt")
        temp_file.write_text(sample_text)
        
        result = basic_rag_pipeline(
            documents_path=str(temp_file),
            query="What is machine learning?"
        )
        
        print(f"Query: What is machine learning?")
        print(f"Answer: {result.get('answer', 'N/A')}")
        print(f"Number of sources: {result.get('num_sources', 0)}")
        
        # Clean up
        temp_file.unlink()
        
    except Exception as e:
        print(f"Error in basic RAG: {e}")
    
    # Example 2: Custom RAG Chain with LCEL
    print("\n2. Custom RAG Chain with LCEL")
    print("-" * 80)
    try:
        # Create sample documents
        from langchain_core.documents import Document
        
        sample_docs = [
            Document(
                page_content="Python is a high-level programming language.",
                metadata={"source": "doc1.txt", "id": "1"}
            ),
            Document(
                page_content="LangChain is a framework for building LLM applications.",
                metadata={"source": "doc2.txt", "id": "2"}
            ),
            Document(
                page_content="RAG combines retrieval and generation for better answers.",
                metadata={"source": "doc3.txt", "id": "3"}
            )
        ]
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=sample_docs, embedding=embeddings)
        
        result = custom_rag_chain_lcel(
            vectorstore=vectorstore,
            query="What is LangChain?"
        )
        
        print(f"Query: What is LangChain?")
        print(f"Answer: {result.get('answer', 'N/A')}")
        print(f"Sources: {result.get('sources', [])}")
        
    except Exception as e:
        print(f"Error in custom RAG chain: {e}")
    
    # Example 3: Multi-query Retriever
    print("\n3. Multi-query Retriever RAG")
    print("-" * 80)
    try:
        from langchain_core.documents import Document
        
        sample_docs = [
            Document(page_content="The weather today is sunny and warm."),
            Document(page_content="Machine learning models require training data."),
            Document(page_content="Python programming is popular for data science.")
        ]
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=sample_docs, embedding=embeddings)
        
        result = multi_query_retriever_rag(
            vectorstore=vectorstore,
            query="Tell me about data science"
        )
        
        print(f"Query: Tell me about data science")
        print(f"Answer: {result.get('answer', 'N/A')}")
        print(f"Number of sources: {result.get('num_sources', 0)}")
        
    except Exception as e:
        print(f"Error in multi-query retriever: {e}")
    
    # Example 4: RAG Evaluation
    print("\n4. RAG Evaluation")
    print("-" * 80)
    try:
        from langchain_core.documents import Document
        
        sample_docs = [
            Document(
                page_content="Python is a programming language.",
                metadata={"id": "doc1"}
            ),
            Document(
                page_content="Java is another programming language.",
                metadata={"id": "doc2"}
            )
        ]
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=sample_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Simple evaluation
        queries = ["What is Python?"]
        ground_truth = {
            "What is Python?": ["doc1"]
        }
        
        eval_result = evaluate_retrieval_quality(
            retriever=retriever,
            queries=queries,
            ground_truth_docs=ground_truth,
            k=2
        )
        
        print(f"Retrieval Evaluation Results:")
        print(f"  Precision: {eval_result.get('precision', 0):.2f}")
        print(f"  Recall: {eval_result.get('recall', 0):.2f}")
        print(f"  F1 Score: {eval_result.get('f1_score', 0):.2f}")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
