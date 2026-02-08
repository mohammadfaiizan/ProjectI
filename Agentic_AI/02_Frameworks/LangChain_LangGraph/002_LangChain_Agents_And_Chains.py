"""
Comprehensive LangChain Agents and Chains Implementation

This module demonstrates various LangChain patterns including:
- Basic chains with LCEL (LangChain Expression Language)
- Conversational chains with memory
- Tool-using agents
- Structured output chains
- RAG (Retrieval Augmented Generation) chains
- Multi-chain routing

Requirements:
    - langchain
    - langchain-openai
    - langchain-community
    - chromadb
    - pydantic
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from operator import add

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain


# ============================================================================
# Configuration and Setup
# ============================================================================

def Setup_Environment():
    """
    Setup environment variables and configuration.
    Ensure OPENAI_API_KEY is set in your environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running this script."
        )
    print("Environment setup complete.")


# ============================================================================
# 1. Basic Chain with LCEL
# ============================================================================

def Create_Simple_Chain():
    """
    Creates a simple chain using LCEL: prompt | model | output_parser.
    
    Returns:
        A runnable chain that processes text input.
    """
    print("\n" + "="*60)
    print("1. Basic Chain with LCEL - Simple Chain")
    print("="*60)
    
    # Create the chain components
    prompt = ChatPromptTemplate.from_template(
        "Translate the following text to French: {text}"
    )
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    output_parser = StrOutputParser()
    
    # Chain them together using LCEL
    chain = prompt | model | output_parser
    
    # Run the chain
    result = chain.invoke({"text": "Hello, how are you?"})
    print(f"Input: Hello, how are you?")
    print(f"Output: {result}")
    
    return chain


def Create_Multi_Step_Chain():
    """
    Creates a chain with multiple processing steps.
    
    Returns:
        A runnable chain with multiple transformations.
    """
    print("\n" + "="*60)
    print("1. Basic Chain with LCEL - Multi-Step Chain")
    print("="*60)
    
    # Step 1: Extract key information
    extract_prompt = ChatPromptTemplate.from_template(
        "Extract the main topic from: {input}"
    )
    
    # Step 2: Generate explanation
    explain_prompt = ChatPromptTemplate.from_template(
        "Explain the following topic in simple terms: {topic}"
    )
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    output_parser = StrOutputParser()
    
    # Create multi-step chain
    chain = (
        {"topic": extract_prompt | model | output_parser}
        | RunnablePassthrough()
        | explain_prompt
        | model
        | output_parser
    )
    
    result = chain.invoke({"input": "Quantum computing uses quantum mechanics"})
    print(f"Input: Quantum computing uses quantum mechanics")
    print(f"Output: {result}")
    
    return chain


def Create_Chain_With_Lambda():
    """
    Demonstrates RunnablePassthrough and RunnableLambda for custom processing.
    
    Returns:
        A chain with custom lambda functions.
    """
    print("\n" + "="*60)
    print("1. Basic Chain with LCEL - Lambda Functions")
    print("="*60)
    
    prompt = ChatPromptTemplate.from_template(
        "Count the words in: {text}"
    )
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    output_parser = StrOutputParser()
    
    # Custom lambda to add metadata
    def Add_Metadata(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        text = input_dict.get("text", "")
        return {
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        }
    
    # Chain with lambda
    chain = (
        RunnableLambda(Add_Metadata)
        | prompt
        | model
        | output_parser
    )
    
    result = chain.invoke({"text": "Hello world from LangChain"})
    print(f"Input: Hello world from LangChain")
    print(f"Output: {result}")
    
    return chain


# ============================================================================
# 2. Conversational Chain
# ============================================================================

def Create_Conversational_Chain_With_Buffer_Memory():
    """
    Creates a conversational chain with ConversationBufferMemory.
    Maintains full conversation history.
    
    Returns:
        A conversation chain with buffer memory.
    """
    print("\n" + "="*60)
    print("2. Conversational Chain - Buffer Memory")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Simulate conversation
    conversation_history = []
    
    def Run_Conversation(user_input: str) -> str:
        nonlocal conversation_history
        # Format messages for the chain
        messages = [
            ("system", "You are a helpful assistant."),
        ] + conversation_history + [("human", user_input)]
        
        result = chain.invoke({"input": user_input, "chat_history": conversation_history})
        conversation_history.append(("human", user_input))
        conversation_history.append(("ai", result))
        return result
    
    # Test conversation
    response1 = Run_Conversation("My name is Alice")
    print(f"User: My name is Alice")
    print(f"Assistant: {response1}")
    
    response2 = Run_Conversation("What's my name?")
    print(f"\nUser: What's my name?")
    print(f"Assistant: {response2}")
    
    return chain


def Create_Conversational_Chain_With_Summary_Memory():
    """
    Creates a conversational chain with ConversationSummaryMemory.
    Summarizes conversation history to save tokens.
    
    Returns:
        A conversation chain with summary memory.
    """
    print("\n" + "="*60)
    print("2. Conversational Chain - Summary Memory")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create summary memory
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Simulate conversation with summary
    conversation_summary = ""
    conversation_history = []
    
    def Run_Conversation_With_Summary(user_input: str) -> str:
        nonlocal conversation_summary, conversation_history
        
        # Build messages with summary
        messages = [("system", "You are a helpful assistant.")]
        if conversation_summary:
            messages.append(("system", f"Previous conversation summary: {conversation_summary}"))
        messages.extend(conversation_history)
        messages.append(("human", user_input))
        
        result = chain.invoke({
            "input": user_input,
            "chat_history": conversation_history
        })
        
        conversation_history.append(("human", user_input))
        conversation_history.append(("ai", result))
        
        # Update summary periodically (simplified)
        if len(conversation_history) >= 4:
            summary_prompt = ChatPromptTemplate.from_template(
                "Summarize this conversation: {history}"
            )
            summary_chain = summary_prompt | llm | StrOutputParser()
            history_text = "\n".join([f"{role}: {msg}" for role, msg in conversation_history])
            conversation_summary = summary_chain.invoke({"history": history_text})
            conversation_history = []
        
        return result
    
    response1 = Run_Conversation_With_Summary("I love programming in Python")
    print(f"User: I love programming in Python")
    print(f"Assistant: {response1}")
    
    response2 = Run_Conversation_With_Summary("What did I say I love?")
    print(f"\nUser: What did I say I love?")
    print(f"Assistant: {response2}")
    
    return chain


# ============================================================================
# 3. Tool-Using Agent
# ============================================================================

@tool
def Calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2")
    
    Returns:
        The result of the calculation as a string.
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def Get_Current_Time() -> str:
    """
    Returns the current date and time.
    
    Returns:
        Current datetime as a string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def Create_Tool_Using_Agent():
    """
    Creates a ReAct agent that can use tools.
    Includes custom tools: calculator and current time.
    Also uses web search tool.
    
    Returns:
        An AgentExecutor with tools.
    """
    print("\n" + "="*60)
    print("3. Tool-Using Agent")
    print("="*60)
    
    # Initialize tools
    tools = [
        Calculate,
        Get_Current_Time,
        DuckDuckGoSearchRun()
    ]
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create prompt for ReAct agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can use tools.
        Use the available tools to answer questions accurately.
        When using tools, provide clear reasoning about why you're using them."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    
    # Create agent executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    # Test the agent
    test_queries = [
        "What is 15 multiplied by 23?",
        "What time is it now?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = agent_executor.invoke({"input": query})
            print(f"Answer: {result['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return agent_executor


def Create_Agent_With_Error_Handling():
    """
    Creates an agent executor with comprehensive error handling.
    
    Returns:
        An AgentExecutor with enhanced error handling.
    """
    print("\n" + "="*60)
    print("3. Tool-Using Agent - Error Handling")
    print("="*60)
    
    tools = [Calculate, Get_Current_Time]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    
    def Handle_Error(error: Exception) -> str:
        """Custom error handler."""
        error_msg = str(error)
        if "parsing" in error_msg.lower():
            return "I apologize, but I had trouble understanding that request. Could you rephrase it?"
        return f"I encountered an error: {error_msg}"
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=Handle_Error,
        max_iterations=3,
        return_intermediate_steps=True
    )
    
    # Test with potential error case
    try:
        result = agent_executor.invoke({
            "input": "Calculate 10 divided by 0"
        })
        print(f"Query: Calculate 10 divided by 0")
        print(f"Answer: {result['output']}")
    except Exception as e:
        print(f"Error handled: {str(e)}")
    
    return agent_executor


# ============================================================================
# 4. Structured Output Chain
# ============================================================================

class Person_Info(BaseModel):
    """Schema for person information."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Job title or occupation")
    location: str = Field(description="City and country where they live")


class Product_Review(BaseModel):
    """Schema for product review."""
    product_name: str = Field(description="Name of the product")
    rating: int = Field(description="Rating from 1 to 5", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")


def Create_Structured_Output_Chain():
    """
    Creates a chain that outputs structured data using Pydantic models.
    
    Returns:
        A chain that returns structured output.
    """
    print("\n" + "="*60)
    print("4. Structured Output Chain")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Method 1: Using with_structured_output
    structured_llm = llm.with_structured_output(Person_Info)
    
    prompt = ChatPromptTemplate.from_template(
        "Extract information about the person from: {text}"
    )
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "text": "John Smith is a 35-year-old software engineer living in San Francisco, USA."
    })
    
    print("Input: John Smith is a 35-year-old software engineer living in San Francisco, USA.")
    print(f"Structured Output:")
    print(f"  Name: {result.name}")
    print(f"  Age: {result.age}")
    print(f"  Occupation: {result.occupation}")
    print(f"  Location: {result.location}")
    
    return chain


def Create_Structured_Output_With_Parser():
    """
    Creates a structured output chain using PydanticOutputParser.
    Alternative method for structured outputs.
    
    Returns:
        A chain with Pydantic output parser.
    """
    print("\n" + "="*60)
    print("4. Structured Output Chain - With Parser")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=Product_Review)
    
    # Create prompt with format instructions
    prompt = ChatPromptTemplate.from_template(
        """Review the following product description and create a structured review.
        
        {format_instructions}
        
        Product description: {description}
        """
    )
    
    # Add format instructions to prompt
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "description": "A wireless mouse with excellent battery life, comfortable design, but sometimes has connectivity issues."
    })
    
    print("Input: Product description about wireless mouse")
    print(f"Structured Output:")
    print(f"  Product Name: {result.product_name}")
    print(f"  Rating: {result.rating}/5")
    print(f"  Pros: {', '.join(result.pros)}")
    print(f"  Cons: {', '.join(result.cons)}")
    print(f"  Summary: {result.summary}")
    
    return chain


# ============================================================================
# 5. RAG Chain
# ============================================================================

def Create_Documents_For_RAG() -> List[str]:
    """
    Creates sample documents for RAG demonstration.
    
    Returns:
        List of document strings.
    """
    documents = [
        "LangChain is a framework for developing applications powered by language models. "
        "It enables applications to connect a language model to other sources of data and "
        "allow a language model to interact with its environment.",
        
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "It extends LangChain with graph-based workflows and state management.",
        
        "Vector stores are databases optimized for storing and querying embeddings. "
        "They enable semantic search by finding similar vectors in high-dimensional space.",
        
        "Retrieval Augmented Generation (RAG) combines retrieval of relevant documents "
        "with generation of responses. This allows LLMs to access up-to-date information "
        "beyond their training data.",
        
        "Embeddings are vector representations of text that capture semantic meaning. "
        "Similar texts have similar embeddings, enabling semantic search capabilities."
    ]
    return documents


def Create_RAG_Chain():
    """
    Creates a RAG (Retrieval Augmented Generation) chain.
    Includes document loading, text splitting, embedding, vector store, and retrieval.
    
    Returns:
        A RAG chain for question answering.
    """
    print("\n" + "="*60)
    print("5. RAG Chain")
    print("="*60)
    
    # Create sample documents
    documents_text = Create_Documents_For_RAG()
    
    # Create document objects
    from langchain_core.documents import Document
    documents = [Document(page_content=text) for text in documents_text]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Created {len(splits)} document chunks")
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    
    # Use ChromaDB for vector storage
    # Note: In production, you'd persist this to disk
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="langchain_docs"
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Create RAG chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context:
        
        {context}
        
        Question: {question}
        
        Provide a detailed answer based on the context provided."""
    )
    
    def Format_Docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | Format_Docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Test the RAG chain
    questions = [
        "What is LangChain?",
        "What is RAG?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag_chain.invoke(question)
        print(f"Answer: {result}")
    
    return rag_chain, vectorstore


def Create_RAG_Chain_With_Source_Citations():
    """
    Creates a RAG chain that includes source citations.
    
    Returns:
        A RAG chain that returns answers with source references.
    """
    print("\n" + "="*60)
    print("5. RAG Chain - With Source Citations")
    print("="*60)
    
    documents_text = Create_Documents_For_RAG()
    from langchain_core.documents import Document
    documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) 
                 for i, text in enumerate(documents_text)]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="langchain_docs_cited"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context.
        Cite the source document numbers in your answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )
    
    def Format_Docs_With_Sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | Format_Docs_With_Sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    question = "What is LangGraph?"
    print(f"Question: {question}")
    result = rag_chain.invoke(question)
    print(f"Answer: {result}")
    
    return rag_chain


# ============================================================================
# 6. Multi-Chain Router
# ============================================================================

def Create_Multi_Chain_Router():
    """
    Creates a router that directs inputs to different chains based on conditions.
    Uses RunnableBranch for conditional logic.
    
    Returns:
        A router chain that routes to different specialized chains.
    """
    print("\n" + "="*60)
    print("6. Multi-Chain Router")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Chain 1: Math chain
    math_prompt = ChatPromptTemplate.from_template(
        "You are a math tutor. Solve this problem step by step: {input}"
    )
    math_chain = math_prompt | llm | StrOutputParser()
    
    # Chain 2: General Q&A chain
    qa_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer this question: {input}"
    )
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    # Chain 3: Code explanation chain
    code_prompt = ChatPromptTemplate.from_template(
        "You are a programming expert. Explain this code or concept: {input}"
    )
    code_chain = code_prompt | llm | StrOutputParser()
    
    # Routing function
    def Route_Input(input_dict: Dict[str, Any]) -> str:
        """Routes input to appropriate chain based on keywords."""
        text = input_dict.get("input", "").lower()
        
        # Check for math keywords
        math_keywords = ["calculate", "solve", "math", "equation", "formula", "+", "-", "*", "/"]
        if any(keyword in text for keyword in math_keywords):
            return "math"
        
        # Check for code keywords
        code_keywords = ["code", "function", "programming", "python", "syntax", "algorithm"]
        if any(keyword in text for keyword in code_keywords):
            return "code"
        
        # Default to Q&A
        return "qa"
    
    # Create router using RunnableBranch
    router = RunnableBranch(
        (lambda x: Route_Input(x) == "math", math_chain),
        (lambda x: Route_Input(x) == "code", code_chain),
        qa_chain  # Default chain
    )
    
    # Test the router
    test_inputs = [
        "What is 25 multiplied by 4?",
        "Explain how Python lists work",
        "What is the capital of France?",
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        route = Route_Input({"input": test_input})
        print(f"Route: {route}")
        result = router.invoke({"input": test_input})
        print(f"Output: {result[:100]}...")  # Truncate for display
    
    return router


def Create_Advanced_Router_With_LLM():
    """
    Creates an advanced router that uses an LLM to determine routing.
    More sophisticated than keyword-based routing.
    
    Returns:
        An LLM-based router chain.
    """
    print("\n" + "="*60)
    print("6. Multi-Chain Router - LLM-Based")
    print("="*60)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Routing chain
    routing_prompt = ChatPromptTemplate.from_template(
        """Determine which category this input belongs to.
        Categories: math, code, general
        
        Input: {input}
        
        Respond with only one word: math, code, or general"""
    )
    
    routing_chain = routing_prompt | llm | StrOutputParser()
    
    # Specialized chains
    math_prompt = ChatPromptTemplate.from_template(
        "Solve this math problem: {input}"
    )
    math_chain = math_prompt | llm | StrOutputParser()
    
    code_prompt = ChatPromptTemplate.from_template(
        "Explain this programming concept: {input}"
    )
    code_chain = code_prompt | llm | StrOutputParser()
    
    general_prompt = ChatPromptTemplate.from_template(
        "Answer this question: {input}"
    )
    general_chain = general_prompt | llm | StrOutputParser()
    
    def Route_With_LLM(input_dict: Dict[str, Any]) -> str:
        """Uses LLM to determine routing."""
        route = routing_chain.invoke(input_dict).strip().lower()
        return route
    
    # Create router
    def Route_To_Chain(input_dict: Dict[str, Any]):
        """Routes to appropriate chain based on LLM decision."""
        route = Route_With_LLM(input_dict)
        
        if "math" in route:
            return math_chain.invoke(input_dict)
        elif "code" in route:
            return code_chain.invoke(input_dict)
        else:
            return general_chain.invoke(input_dict)
    
    router = RunnableLambda(Route_To_Chain)
    
    # Test
    test_input = "How do I reverse a list in Python?"
    print(f"Input: {test_input}")
    route = Route_With_LLM({"input": test_input})
    print(f"Detected Route: {route}")
    result = router.invoke({"input": test_input})
    print(f"Output: {result[:150]}...")
    
    return router


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function that demonstrates all LangChain agents and chains features.
    """
    print("="*60)
    print("LangChain Agents and Chains - Comprehensive Demo")
    print("="*60)
    
    try:
        # Setup
        Setup_Environment()
        
        # 1. Basic Chains
        Create_Simple_Chain()
        Create_Multi_Step_Chain()
        Create_Chain_With_Lambda()
        
        # 2. Conversational Chains
        Create_Conversational_Chain_With_Buffer_Memory()
        Create_Conversational_Chain_With_Summary_Memory()
        
        # 3. Tool-Using Agents
        Create_Tool_Using_Agent()
        Create_Agent_With_Error_Handling()
        
        # 4. Structured Output Chains
        Create_Structured_Output_Chain()
        Create_Structured_Output_With_Parser()
        
        # 5. RAG Chains
        Create_RAG_Chain()
        Create_RAG_Chain_With_Source_Citations()
        
        # 6. Multi-Chain Routers
        Create_Multi_Chain_Router()
        Create_Advanced_Router_With_LLM()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
