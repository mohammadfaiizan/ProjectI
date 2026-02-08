"""
Agent module for RAG Chatbot.
Contains LangGraph-based RAG agent and LCEL-based conversational chain.
"""

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class RAG_State(TypedDict):
    """State schema for RAG agent graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    question: str
    answer: str
    sources: List[str]
    chat_history: List[Dict[str, str]]


class RAG_Agent:
    """LangGraph-based RAG agent with state management."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG agent.
        
        Args:
            llm: Language model instance
            retriever: Document retriever instance
            system_prompt: Optional system prompt for the LLM
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt or self._Get_Default_System_Prompt()
        self.graph = None
        self.chat_history: List[Dict[str, str]] = []
    
    def _Get_Default_System_Prompt(self) -> str:
        """Return default system prompt."""
        return (
            "You are a helpful AI assistant that answers questions based on "
            "the provided context documents. Use only the information from the "
            "context to answer questions. If the context doesn't contain enough "
            "information to answer the question, say so. Always cite your sources "
            "when providing information."
        )
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with nodes.
        
        Returns:
            StateGraph instance
        """
        workflow = StateGraph(RAG_State)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._Retrieve_Context)
        workflow.add_node("generate_answer", self._Generate_Answer)
        workflow.add_node("format_response", self._Format_Response)
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        self.graph = workflow
        return workflow
    
    def _Retrieve_Context(self, state: RAG_State) -> Dict[str, Any]:
        """
        Retrieve relevant context documents for the question.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with context
        """
        question = state["question"]
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Extract context and sources
        context = [doc.page_content for doc in docs]
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        return {
            "context": context,
            "sources": sources
        }
    
    def _Generate_Answer(self, state: RAG_State) -> Dict[str, Any]:
        """
        Generate answer using LLM with context and chat history.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with answer
        """
        question = state["question"]
        context = "\n\n".join(state["context"])
        chat_history = state.get("chat_history", [])
        
        # Build conversation history for prompt
        history_messages = []
        for turn in chat_history:
            if "human" in turn:
                history_messages.append(HumanMessage(content=turn["human"]))
            if "ai" in turn:
                history_messages.append(AIMessage(content=turn["ai"]))
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        # Format prompt with values
        formatted_prompt = prompt.format_messages(
            context=context,
            question=question,
            chat_history=history_messages
        )
        
        # Generate response
        response = self.llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        
        return {"answer": answer}
    
    def _Format_Response(self, state: RAG_State) -> Dict[str, Any]:
        """
        Format the final response with source citations.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with formatted response
        """
        answer = state["answer"]
        sources = state.get("sources", [])
        
        # Format answer with sources
        if sources:
            unique_sources = list(set(sources))
            sources_text = "\n\nSources:\n" + "\n".join(
                f"- {source}" for source in unique_sources
            )
            formatted_answer = answer + sources_text
        else:
            formatted_answer = answer
        
        # Update messages
        messages = state.get("messages", [])
        messages.append(AIMessage(content=formatted_answer))
        
        return {
            "answer": formatted_answer,
            "messages": messages
        }
    
    def Compile(self) -> Any:
        """
        Compile the graph for execution.
        
        Returns:
            Compiled graph
        """
        if self.graph is None:
            self.Build_Graph()
        
        return self.graph.compile()
    
    def Query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question.
        
        Args:
            question: User question
            chat_history: Optional conversation history
            
        Returns:
            Dictionary with answer, sources, and context
        """
        if self.graph is None:
            self.Build_Graph()
        
        compiled_graph = self.Compile()
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "context": [],
            "answer": "",
            "sources": [],
            "chat_history": chat_history or []
        }
        
        # Invoke graph
        result = compiled_graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "context": result["context"]
        }
    
    def Chat(self, question: str) -> str:
        """
        Chat with the agent (maintains internal conversation history).
        
        Args:
            question: User question
            
        Returns:
            Agent response
        """
        result = self.Query(question, self.chat_history)
        
        # Update chat history
        self.chat_history.append({"human": question})
        self.chat_history.append({"ai": result["answer"]})
        
        return result["answer"]
    
    def Reset_History(self) -> None:
        """Reset the conversation history."""
        self.chat_history = []


class Conversational_RAG_Chain:
    """LCEL-based conversational RAG chain using LangChain chains."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize conversational RAG chain.
        
        Args:
            llm: Language model instance
            retriever: Document retriever instance
            system_prompt: Optional system prompt for the LLM
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt or self._Get_Default_System_Prompt()
        self.chain = None
        self._Build_Chain()
    
    def _Get_Default_System_Prompt(self) -> str:
        """Return default system prompt."""
        return (
            "You are a helpful AI assistant that answers questions based on "
            "the provided context documents. Use only the information from the "
            "context to answer questions. If the context doesn't contain enough "
            "information to answer the question, say so."
        )
    
    def _Build_Chain(self) -> None:
        """Build the LCEL chain."""
        # Create history-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt
        )
        
        # Create question answering chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )
        
        # Create retrieval chain
        self.chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
    
    def Query(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """
        Query the chain with a question.
        
        Args:
            question: User question
            chat_history: Optional conversation history as messages
            
        Returns:
            Dictionary with answer and context
        """
        if chat_history is None:
            chat_history = []
        
        result = self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return {
            "answer": result["answer"],
            "context": [doc.page_content for doc in result["context"]],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["context"]]
        }
    
    def Get_Chain(self) -> Any:
        """Return the underlying chain."""
        return self.chain
