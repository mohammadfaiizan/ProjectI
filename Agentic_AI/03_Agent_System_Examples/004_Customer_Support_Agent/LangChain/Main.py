"""
Main module for Customer Support Agent.

This module provides the main entry point and interactive demo for the
customer support system using LangChain and LangGraph.
"""

from typing import Optional
from Config import LLM_Config, Support_Config, FAQ_Config
from Tools import Knowledge_Base
from Agent import Customer_Support_Graph


# ============================================================================
# System Setup
# ============================================================================

def Setup_Support_System(
    model_name: str = "gpt-4o-mini",
    company_name: str = "TechStore",
    use_vector_store: bool = True
) -> Customer_Support_Graph:
    """
    Setup and initialize the customer support system.
    
    Args:
        model_name: Name of the LLM model to use
        company_name: Name of the company
        use_vector_store: Whether to use vector store for FAQ search
    
    Returns:
        Initialized Customer_Support_Graph instance
    """
    # Initialize configurations
    llm_config = LLM_Config(model_name=model_name, temperature=0.0)
    support_config = Support_Config(company_name=company_name)
    
    # Initialize knowledge base if vector store is enabled
    knowledge_base = None
    if use_vector_store:
        try:
            knowledge_base = Knowledge_Base()
            print("Vector store knowledge base initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            print("Falling back to keyword-based FAQ search.")
            knowledge_base = None
    
    # Create and return the support graph
    support_graph = Customer_Support_Graph(
        llm_config=llm_config,
        support_config=support_config,
        knowledge_base=knowledge_base
    )
    
    print(f"Customer Support System initialized for {company_name}.")
    return support_graph


# ============================================================================
# Query Handling
# ============================================================================

def Handle_Customer_Query(
    support_system: Customer_Support_Graph,
    message: str,
    customer_id: str = "CUST-001"
) -> dict:
    """
    Handle a customer query through the support system.
    
    Args:
        support_system: Initialized Customer_Support_Graph instance
        message: Customer message
        customer_id: Customer identifier
    
    Returns:
        Dictionary containing response and metadata
    """
    try:
        result = support_system.Chat(message, customer_id)
        return result
    except Exception as e:
        return {
            "response": f"I apologize, but I encountered an error: {str(e)}",
            "error": str(e),
            "intent": None,
            "escalated": False
        }


# ============================================================================
# Interactive Demo
# ============================================================================

def Run_Demo():
    """
    Run an interactive customer support chat demo.
    
    Allows users to interact with the support system in a conversational manner.
    """
    print("=" * 70)
    print("Customer Support Agent - Interactive Demo")
    print("=" * 70)
    print("\nInitializing support system...")
    
    try:
        support_system = Setup_Support_System()
        print("\nSystem ready! You can now chat with the support agent.")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        customer_id = "CUST-001"
        conversation_count = 0
        
        while True:
            # Get user input
            user_input = input("Customer: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nThank you for contacting customer support. Goodbye!")
                break
            
            # Process query
            conversation_count += 1
            print(f"\n[Processing query #{conversation_count}...]")
            
            result = Handle_Customer_Query(support_system, user_input, customer_id)
            
            # Display response
            print(f"\nAgent: {result.get('response', 'No response generated.')}")
            
            # Display metadata if available
            if result.get("intent"):
                print(f"[Intent: {result['intent']}]")
            if result.get("ticket_created"):
                print(f"[Ticket Created: {result['ticket_created'].get('ticket_id')}]")
            if result.get("escalated"):
                print("[Status: Escalated to human agent]")
            
            print()  # Empty line for readability
        
    except KeyboardInterrupt:
        print("\n\nConversation interrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your configuration and try again.")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    Run_Demo()
