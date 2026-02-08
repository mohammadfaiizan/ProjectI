"""
Sample Input module for Customer Support Agent.

This module contains predefined customer scenarios for testing and demonstration
of the customer support system capabilities.
"""

from typing import List, Dict, Any
from Config import LLM_Config, Support_Config
from Tools import Knowledge_Base
from Agent import Customer_Support_Graph


# ============================================================================
# Customer Scenarios
# ============================================================================

CUSTOMER_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": 1,
        "name": "Order Status Inquiry",
        "description": "Customer asks about the status of an existing order",
        "customer_id": "CUST-101",
        "messages": [
            "Hi, I'd like to check the status of my order ORD-001"
        ],
        "expected_intent": "order_status"
    },
    {
        "scenario_id": 2,
        "name": "Return Request",
        "description": "Customer wants to return an item with a valid order",
        "customer_id": "CUST-102",
        "messages": [
            "I need to return my order ORD-003. The item doesn't fit my desk."
        ],
        "expected_intent": "return"
    },
    {
        "scenario_id": 3,
        "name": "FAQ Question - Shipping",
        "description": "Customer asks a question about shipping that should match FAQ",
        "customer_id": "CUST-103",
        "messages": [
            "What are your shipping options and how long does it take?"
        ],
        "expected_intent": "faq"
    },
    {
        "scenario_id": 4,
        "name": "Product Complaint",
        "description": "Customer has a complaint that requires escalation and ticket creation",
        "customer_id": "CUST-104",
        "messages": [
            "I'm very upset! The product I received is completely broken and unusable. This is unacceptable!"
        ],
        "expected_intent": "complaint"
    },
    {
        "scenario_id": 5,
        "name": "General Question - Store Hours",
        "description": "Customer asks about store hours, should match FAQ",
        "customer_id": "CUST-105",
        "messages": [
            "What are your store hours?"
        ],
        "expected_intent": "faq"
    },
    {
        "scenario_id": 6,
        "name": "Multi-Turn Conversation",
        "description": "Customer first checks order status, then requests a return",
        "customer_id": "CUST-101",
        "messages": [
            "Can you tell me the status of order ORD-001?",
            "Actually, I want to return that order. The mouse doesn't work properly."
        ],
        "expected_intent": ["order_status", "return"]
    }
]


# ============================================================================
# Sample Execution
# ============================================================================

def Run_Samples(
    use_vector_store: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Execute all customer scenarios and return results.
    
    Args:
        use_vector_store: Whether to use vector store for FAQ search
        verbose: Whether to print detailed output
    
    Returns:
        List of results for each scenario
    """
    print("=" * 80)
    print("Customer Support Agent - Sample Scenarios")
    print("=" * 80)
    print("\nInitializing support system...\n")
    
    # Initialize support system
    try:
        llm_config = LLM_Config(model_name="gpt-4o-mini", temperature=0.0)
        support_config = Support_Config(company_name="TechStore")
        
        knowledge_base = None
        if use_vector_store:
            try:
                knowledge_base = Knowledge_Base()
                if verbose:
                    print("Vector store knowledge base initialized.")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not initialize vector store: {e}")
                    print("Falling back to keyword-based FAQ search.")
                knowledge_base = None
        
        support_system = Customer_Support_Graph(
            llm_config=llm_config,
            support_config=support_config,
            knowledge_base=knowledge_base
        )
        
        if verbose:
            print("Support system initialized successfully.\n")
            print("-" * 80)
    
    except Exception as e:
        print(f"Error initializing support system: {e}")
        return []
    
    results = []
    
    # Process each scenario
    for scenario in CUSTOMER_SCENARIOS:
        scenario_id = scenario["scenario_id"]
        name = scenario["name"]
        customer_id = scenario["customer_id"]
        messages = scenario["messages"]
        expected_intent = scenario["expected_intent"]
        
        if verbose:
            print(f"\nScenario {scenario_id}: {name}")
            print(f"Description: {scenario['description']}")
            print(f"Customer ID: {customer_id}")
            print("-" * 80)
        
        scenario_results = []
        
        # Process each message in the scenario
        for idx, message in enumerate(messages, 1):
            if verbose:
                print(f"\n[Turn {idx}]")
                print(f"Customer: {message}")
            
            try:
                result = support_system.Chat(message, customer_id)
                response = result.get("response", "No response")
                detected_intent = result.get("intent")
                
                if verbose:
                    print(f"Agent: {response}")
                    print(f"Detected Intent: {detected_intent}")
                    
                    if result.get("order_info"):
                        order_info = result["order_info"]
                        if order_info.get("found"):
                            print(f"Order Status: {order_info.get('status', 'unknown')}")
                        elif order_info.get("success"):
                            print(f"Return Auth: {order_info.get('return_authorization')}")
                    
                    if result.get("faq_results"):
                        print(f"FAQ Results: {len(result['faq_results'])} entries found")
                    
                    if result.get("ticket_created"):
                        ticket = result["ticket_created"]
                        print(f"Ticket Created: {ticket.get('ticket_id')}")
                        print(f"Priority: {ticket.get('priority')}")
                    
                    if result.get("escalated"):
                        print("Status: ESCALATED TO HUMAN AGENT")
                
                # Check if intent matches expectation
                intent_match = False
                if isinstance(expected_intent, list):
                    intent_match = detected_intent in expected_intent
                else:
                    intent_match = detected_intent == expected_intent
                
                scenario_results.append({
                    "message": message,
                    "response": response,
                    "detected_intent": detected_intent,
                    "expected_intent": expected_intent if not isinstance(expected_intent, list) else expected_intent[idx - 1],
                    "intent_match": intent_match,
                    "result": result
                })
            
            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                if verbose:
                    print(f"ERROR: {error_msg}")
                
                scenario_results.append({
                    "message": message,
                    "response": None,
                    "error": error_msg,
                    "detected_intent": None,
                    "expected_intent": expected_intent if not isinstance(expected_intent, list) else expected_intent[idx - 1],
                    "intent_match": False
                })
        
        results.append({
            "scenario_id": scenario_id,
            "name": name,
            "results": scenario_results
        })
        
        if verbose:
            print("\n" + "=" * 80)
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_messages = sum(len(s["results"]) for s in results)
        successful_intents = sum(
            sum(1 for r in s["results"] if r.get("intent_match", False))
            for s in results
        )
        
        print(f"Total Scenarios: {len(results)}")
        print(f"Total Messages: {total_messages}")
        print(f"Intent Matches: {successful_intents}/{total_messages}")
        print(f"Success Rate: {successful_intents/total_messages*100:.1f}%")
        print("=" * 80)
    
    return results


def Run_Single_Scenario(
    scenario_id: int,
    use_vector_store: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single scenario by ID.
    
    Args:
        scenario_id: ID of the scenario to run (1-6)
        use_vector_store: Whether to use vector store for FAQ search
        verbose: Whether to print detailed output
    
    Returns:
        Result dictionary for the scenario
    """
    scenario = next(
        (s for s in CUSTOMER_SCENARIOS if s["scenario_id"] == scenario_id),
        None
    )
    
    if not scenario:
        print(f"Scenario {scenario_id} not found.")
        return {}
    
    # Temporarily replace CUSTOMER_SCENARIOS with single scenario
    original_scenarios = CUSTOMER_SCENARIOS.copy()
    CUSTOMER_SCENARIOS.clear()
    CUSTOMER_SCENARIOS.append(scenario)
    
    try:
        results = Run_Samples(use_vector_store=use_vector_store, verbose=verbose)
        return results[0] if results else {}
    finally:
        # Restore original scenarios
        CUSTOMER_SCENARIOS.clear()
        CUSTOMER_SCENARIOS.extend(original_scenarios)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    Run_Samples(use_vector_store=True, verbose=True)
