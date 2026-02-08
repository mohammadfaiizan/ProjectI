"""
Agent module for Customer Support Agent.

This module contains the LangGraph-based customer support agent with stateful
conversation management, intent classification, and routing to appropriate handlers.
"""

from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from Config import LLM_Config, Support_Config
from Tools import (
    Check_Order_Status,
    Search_FAQ,
    Process_Return,
    Create_Support_Ticket,
    Escalate_To_Human,
    Knowledge_Base
)


# ============================================================================
# State Definition
# ============================================================================

class Support_State(TypedDict):
    """
    State schema for the customer support agent.
    
    Tracks conversation history, customer information, intent classification,
    order details, FAQ results, ticket creation, escalation status, and responses.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    intent: Optional[str]
    order_info: Optional[Dict[str, Any]]
    faq_results: Optional[List[Dict[str, Any]]]
    ticket_created: Optional[Dict[str, Any]]
    escalated: bool
    response: Optional[str]
    conversation_history: List[str]
    turn_count: int


# ============================================================================
# Customer Support Graph
# ============================================================================

class Customer_Support_Graph:
    """
    LangGraph-based customer support agent.
    
    Implements a stateful workflow with intent classification and routing
    to specialized handlers for different customer intents.
    """
    
    def __init__(
        self,
        llm_config: LLM_Config,
        support_config: Support_Config,
        knowledge_base: Optional[Knowledge_Base] = None
    ):
        """
        Initialize the customer support graph.
        
        Args:
            llm_config: LLM configuration instance
            support_config: Support system configuration
            knowledge_base: Optional knowledge base for FAQ search
        """
        self.llm = llm_config.Get_LLM()
        self.support_config = support_config
        self.knowledge_base = knowledge_base
        self.graph = None
        self.app = None
        self._Build_Graph()
    
    def _Build_Graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(Support_State)
        
        # Add nodes
        workflow.add_node("classify_intent", self._Classify_Intent)
        workflow.add_node("handle_order", self._Handle_Order)
        workflow.add_node("handle_return", self._Handle_Return)
        workflow.add_node("handle_faq", self._Handle_FAQ)
        workflow.add_node("handle_complaint", self._Handle_Complaint)
        workflow.add_node("generate_response", self._Generate_Response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional routing from classify_intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._Route_Intent,
            {
                "order_status": "handle_order",
                "return": "handle_return",
                "faq": "handle_faq",
                "complaint": "handle_complaint",
                "other": "generate_response"
            }
        )
        
        # All handlers route to generate_response
        workflow.add_edge("handle_order", "generate_response")
        workflow.add_edge("handle_return", "generate_response")
        workflow.add_edge("handle_faq", "generate_response")
        workflow.add_edge("handle_complaint", "generate_response")
        
        # Generate response routes to END
        workflow.add_edge("generate_response", END)
        
        # Compile the graph
        self.graph = workflow
        self.app = workflow.compile()
    
    def _Classify_Intent(self, state: Support_State) -> Support_State:
        """
        Classify customer intent from their message.
        
        Uses LLM to classify intent into: order_status, return, faq, complaint, other
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classification system for customer support.
Classify the customer's message into one of these categories:
- order_status: Customer wants to check order status or tracking
- return: Customer wants to return or refund an item
- faq: Customer has a general question that might be in FAQ
- complaint: Customer has a complaint or issue requiring escalation
- other: Anything else that doesn't fit the above categories

Respond with ONLY the category name, nothing else."""),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"message": last_message})
        intent = response.content.strip().lower()
        
        # Normalize intent
        if "order" in intent or "status" in intent or "track" in intent:
            intent = "order_status"
        elif "return" in intent or "refund" in intent:
            intent = "return"
        elif "complaint" in intent or "problem" in intent or "issue" in intent:
            intent = "complaint"
        elif "faq" in intent or "question" in intent or "how" in intent or "what" in intent:
            intent = "faq"
        else:
            intent = "other"
        
        return {
            **state,
            "intent": intent,
            "turn_count": state.get("turn_count", 0) + 1
        }
    
    def _Route_Intent(self, state: Support_State) -> str:
        """
        Route to appropriate handler based on classified intent.
        
        Returns the name of the next node to execute.
        """
        intent = state.get("intent", "other")
        return intent
    
    def _Handle_Order(self, state: Support_State) -> Support_State:
        """
        Handle order status inquiries.
        
        Extracts order ID from message and looks up order status.
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Try to extract order ID from message
        order_id = None
        for word in last_message.split():
            if word.upper().startswith("ORD-"):
                order_id = word.upper()
                break
        
        order_info = None
        if order_id:
            order_info = Check_Order_Status.invoke({"order_id": order_id})
        else:
            # Ask for order ID
            order_info = {
                "found": False,
                "error": "Please provide your order ID (e.g., ORD-001) to check status."
            }
        
        return {
            **state,
            "order_info": order_info
        }
    
    def _Handle_Return(self, state: Support_State) -> Support_State:
        """
        Handle return requests.
        
        Extracts order ID and reason, then processes the return.
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Try to extract order ID
        order_id = None
        for word in last_message.split():
            if word.upper().startswith("ORD-"):
                order_id = word.upper()
                break
        
        # Extract reason (everything after "return" or "refund")
        reason = "Customer requested return"
        if "return" in last_message.lower():
            parts = last_message.lower().split("return", 1)
            if len(parts) > 1:
                reason = parts[1].strip()
        elif "refund" in last_message.lower():
            parts = last_message.lower().split("refund", 1)
            if len(parts) > 1:
                reason = parts[1].strip()
        
        return_result = None
        if order_id:
            return_result = Process_Return.invoke({
                "order_id": order_id,
                "reason": reason
            })
        else:
            return_result = {
                "success": False,
                "error": "Please provide your order ID (e.g., ORD-001) to process the return."
            }
        
        return {
            **state,
            "order_info": return_result
        }
    
    def _Handle_FAQ(self, state: Support_State) -> Support_State:
        """
        Handle FAQ questions using knowledge base or keyword search.
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        faq_results = []
        
        if self.knowledge_base:
            # Use semantic search
            faq_results = self.knowledge_base.Search(last_message, k=3)
        else:
            # Fallback to keyword search
            faq_results = Search_FAQ.invoke({"query": last_message})
        
        return {
            **state,
            "faq_results": faq_results
        }
    
    def _Handle_Complaint(self, state: Support_State) -> Support_State:
        """
        Handle complaints by creating a ticket and optionally escalating.
        """
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        customer_id = state.get("customer_id", "UNKNOWN")
        
        # Create support ticket
        ticket = Create_Support_Ticket.invoke({
            "customer_id": customer_id,
            "issue": last_message,
            "priority": "high"
        })
        
        # Auto-escalate if configured
        escalated = False
        if self.support_config.Should_Auto_Escalate_Complaints():
            escalation = Escalate_To_Human.invoke({
                "reason": "Complaint requires human attention"
            })
            escalated = escalation.get("escalated", False)
        
        return {
            **state,
            "ticket_created": ticket,
            "escalated": escalated
        }
    
    def _Generate_Response(self, state: Support_State) -> Support_State:
        """
        Generate final customer-facing response based on handler results.
        """
        intent = state.get("intent", "other")
        order_info = state.get("order_info")
        faq_results = state.get("faq_results")
        ticket_created = state.get("ticket_created")
        escalated = state.get("escalated", False)
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        company_name = self.support_config.Get_Company_Name()
        
        # Build context for response generation
        context_parts = []
        
        if intent == "order_status":
            if order_info and order_info.get("found"):
                order = order_info
                context_parts.append(f"Order Status: {order.get('status', 'unknown').upper()}")
                context_parts.append(f"Order Total: ${order.get('total', 0):.2f}")
                if order.get("tracking_number"):
                    context_parts.append(f"Tracking: {order['tracking_number']}")
                if order.get("estimated_delivery"):
                    context_parts.append(f"Estimated Delivery: {order['estimated_delivery']}")
            else:
                context_parts.append("Order not found or order ID missing.")
        
        elif intent == "return":
            if order_info and order_info.get("success"):
                context_parts.append(f"Return authorized: {order_info.get('return_authorization')}")
                context_parts.append(f"Refund amount: ${order_info.get('refund_amount', 0):.2f}")
                context_parts.append(f"Instructions: {order_info.get('instructions', '')}")
            else:
                context_parts.append(f"Return processing failed: {order_info.get('error', 'Unknown error')}")
        
        elif intent == "faq":
            if faq_results:
                context_parts.append("Found relevant FAQ entries:")
                for idx, faq in enumerate(faq_results[:2], 1):
                    if isinstance(faq, dict):
                        q = faq.get("question", "")
                        a = faq.get("answer", "")
                        context_parts.append(f"{idx}. Q: {q}\n   A: {a}")
            else:
                context_parts.append("No matching FAQ entries found.")
        
        elif intent == "complaint":
            if ticket_created:
                context_parts.append(f"Ticket created: {ticket_created.get('ticket_id')}")
                context_parts.append(f"Priority: {ticket_created.get('priority', 'medium')}")
            if escalated:
                context_parts.append("Conversation escalated to human agent.")
        
        context = "\n".join(context_parts)
        
        # Generate response using LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful customer support agent for {company_name}.
Generate a friendly, professional, and concise response to the customer's query.
Use the provided context to answer accurately. Be empathetic and solution-oriented.
Keep responses clear and actionable."""),
            ("human", """Customer Message: {message}

Context Information:
{context}

Generate a helpful response:""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "message": last_message,
            "context": context
        })
        
        response_text = response.content
        
        # Add to conversation history
        conversation_history = state.get("conversation_history", [])
        conversation_history.append(f"Customer: {last_message}")
        conversation_history.append(f"Agent: {response_text}")
        
        return {
            **state,
            "response": response_text,
            "conversation_history": conversation_history
        }
    
    def Chat(self, message: str, customer_id: str = "CUST-001") -> Dict[str, Any]:
        """
        Process a customer message through the support graph.
        
        Args:
            message: Customer message
            customer_id: Customer identifier
        
        Returns:
            Dictionary with response and state information
        """
        if not self.app:
            raise RuntimeError("Graph not compiled. Call _Build_Graph() first.")
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "intent": None,
            "order_info": None,
            "faq_results": None,
            "ticket_created": None,
            "escalated": False,
            "response": None,
            "conversation_history": [],
            "turn_count": 0
        }
        
        result = self.app.invoke(initial_state)
        
        return {
            "response": result.get("response", "I apologize, but I couldn't process your request."),
            "intent": result.get("intent"),
            "order_info": result.get("order_info"),
            "faq_results": result.get("faq_results"),
            "ticket_created": result.get("ticket_created"),
            "escalated": result.get("escalated", False),
            "conversation_history": result.get("conversation_history", [])
        }
