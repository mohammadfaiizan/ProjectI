"""
Tools module for Customer Support Agent.

This module contains mock databases and tool functions for order management,
FAQ search, return processing, ticket creation, and escalation handling.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ============================================================================
# Mock Databases
# ============================================================================

ORDER_DATABASE: Dict[str, Dict[str, Any]] = {
    "ORD-001": {
        "id": "ORD-001",
        "customer_id": "CUST-101",
        "customer_name": "John Smith",
        "items": [
            {"name": "Wireless Mouse", "quantity": 1, "price": 29.99},
            {"name": "USB-C Cable", "quantity": 2, "price": 12.99}
        ],
        "status": "shipped",
        "date": "2026-01-15",
        "total": 55.97,
        "tracking_number": "TRACK-12345",
        "estimated_delivery": "2026-01-20"
    },
    "ORD-002": {
        "id": "ORD-002",
        "customer_id": "CUST-102",
        "customer_name": "Sarah Johnson",
        "items": [
            {"name": "Mechanical Keyboard", "quantity": 1, "price": 89.99}
        ],
        "status": "processing",
        "date": "2026-02-01",
        "total": 89.99,
        "tracking_number": None,
        "estimated_delivery": "2026-02-08"
    },
    "ORD-003": {
        "id": "ORD-003",
        "customer_id": "CUST-103",
        "customer_name": "Michael Chen",
        "items": [
            {"name": "Monitor Stand", "quantity": 1, "price": 45.00},
            {"name": "Desk Organizer", "quantity": 1, "price": 25.00}
        ],
        "status": "delivered",
        "date": "2025-12-20",
        "total": 70.00,
        "tracking_number": "TRACK-67890",
        "estimated_delivery": "2025-12-28",
        "delivery_date": "2025-12-26"
    },
    "ORD-004": {
        "id": "ORD-004",
        "customer_id": "CUST-104",
        "customer_name": "Emily Davis",
        "items": [
            {"name": "Laptop Stand", "quantity": 1, "price": 65.00},
            {"name": "Webcam", "quantity": 1, "price": 79.99}
        ],
        "status": "pending",
        "date": "2026-02-05",
        "total": 144.99,
        "tracking_number": None,
        "estimated_delivery": "2026-02-12"
    },
    "ORD-005": {
        "id": "ORD-005",
        "customer_id": "CUST-105",
        "customer_name": "David Wilson",
        "items": [
            {"name": "Wireless Headphones", "quantity": 1, "price": 129.99}
        ],
        "status": "shipped",
        "date": "2026-01-25",
        "total": 129.99,
        "tracking_number": "TRACK-11111",
        "estimated_delivery": "2026-02-02"
    },
    "ORD-006": {
        "id": "ORD-006",
        "customer_id": "CUST-101",
        "customer_name": "John Smith",
        "items": [
            {"name": "USB-C Hub", "quantity": 1, "price": 49.99},
            {"name": "HDMI Cable", "quantity": 1, "price": 15.99}
        ],
        "status": "delivered",
        "date": "2025-11-10",
        "total": 65.98,
        "tracking_number": "TRACK-22222",
        "estimated_delivery": "2025-11-17",
        "delivery_date": "2025-11-15"
    },
    "ORD-007": {
        "id": "ORD-007",
        "customer_id": "CUST-106",
        "customer_name": "Lisa Anderson",
        "items": [
            {"name": "Standing Desk Converter", "quantity": 1, "price": 199.99}
        ],
        "status": "processing",
        "date": "2026-02-06",
        "total": 199.99,
        "tracking_number": None,
        "estimated_delivery": "2026-02-15"
    },
    "ORD-008": {
        "id": "ORD-008",
        "customer_id": "CUST-107",
        "customer_name": "Robert Taylor",
        "items": [
            {"name": "Ergonomic Chair", "quantity": 1, "price": 299.99},
            {"name": "Desk Mat", "quantity": 1, "price": 19.99}
        ],
        "status": "shipped",
        "date": "2026-01-30",
        "total": 319.98,
        "tracking_number": "TRACK-33333",
        "estimated_delivery": "2026-02-10"
    },
    "ORD-009": {
        "id": "ORD-009",
        "customer_id": "CUST-108",
        "customer_name": "Jennifer Martinez",
        "items": [
            {"name": "Monitor", "quantity": 1, "price": 249.99}
        ],
        "status": "delivered",
        "date": "2026-01-05",
        "total": 249.99,
        "tracking_number": "TRACK-44444",
        "estimated_delivery": "2026-01-12",
        "delivery_date": "2026-01-10"
    },
    "ORD-010": {
        "id": "ORD-010",
        "customer_id": "CUST-109",
        "customer_name": "William Brown",
        "items": [
            {"name": "Keyboard Wrist Rest", "quantity": 1, "price": 24.99},
            {"name": "Mouse Pad", "quantity": 1, "price": 14.99},
            {"name": "Cable Management", "quantity": 1, "price": 9.99}
        ],
        "status": "pending",
        "date": "2026-02-07",
        "total": 49.97,
        "tracking_number": None,
        "estimated_delivery": "2026-02-14"
    }
}

FAQ_DATABASE: List[Dict[str, str]] = [
    {
        "question": "What are your shipping options?",
        "answer": "We offer standard shipping (5-7 business days), express shipping (2-3 business days), and overnight shipping (next business day). Shipping costs vary based on order size and destination."
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number via email. You can use this tracking number on our website's tracking page or the carrier's website to monitor your shipment."
    },
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy on most items. Items must be unused and in original packaging. To initiate a return, contact customer support with your order number and reason for return."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to most countries worldwide. International shipping times vary by destination, typically 10-21 business days. Additional customs fees may apply."
    },
    {
        "question": "How do I cancel my order?",
        "answer": "Orders can be cancelled within 24 hours of placement if they haven't shipped yet. Contact customer support with your order number to cancel. Once shipped, you'll need to process a return instead."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, Google Pay, and bank transfers for orders over $500."
    },
    {
        "question": "Are there any discounts or promo codes available?",
        "answer": "We regularly offer promotions and discounts. Sign up for our newsletter to receive exclusive offers. You can also check our promotions page for current deals and promo codes."
    },
    {
        "question": "What are your store hours?",
        "answer": "Our customer support is available Monday through Friday, 9 AM to 6 PM EST. We also offer email support 24/7, with responses typically within 24 hours."
    },
    {
        "question": "How do I change my shipping address?",
        "answer": "If your order hasn't shipped, contact customer support immediately with your order number and new address. Once shipped, address changes may not be possible and you may need to contact the carrier directly."
    },
    {
        "question": "What if my order arrives damaged?",
        "answer": "If your order arrives damaged, please contact us within 48 hours with photos of the damage. We'll arrange for a replacement or full refund, including return shipping costs."
    },
    {
        "question": "Do you offer gift wrapping?",
        "answer": "Yes, we offer gift wrapping for an additional $5.99. Select this option during checkout and include a gift message if desired."
    },
    {
        "question": "How do I create an account?",
        "answer": "Click the 'Sign Up' button in the top right corner of our website. You'll need to provide your email address, create a password, and verify your email. Account creation is free and takes less than a minute."
    },
    {
        "question": "Can I modify my order after placing it?",
        "answer": "Orders can be modified within 2 hours of placement if they haven't entered processing. Contact customer support with your order number and requested changes. After 2 hours, you may need to cancel and reorder."
    },
    {
        "question": "What is your warranty policy?",
        "answer": "Most products come with a manufacturer's warranty, typically 1-2 years. Extended warranties are available for purchase. Warranty details are included with each product listing and in your order confirmation."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach us via email at support@techstore.com, phone at 1-800-TECH-HELP, or through our live chat during business hours. We also offer support tickets through your account dashboard."
    }
]

TICKET_DATABASE: List[Dict[str, Any]] = []


# ============================================================================
# Tool Functions
# ============================================================================

@tool
def Check_Order_Status(order_id: str) -> Dict[str, Any]:
    """
    Check the status of an order by order ID.
    
    Args:
        order_id: The order ID to look up (e.g., "ORD-001")
    
    Returns:
        Dictionary containing order information including status, items, total, etc.
        Returns error message if order not found.
    """
    order = ORDER_DATABASE.get(order_id.upper())
    if not order:
        return {
            "found": False,
            "error": f"Order {order_id} not found. Please verify the order ID."
        }
    
    return {
        "found": True,
        "order_id": order["id"],
        "customer_name": order["customer_name"],
        "status": order["status"],
        "items": order["items"],
        "total": order["total"],
        "date": order["date"],
        "tracking_number": order.get("tracking_number"),
        "estimated_delivery": order.get("estimated_delivery"),
        "delivery_date": order.get("delivery_date")
    }


@tool
def Search_FAQ(query: str) -> List[Dict[str, str]]:
    """
    Search FAQ entries by keyword matching.
    
    Args:
        query: Search query string
    
    Returns:
        List of FAQ entries matching the query, sorted by relevance
    """
    query_lower = query.lower()
    results = []
    
    for faq in FAQ_DATABASE:
        question_lower = faq["question"].lower()
        answer_lower = faq["answer"].lower()
        
        # Simple keyword matching
        query_words = set(query_lower.split())
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())
        
        # Calculate relevance score
        question_matches = len(query_words.intersection(question_words))
        answer_matches = len(query_words.intersection(answer_words))
        relevance = question_matches * 2 + answer_matches
        
        if relevance > 0:
            results.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "relevance": relevance
            })
    
    # Sort by relevance and return top 5
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:5]


@tool
def Process_Return(order_id: str, reason: str) -> Dict[str, Any]:
    """
    Process a return request for an order.
    
    Args:
        order_id: The order ID to return
        reason: Reason for the return
    
    Returns:
        Dictionary with return processing status and return authorization number
    """
    order = ORDER_DATABASE.get(order_id.upper())
    if not order:
        return {
            "success": False,
            "error": f"Order {order_id} not found."
        }
    
    # Check if order is eligible for return (delivered or shipped)
    if order["status"] not in ["delivered", "shipped"]:
        return {
            "success": False,
            "error": f"Order {order_id} is in '{order['status']}' status and cannot be returned yet."
        }
    
    # Generate return authorization
    return_auth = f"RET-{order_id}-{datetime.now().strftime('%Y%m%d')}"
    
    return {
        "success": True,
        "order_id": order_id,
        "return_authorization": return_auth,
        "reason": reason,
        "refund_amount": order["total"],
        "instructions": "Please package the items securely and ship to our returns center. Include the return authorization number on the package."
    }


@tool
def Create_Support_Ticket(customer_id: str, issue: str, priority: str = "medium") -> Dict[str, Any]:
    """
    Create a support ticket for a customer issue.
    
    Args:
        customer_id: Customer ID
        issue: Description of the issue
        priority: Priority level (low, medium, high, urgent)
    
    Returns:
        Dictionary with ticket information including ticket ID
    """
    ticket_id = f"TICKET-{len(TICKET_DATABASE) + 1:04d}"
    
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().isoformat(),
        "assigned_to": None
    }
    
    TICKET_DATABASE.append(ticket)
    
    return {
        "success": True,
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "priority": priority,
        "status": "open",
        "message": f"Support ticket {ticket_id} has been created and will be reviewed by our team."
    }


@tool
def Escalate_To_Human(reason: str) -> Dict[str, str]:
    """
    Escalate the conversation to a human agent.
    
    Args:
        reason: Reason for escalation
    
    Returns:
        Dictionary with escalation confirmation
    """
    return {
        "escalated": True,
        "reason": reason,
        "message": "Your conversation has been escalated to a human agent. Please hold while we connect you."
    }


# ============================================================================
# Knowledge Base Class
# ============================================================================

class Knowledge_Base:
    """
    Vector store-based knowledge base for semantic FAQ search.
    
    Uses ChromaDB with OpenAI embeddings for semantic similarity search
    over FAQ entries.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the knowledge base with FAQ entries.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self._Build_Vector_Store()
    
    def _Build_Vector_Store(self):
        """Build the vector store from FAQ database."""
        documents = []
        for idx, faq in enumerate(FAQ_DATABASE):
            content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            doc = Document(
                page_content=content,
                metadata={"question": faq["question"], "index": idx}
            )
            documents.append(doc)
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
    
    def Search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic search on FAQ entries.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of FAQ entries with similarity scores
        """
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            # Extract question and answer from document content
            content = doc.page_content
            parts = content.split("\nAnswer: ")
            question = parts[0].replace("Question: ", "")
            answer = parts[1] if len(parts) > 1 else ""
            
            formatted_results.append({
                "question": question,
                "answer": answer,
                "similarity_score": float(score),
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def Get_All_FAQs(self) -> List[Dict[str, str]]:
        """Return all FAQ entries."""
        return FAQ_DATABASE.copy()
