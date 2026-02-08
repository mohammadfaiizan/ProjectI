"""
Customer Support Agent Implementation

An intelligent customer support agent that handles queries, looks up orders,
processes returns, answers FAQs, and escalates when needed using OpenAI
function calling.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any
from openai import OpenAI


class Order_Status(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Ticket_Status(Enum):
    """Ticket status enumeration."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Customer_Intent(Enum):
    """Customer intent categories."""
    ORDER_INQUIRY = "order_inquiry"
    RETURN_REQUEST = "return_request"
    FAQ = "faq"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"
    OTHER = "other"


@dataclass
class Order:
    """Represents a customer order."""
    order_id: str
    customer_email: str
    customer_phone: str
    items: List[Dict[str, Any]]
    total_amount: float
    status: Order_Status
    created_date: datetime
    shipping_address: Dict[str, str]
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[datetime] = None


@dataclass
class Ticket:
    """Represents a support ticket."""
    ticket_id: str
    customer_email: str
    subject: str
    description: str
    status: Ticket_Status
    created_date: datetime
    assigned_agent: Optional[str] = None


class Knowledge_Base:
    """In-memory FAQ knowledge base."""

    def __init__(self):
        """Initialize knowledge base with sample FAQs."""
        self.faqs = [
            {
                "id": "1",
                "question": "How do I track my order?",
                "answer": "You can track your order using the tracking number sent to your email, or by visiting our website and entering your order ID.",
                "category": "shipping",
                "tags": ["tracking", "order", "shipping"],
            },
            {
                "id": "2",
                "question": "What is your return policy?",
                "answer": "We offer a 30-day return policy. Items must be unused and in original packaging. Please contact us to initiate a return.",
                "category": "returns",
                "tags": ["return", "refund", "policy"],
            },
            {
                "id": "3",
                "question": "How long does shipping take?",
                "answer": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) and overnight shipping are also available.",
                "category": "shipping",
                "tags": ["shipping", "delivery", "time"],
            },
            {
                "id": "4",
                "question": "Can I cancel my order?",
                "answer": "Orders can be cancelled within 24 hours of placement if they haven't shipped yet. Contact us with your order ID to cancel.",
                "category": "orders",
                "tags": ["cancel", "order", "modification"],
            },
            {
                "id": "5",
                "question": "Do you offer international shipping?",
                "answer": "Yes, we ship internationally. Shipping costs and delivery times vary by country. Check our shipping page for details.",
                "category": "shipping",
                "tags": ["international", "shipping", "global"],
            },
        ]

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search FAQs for relevant answers."""
        query_lower = query.lower()
        results = []

        for faq in self.faqs:
            score = 0
            question_lower = faq["question"].lower()
            answer_lower = faq["answer"].lower()

            # Simple keyword matching
            for word in query_lower.split():
                if word in question_lower:
                    score += 2
                if word in answer_lower:
                    score += 1
                if word in faq.get("tags", []):
                    score += 1

            if score > 0:
                results.append((score, faq))

        results.sort(key=lambda x: x[0], reverse=True)
        return [faq for _, faq in results[:limit]]

    def get_answer(self, query: str) -> Optional[str]:
        """Get best matching answer for query."""
        results = self.search(query, limit=1)
        if results:
            return results[0]["answer"]
        return None


class Order_System:
    """Mock order management system."""

    def __init__(self):
        """Initialize order system with sample orders."""
        self.orders: Dict[str, Order] = {}
        self._initialize_sample_orders()

    def _initialize_sample_orders(self):
        """Initialize with sample orders."""
        now = datetime.now()

        order1 = Order(
            order_id="ORD-001",
            customer_email="customer1@example.com",
            customer_phone="555-0101",
            items=[
                {"name": "Widget A", "quantity": 2, "price": 29.99},
                {"name": "Widget B", "quantity": 1, "price": 49.99},
            ],
            total_amount=109.97,
            status=Order_Status.SHIPPED,
            created_date=now - timedelta(days=5),
            shipping_address={
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
            },
            tracking_number="TRACK-12345",
            estimated_delivery=now + timedelta(days=2),
        )

        order2 = Order(
            order_id="ORD-002",
            customer_email="customer2@example.com",
            customer_phone="555-0102",
            items=[{"name": "Widget C", "quantity": 3, "price": 19.99}],
            total_amount=59.97,
            status=Order_Status.DELIVERED,
            created_date=now - timedelta(days=10),
            shipping_address={
                "street": "456 Oak Ave",
                "city": "Los Angeles",
                "state": "CA",
                "zip": "90001",
            },
            tracking_number="TRACK-67890",
            estimated_delivery=now - timedelta(days=3),
        )

        order3 = Order(
            order_id="ORD-003",
            customer_email="customer1@example.com",
            customer_phone="555-0101",
            items=[{"name": "Widget D", "quantity": 1, "price": 79.99}],
            total_amount=79.99,
            status=Order_Status.PROCESSING,
            created_date=now - timedelta(hours=2),
            shipping_address={
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
            },
        )

        self.orders[order1.order_id] = order1
        self.orders[order2.order_id] = order2
        self.orders[order3.order_id] = order3

    def lookup_order(self, order_id: str) -> Optional[Order]:
        """Look up order by ID."""
        return self.orders.get(order_id)

    def lookup_orders_by_email(self, email: str) -> List[Order]:
        """Look up all orders for a customer email."""
        return [order for order in self.orders.values() if order.customer_email == email]

    def lookup_orders_by_phone(self, phone: str) -> List[Order]:
        """Look up all orders for a customer phone."""
        return [order for order in self.orders.values() if order.customer_phone == phone]

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        order = self.lookup_order(order_id)
        if order:
            return order.status.value
        return None

    def process_return(self, order_id: str, reason: str) -> Dict[str, Any]:
        """Process a return request."""
        order = self.lookup_order(order_id)
        if not order:
            return {
                "success": False,
                "message": f"Order {order_id} not found",
            }

        if order.status == Order_Status.CANCELLED:
            return {
                "success": False,
                "message": "Order is already cancelled",
            }

        return {
            "success": True,
            "message": f"Return request processed for order {order_id}",
            "return_id": f"RET-{order_id}",
            "refund_amount": order.total_amount,
            "estimated_refund_days": 5,
        }


class Ticket_System:
    """Support ticket management system."""

    def __init__(self):
        """Initialize ticket system."""
        self.tickets: Dict[str, Ticket] = {}
        self.next_ticket_id = 1

    def create_ticket(
        self, customer_email: str, subject: str, description: str
    ) -> Ticket:
        """Create a new support ticket."""
        ticket_id = f"TICKET-{self.next_ticket_id:04d}"
        self.next_ticket_id += 1

        ticket = Ticket(
            ticket_id=ticket_id,
            customer_email=customer_email,
            subject=subject,
            description=description,
            status=Ticket_Status.OPEN,
            created_date=datetime.now(),
        )

        self.tickets[ticket_id] = ticket
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get ticket by ID."""
        return self.tickets.get(ticket_id)

    def update_ticket_status(
        self, ticket_id: str, status: Ticket_Status
    ) -> bool:
        """Update ticket status."""
        ticket = self.get_ticket(ticket_id)
        if ticket:
            ticket.status = status
            return True
        return False


class Intent_Classifier:
    """Classifies customer intent from queries."""

    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        """Initialize intent classifier."""
        self.client = client
        self.model = model

    def classify(self, query: str) -> Customer_Intent:
        """Classify customer intent."""
        prompt = f"""Classify the following customer support query into one of these categories:
- order_inquiry: Questions about order status, tracking, delivery
- return_request: Requests to return or exchange products
- faq: General questions that might be answered from FAQ
- complaint: Issues or concerns
- escalation: Requests to speak with a human agent
- other: Anything else

Query: "{query}"

Respond with only the category name."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            category = response.choices[0].message.content.strip().lower()
            for intent in Customer_Intent:
                if intent.value == category:
                    return intent
        except Exception:
            pass

        return Customer_Intent.OTHER


class Customer_Support_Agent:
    """Main customer support agent with function calling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
    ):
        """Initialize customer support agent."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.knowledge_base = Knowledge_Base()
        self.order_system = Order_System()
        self.ticket_system = Ticket_System()
        self.intent_classifier = Intent_Classifier(self.client, model)
        self.conversation_history: List[Dict[str, str]] = []

    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": "Check the status of an order by order ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The order ID to check",
                            },
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_orders_by_email",
                    "description": "Look up all orders for a customer by email address",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Customer email address",
                            },
                        },
                        "required": ["email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_return",
                    "description": "Process a return request for an order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The order ID to return",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for return",
                            },
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_faq",
                    "description": "Search the knowledge base for answers to frequently asked questions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or query to search for",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create a support ticket for issues that need tracking or human attention",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_email": {
                                "type": "string",
                                "description": "Customer email address",
                            },
                            "subject": {
                                "type": "string",
                                "description": "Brief subject of the issue",
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the issue",
                            },
                        },
                        "required": ["customer_email", "subject", "description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "escalate_to_human",
                    "description": "Escalate the conversation to a human agent when the issue is too complex or requires human judgment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for escalation",
                            },
                        },
                        "required": ["reason"],
                    },
                },
            },
        ]

    def _execute_tool(
        self, function_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool function."""
        if function_name == "check_order_status":
            order_id = arguments.get("order_id")
            order = self.order_system.lookup_order(order_id)
            if order:
                return {
                    "order_id": order.order_id,
                    "status": order.status.value,
                    "items": order.items,
                    "total_amount": order.total_amount,
                    "tracking_number": order.tracking_number,
                    "estimated_delivery": (
                        order.estimated_delivery.isoformat()
                        if order.estimated_delivery
                        else None
                    ),
                }
            return {"error": f"Order {order_id} not found"}

        elif function_name == "lookup_orders_by_email":
            email = arguments.get("email")
            orders = self.order_system.lookup_orders_by_email(email)
            return {
                "orders": [
                    {
                        "order_id": o.order_id,
                        "status": o.status.value,
                        "total_amount": o.total_amount,
                        "created_date": o.created_date.isoformat(),
                    }
                    for o in orders
                ]
            }

        elif function_name == "process_return":
            order_id = arguments.get("order_id")
            reason = arguments.get("reason")
            return self.order_system.process_return(order_id, reason)

        elif function_name == "search_faq":
            query = arguments.get("query")
            answer = self.knowledge_base.get_answer(query)
            if answer:
                return {"answer": answer}
            return {"answer": "I couldn't find a specific answer to your question. Let me connect you with a human agent."}

        elif function_name == "create_ticket":
            customer_email = arguments.get("customer_email")
            subject = arguments.get("subject")
            description = arguments.get("description")
            ticket = self.ticket_system.create_ticket(
                customer_email, subject, description
            )
            return {
                "ticket_id": ticket.ticket_id,
                "status": ticket.status.value,
                "message": f"Ticket {ticket.ticket_id} created successfully",
            }

        elif function_name == "escalate_to_human":
            reason = arguments.get("reason")
            return {
                "escalated": True,
                "message": "Your conversation has been escalated to a human agent. They will be with you shortly.",
                "reason": reason,
            }

        return {"error": f"Unknown function: {function_name}"}

    def handle_query(
        self, customer_query: str, customer_email: Optional[str] = None
    ) -> str:
        """Handle a customer query and return response."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful customer support agent. You can help customers with:
- Checking order status and tracking information
- Processing returns and refunds
- Answering frequently asked questions
- Creating support tickets
- Escalating to human agents when needed

Be friendly, professional, and helpful. Use the available tools to look up information and process requests. If you need information you don't have access to, escalate to a human agent.""",
            }
        ]

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": customer_query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._define_tools(),
            tool_choice="auto",
        )

        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                tool_result = self._execute_tool(function_name, arguments)

                messages.append(message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result),
                    }
                )

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._define_tools(),
            )

            assistant_message = final_response.choices[0].message.content
        else:
            assistant_message = message.content

        self.conversation_history.append({"role": "user", "content": customer_query})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        return assistant_message

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


def main():
    """Example usage of the Customer Support Agent."""

    print("Initializing Customer Support Agent...")
    agent = Customer_Support_Agent()

    print("\n" + "=" * 80)
    print("Customer Support Agent - Example Interactions")
    print("=" * 80)

    scenarios = [
        {
            "name": "Order Status Inquiry",
            "query": "Can you check the status of order ORD-001?",
        },
        {
            "name": "FAQ Query",
            "query": "What is your return policy?",
        },
        {
            "name": "Return Request",
            "query": "I want to return order ORD-002 because the item was damaged",
        },
        {
            "name": "Order History",
            "query": "Can you show me all my orders? My email is customer1@example.com",
        },
        {
            "name": "Shipping Question",
            "query": "How long does shipping take?",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"Customer: {scenario['query']}")
        print("\nAgent: ", end="")
        response = agent.handle_query(scenario["query"])
        print(response)

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
