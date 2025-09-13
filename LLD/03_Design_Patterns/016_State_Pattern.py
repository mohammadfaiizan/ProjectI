"""
STATE PATTERN - Behavioral Design Pattern
=========================================

Problem Statement:
Implement the State pattern to allow an object to alter its behavior when its
internal state changes, appearing as if the object changed its class:
- State-dependent behavior encapsulation
- Clean state transitions and management
- Finite state machine implementation
- Context-state collaboration
- State-specific operations and validations

Learning Objectives:
- Understand State vs Strategy pattern differences
- Implement state machines with clean transitions
- Design context-state relationships
- Handle state-specific behavior and validation
- Create complex state-dependent systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable
import time
from datetime import datetime, timedelta
from enum import Enum
import threading
import json


# ============================================================================
# STATE INTERFACE
# ============================================================================

class State(ABC):
    """Abstract state interface."""
    
    @abstractmethod
    def handle_request(self, context: 'Context', request: str, *args, **kwargs) -> Any:
        """Handle request in this state."""
        pass
    
    @abstractmethod
    def get_state_name(self) -> str:
        """Get state name."""
        pass
    
    def enter_state(self, context: 'Context') -> None:
        """Called when entering this state."""
        print(f"Entering state: {self.get_state_name()}")
    
    def exit_state(self, context: 'Context') -> None:
        """Called when exiting this state."""
        print(f"Exiting state: {self.get_state_name()}")
    
    def get_allowed_transitions(self) -> List[str]:
        """Get list of allowed state transitions from this state."""
        return []
    
    def can_transition_to(self, state_name: str) -> bool:
        """Check if transition to given state is allowed."""
        return state_name in self.get_allowed_transitions()


class Context:
    """Context class that maintains state."""
    
    def __init__(self, initial_state: State):
        self._state = initial_state
        self.state_history: List[Dict[str, Any]] = []
        self.transition_count = 0
        
        # Record initial state
        self._record_state_change(None, initial_state.get_state_name())
        initial_state.enter_state(self)
    
    def set_state(self, new_state: State) -> None:
        """Change to new state."""
        old_state_name = self._state.get_state_name()
        new_state_name = new_state.get_state_name()
        
        # Check if transition is allowed
        if not self._state.can_transition_to(new_state_name):
            raise ValueError(f"Invalid transition from {old_state_name} to {new_state_name}")
        
        # Exit current state
        self._state.exit_state(self)
        
        # Change state
        old_state = self._state
        self._state = new_state
        self.transition_count += 1
        
        # Enter new state
        new_state.enter_state(self)
        
        # Record transition
        self._record_state_change(old_state_name, new_state_name)
    
    def get_current_state(self) -> State:
        """Get current state."""
        return self._state
    
    def get_current_state_name(self) -> str:
        """Get current state name."""
        return self._state.get_state_name()
    
    def handle_request(self, request: str, *args, **kwargs) -> Any:
        """Delegate request to current state."""
        return self._state.handle_request(self, request, *args, **kwargs)
    
    def _record_state_change(self, from_state: str, to_state: str) -> None:
        """Record state change in history."""
        record = {
            'from_state': from_state,
            'to_state': to_state,
            'timestamp': datetime.now().isoformat(),
            'transition_number': self.transition_count
        }
        self.state_history.append(record)
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state transition history."""
        return self.state_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context statistics."""
        state_counts = {}
        for record in self.state_history:
            state = record['to_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            'current_state': self.get_current_state_name(),
            'total_transitions': self.transition_count,
            'state_counts': state_counts,
            'most_visited_state': max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else None
        }


# ============================================================================
# VENDING MACHINE STATES
# ============================================================================

class VendingMachine(Context):
    """Vending machine context with state-dependent behavior."""
    
    def __init__(self):
        self.balance = 0.0
        self.inventory = {
            'A1': {'name': 'Coke', 'price': 1.50, 'quantity': 10},
            'A2': {'name': 'Pepsi', 'price': 1.50, 'quantity': 8},
            'B1': {'name': 'Chips', 'price': 2.00, 'quantity': 5},
            'B2': {'name': 'Candy', 'price': 1.25, 'quantity': 12}
        }
        self.selected_item = None
        self.dispensed_items: List[Dict[str, Any]] = []
        
        # Initialize with idle state
        super().__init__(IdleState())
    
    def insert_money(self, amount: float) -> str:
        """Insert money into machine."""
        return self.handle_request('insert_money', amount)
    
    def select_item(self, item_code: str) -> str:
        """Select item to purchase."""
        return self.handle_request('select_item', item_code)
    
    def dispense_item(self) -> str:
        """Dispense selected item."""
        return self.handle_request('dispense_item')
    
    def return_money(self) -> str:
        """Return money to customer."""
        return self.handle_request('return_money')
    
    def cancel_transaction(self) -> str:
        """Cancel current transaction."""
        return self.handle_request('cancel_transaction')
    
    def get_machine_status(self) -> Dict[str, Any]:
        """Get machine status."""
        return {
            'balance': self.balance,
            'selected_item': self.selected_item,
            'inventory': self.inventory,
            'current_state': self.get_current_state_name(),
            'total_sales': len(self.dispensed_items)
        }


class IdleState(State):
    """Idle state - waiting for money or selection."""
    
    def handle_request(self, context: VendingMachine, request: str, *args, **kwargs) -> Any:
        if request == 'insert_money':
            amount = args[0]
            if amount <= 0:
                return "Invalid amount. Please insert valid money."
            
            context.balance += amount
            print(f"Inserted ${amount:.2f}. Balance: ${context.balance:.2f}")
            
            # Transition to has_money state
            context.set_state(HasMoneyState())
            return f"Money inserted. Current balance: ${context.balance:.2f}"
        
        elif request == 'select_item':
            return "Please insert money first."
        
        elif request == 'dispense_item':
            return "No item selected. Please insert money and select an item."
        
        elif request == 'return_money':
            return "No money to return."
        
        elif request == 'cancel_transaction':
            return "No transaction to cancel."
        
        else:
            return f"Invalid request '{request}' in idle state."
    
    def get_state_name(self) -> str:
        return "Idle"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["HasMoney"]


class HasMoneyState(State):
    """Has money state - can select items or return money."""
    
    def handle_request(self, context: VendingMachine, request: str, *args, **kwargs) -> Any:
        if request == 'insert_money':
            amount = args[0]
            if amount <= 0:
                return "Invalid amount."
            
            context.balance += amount
            return f"Added ${amount:.2f}. Total balance: ${context.balance:.2f}"
        
        elif request == 'select_item':
            item_code = args[0]
            
            # Check if item exists
            if item_code not in context.inventory:
                return f"Item {item_code} not found."
            
            item = context.inventory[item_code]
            
            # Check if item is in stock
            if item['quantity'] <= 0:
                return f"{item['name']} is out of stock."
            
            # Check if sufficient balance
            if context.balance < item['price']:
                needed = item['price'] - context.balance
                return f"Insufficient funds. Need ${needed:.2f} more for {item['name']}."
            
            # Select item and transition to item_selected state
            context.selected_item = item_code
            context.set_state(ItemSelectedState())
            return f"Selected {item['name']} (${item['price']:.2f}). Press dispense to complete purchase."
        
        elif request == 'dispense_item':
            return "Please select an item first."
        
        elif request == 'return_money':
            returned_amount = context.balance
            context.balance = 0.0
            
            # Transition back to idle state
            context.set_state(IdleState())
            return f"Returned ${returned_amount:.2f}. Thank you!"
        
        elif request == 'cancel_transaction':
            returned_amount = context.balance
            context.balance = 0.0
            
            # Transition back to idle state
            context.set_state(IdleState())
            return f"Transaction cancelled. Returned ${returned_amount:.2f}."
        
        else:
            return f"Invalid request '{request}' in has_money state."
    
    def get_state_name(self) -> str:
        return "HasMoney"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["ItemSelected", "Idle"]


class ItemSelectedState(State):
    """Item selected state - ready to dispense or can change selection."""
    
    def handle_request(self, context: VendingMachine, request: str, *args, **kwargs) -> Any:
        if request == 'insert_money':
            amount = args[0]
            if amount <= 0:
                return "Invalid amount."
            
            context.balance += amount
            return f"Added ${amount:.2f}. Total balance: ${context.balance:.2f}"
        
        elif request == 'select_item':
            # Allow changing selection
            item_code = args[0]
            
            if item_code not in context.inventory:
                return f"Item {item_code} not found."
            
            item = context.inventory[item_code]
            
            if item['quantity'] <= 0:
                return f"{item['name']} is out of stock."
            
            if context.balance < item['price']:
                needed = item['price'] - context.balance
                return f"Insufficient funds. Need ${needed:.2f} more for {item['name']}."
            
            context.selected_item = item_code
            return f"Changed selection to {item['name']} (${item['price']:.2f})."
        
        elif request == 'dispense_item':
            item_code = context.selected_item
            item = context.inventory[item_code]
            
            # Deduct price from balance
            context.balance -= item['price']
            
            # Reduce inventory
            context.inventory[item_code]['quantity'] -= 1
            
            # Record dispensed item
            dispensed_record = {
                'item_code': item_code,
                'name': item['name'],
                'price': item['price'],
                'timestamp': datetime.now().isoformat()
            }
            context.dispensed_items.append(dispensed_record)
            
            # Clear selection
            context.selected_item = None
            
            # Determine next state based on remaining balance
            if context.balance > 0:
                context.set_state(HasMoneyState())
                return f"Dispensed {item['name']}. Remaining balance: ${context.balance:.2f}"
            else:
                context.set_state(IdleState())
                return f"Dispensed {item['name']}. Thank you for your purchase!"
        
        elif request == 'return_money':
            returned_amount = context.balance
            context.balance = 0.0
            context.selected_item = None
            
            context.set_state(IdleState())
            return f"Transaction cancelled. Returned ${returned_amount:.2f}."
        
        elif request == 'cancel_transaction':
            returned_amount = context.balance
            context.balance = 0.0
            context.selected_item = None
            
            context.set_state(IdleState())
            return f"Transaction cancelled. Returned ${returned_amount:.2f}."
        
        else:
            return f"Invalid request '{request}' in item_selected state."
    
    def get_state_name(self) -> str:
        return "ItemSelected"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["HasMoney", "Idle"]


# ============================================================================
# DOCUMENT WORKFLOW STATES
# ============================================================================

class DocumentStatus(Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Document(Context):
    """Document with workflow states."""
    
    def __init__(self, title: str, author: str):
        self.title = title
        self.author = author
        self.content = ""
        self.reviewers: List[str] = []
        self.approvers: List[str] = []
        self.comments: List[Dict[str, Any]] = []
        self.version = 1
        self.created_at = datetime.now()
        self.published_at = None
        
        # Initialize with draft state
        super().__init__(DraftState())
    
    def edit_content(self, new_content: str, editor: str) -> str:
        """Edit document content."""
        return self.handle_request('edit_content', new_content, editor)
    
    def submit_for_review(self, reviewers: List[str]) -> str:
        """Submit document for review."""
        return self.handle_request('submit_for_review', reviewers)
    
    def add_review_comment(self, reviewer: str, comment: str, rating: int) -> str:
        """Add review comment."""
        return self.handle_request('add_review_comment', reviewer, comment, rating)
    
    def approve_document(self, approver: str) -> str:
        """Approve document."""
        return self.handle_request('approve_document', approver)
    
    def reject_document(self, reviewer: str, reason: str) -> str:
        """Reject document."""
        return self.handle_request('reject_document', reviewer, reason)
    
    def publish_document(self, publisher: str) -> str:
        """Publish document."""
        return self.handle_request('publish_document', publisher)
    
    def archive_document(self, archiver: str) -> str:
        """Archive document."""
        return self.handle_request('archive_document', archiver)
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get document information."""
        return {
            'title': self.title,
            'author': self.author,
            'status': self.get_current_state_name(),
            'version': self.version,
            'content_length': len(self.content),
            'reviewers': self.reviewers,
            'approvers': self.approvers,
            'comments_count': len(self.comments),
            'created_at': self.created_at.isoformat(),
            'published_at': self.published_at.isoformat() if self.published_at else None
        }


class DraftState(State):
    """Draft state - document can be edited."""
    
    def handle_request(self, context: Document, request: str, *args, **kwargs) -> Any:
        if request == 'edit_content':
            new_content, editor = args
            
            # Only author can edit in draft state
            if editor != context.author:
                return f"Only the author ({context.author}) can edit the document in draft state."
            
            context.content = new_content
            return f"Document content updated by {editor}."
        
        elif request == 'submit_for_review':
            reviewers = args[0]
            
            if not context.content.strip():
                return "Cannot submit empty document for review."
            
            if not reviewers:
                return "At least one reviewer must be specified."
            
            context.reviewers = reviewers
            context.set_state(UnderReviewState())
            return f"Document submitted for review to: {', '.join(reviewers)}"
        
        elif request == 'add_review_comment':
            return "Document must be under review to add comments."
        
        elif request == 'approve_document':
            return "Document must be under review to approve."
        
        elif request == 'reject_document':
            return "Document must be under review to reject."
        
        elif request == 'publish_document':
            return "Document must be approved before publishing."
        
        elif request == 'archive_document':
            return "Cannot archive document in draft state."
        
        else:
            return f"Invalid request '{request}' in draft state."
    
    def get_state_name(self) -> str:
        return "Draft"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["UnderReview"]


class UnderReviewState(State):
    """Under review state - reviewers can comment and approve/reject."""
    
    def handle_request(self, context: Document, request: str, *args, **kwargs) -> Any:
        if request == 'edit_content':
            return "Cannot edit document while under review. Document must be rejected first."
        
        elif request == 'submit_for_review':
            return "Document is already under review."
        
        elif request == 'add_review_comment':
            reviewer, comment, rating = args
            
            if reviewer not in context.reviewers:
                return f"{reviewer} is not authorized to review this document."
            
            if not (1 <= rating <= 5):
                return "Rating must be between 1 and 5."
            
            comment_record = {
                'reviewer': reviewer,
                'comment': comment,
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            }
            context.comments.append(comment_record)
            
            return f"Review comment added by {reviewer} (Rating: {rating}/5)."
        
        elif request == 'approve_document':
            approver = args[0]
            
            if approver not in context.reviewers:
                return f"{approver} is not authorized to approve this document."
            
            # Check if all reviewers have commented
            reviewers_who_commented = {comment['reviewer'] for comment in context.comments}
            if not all(reviewer in reviewers_who_commented for reviewer in context.reviewers):
                missing_reviewers = set(context.reviewers) - reviewers_who_commented
                return f"Waiting for reviews from: {', '.join(missing_reviewers)}"
            
            # Check if average rating is acceptable (>= 3)
            if context.comments:
                avg_rating = sum(comment['rating'] for comment in context.comments) / len(context.comments)
                if avg_rating < 3.0:
                    return f"Document cannot be approved. Average rating: {avg_rating:.1f}/5 (minimum: 3.0)"
            
            context.approvers.append(approver)
            context.set_state(ApprovedState())
            return f"Document approved by {approver}. Ready for publishing."
        
        elif request == 'reject_document':
            reviewer, reason = args
            
            if reviewer not in context.reviewers:
                return f"{reviewer} is not authorized to reject this document."
            
            # Add rejection comment
            rejection_comment = {
                'reviewer': reviewer,
                'comment': f"REJECTED: {reason}",
                'rating': 1,
                'timestamp': datetime.now().isoformat()
            }
            context.comments.append(rejection_comment)
            
            # Clear reviewers and transition back to draft
            context.reviewers = []
            context.set_state(DraftState())
            return f"Document rejected by {reviewer}. Returned to draft state."
        
        elif request == 'publish_document':
            return "Document must be approved before publishing."
        
        elif request == 'archive_document':
            return "Cannot archive document while under review."
        
        else:
            return f"Invalid request '{request}' in under_review state."
    
    def get_state_name(self) -> str:
        return "UnderReview"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Approved", "Draft"]


class ApprovedState(State):
    """Approved state - document can be published."""
    
    def handle_request(self, context: Document, request: str, *args, **kwargs) -> Any:
        if request == 'edit_content':
            return "Cannot edit approved document. Create a new version instead."
        
        elif request == 'submit_for_review':
            return "Document is already approved."
        
        elif request == 'add_review_comment':
            return "Cannot add comments to approved document."
        
        elif request == 'approve_document':
            return "Document is already approved."
        
        elif request == 'reject_document':
            return "Cannot reject approved document."
        
        elif request == 'publish_document':
            publisher = args[0]
            
            # Only author or approvers can publish
            if publisher != context.author and publisher not in context.approvers:
                return f"{publisher} is not authorized to publish this document."
            
            context.published_at = datetime.now()
            context.set_state(PublishedState())
            return f"Document published by {publisher} at {context.published_at.isoformat()}."
        
        elif request == 'archive_document':
            archiver = args[0]
            
            if archiver != context.author:
                return f"Only the author ({context.author}) can archive the document."
            
            context.set_state(ArchivedState())
            return f"Document archived by {archiver}."
        
        else:
            return f"Invalid request '{request}' in approved state."
    
    def get_state_name(self) -> str:
        return "Approved"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Published", "Archived"]


class PublishedState(State):
    """Published state - document is live and read-only."""
    
    def handle_request(self, context: Document, request: str, *args, **kwargs) -> Any:
        if request == 'edit_content':
            return "Cannot edit published document. Create a new version instead."
        
        elif request == 'submit_for_review':
            return "Published document cannot be submitted for review."
        
        elif request == 'add_review_comment':
            return "Cannot add comments to published document."
        
        elif request == 'approve_document':
            return "Published document is already approved."
        
        elif request == 'reject_document':
            return "Cannot reject published document."
        
        elif request == 'publish_document':
            return "Document is already published."
        
        elif request == 'archive_document':
            archiver = args[0]
            
            if archiver != context.author:
                return f"Only the author ({context.author}) can archive the document."
            
            context.set_state(ArchivedState())
            return f"Published document archived by {archiver}."
        
        else:
            return f"Invalid request '{request}' in published state."
    
    def get_state_name(self) -> str:
        return "Published"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Archived"]


class ArchivedState(State):
    """Archived state - document is read-only and inactive."""
    
    def handle_request(self, context: Document, request: str, *args, **kwargs) -> Any:
        # All modification requests are denied in archived state
        modification_requests = [
            'edit_content', 'submit_for_review', 'add_review_comment',
            'approve_document', 'reject_document', 'publish_document', 'archive_document'
        ]
        
        if request in modification_requests:
            return f"Cannot perform '{request}' on archived document."
        else:
            return f"Invalid request '{request}' in archived state."
    
    def get_state_name(self) -> str:
        return "Archived"
    
    def get_allowed_transitions(self) -> List[str]:
        return []  # No transitions allowed from archived state


# ============================================================================
# NETWORK CONNECTION STATES
# ============================================================================

class NetworkConnection(Context):
    """Network connection with state-dependent behavior."""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.server_address = None
        self.port = None
        self.connection_time = None
        self.data_sent = 0
        self.data_received = 0
        self.error_count = 0
        self.last_activity = datetime.now()
        
        # Initialize with disconnected state
        super().__init__(DisconnectedState())
    
    def connect(self, server_address: str, port: int) -> str:
        """Connect to server."""
        return self.handle_request('connect', server_address, port)
    
    def disconnect(self) -> str:
        """Disconnect from server."""
        return self.handle_request('disconnect')
    
    def send_data(self, data: str) -> str:
        """Send data through connection."""
        return self.handle_request('send_data', data)
    
    def receive_data(self) -> str:
        """Receive data from connection."""
        return self.handle_request('receive_data')
    
    def handle_error(self, error: str) -> str:
        """Handle connection error."""
        return self.handle_request('handle_error', error)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            'connection_id': self.connection_id,
            'state': self.get_current_state_name(),
            'server_address': self.server_address,
            'port': self.port,
            'connected_at': self.connection_time.isoformat() if self.connection_time else None,
            'data_sent': self.data_sent,
            'data_received': self.data_received,
            'error_count': self.error_count,
            'last_activity': self.last_activity.isoformat()
        }


class DisconnectedState(State):
    """Disconnected state - can only connect."""
    
    def handle_request(self, context: NetworkConnection, request: str, *args, **kwargs) -> Any:
        if request == 'connect':
            server_address, port = args
            
            if not server_address or not (1 <= port <= 65535):
                return "Invalid server address or port."
            
            # Simulate connection attempt
            context.server_address = server_address
            context.port = port
            context.connection_time = datetime.now()
            context.last_activity = datetime.now()
            
            # Transition to connecting state
            context.set_state(ConnectingState())
            return f"Connecting to {server_address}:{port}..."
        
        elif request == 'disconnect':
            return "Already disconnected."
        
        elif request == 'send_data':
            return "Cannot send data. Not connected."
        
        elif request == 'receive_data':
            return "Cannot receive data. Not connected."
        
        elif request == 'handle_error':
            return "No active connection to handle errors."
        
        else:
            return f"Invalid request '{request}' in disconnected state."
    
    def get_state_name(self) -> str:
        return "Disconnected"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Connecting"]


class ConnectingState(State):
    """Connecting state - attempting to establish connection."""
    
    def handle_request(self, context: NetworkConnection, request: str, *args, **kwargs) -> Any:
        if request == 'connect':
            return "Already attempting to connect."
        
        elif request == 'disconnect':
            # Cancel connection attempt
            context.server_address = None
            context.port = None
            context.connection_time = None
            
            context.set_state(DisconnectedState())
            return "Connection attempt cancelled."
        
        elif request == 'send_data':
            return "Cannot send data while connecting."
        
        elif request == 'receive_data':
            return "Cannot receive data while connecting."
        
        elif request == 'handle_error':
            error = args[0]
            context.error_count += 1
            
            # Connection failed, return to disconnected state
            context.server_address = None
            context.port = None
            context.connection_time = None
            
            context.set_state(DisconnectedState())
            return f"Connection failed: {error}"
        
        else:
            return f"Invalid request '{request}' in connecting state."
    
    def enter_state(self, context: NetworkConnection) -> None:
        """Simulate connection establishment."""
        super().enter_state(context)
        
        # Simulate connection delay and success
        def establish_connection():
            time.sleep(0.1)  # Simulate connection time
            
            # Simulate 90% success rate
            import random
            if random.random() < 0.9:
                # Connection successful
                context.set_state(ConnectedState())
            else:
                # Connection failed
                context.handle_error("Connection timeout")
        
        # Start connection in background (simplified for demo)
        threading.Thread(target=establish_connection, daemon=True).start()
    
    def get_state_name(self) -> str:
        return "Connecting"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Connected", "Disconnected"]


class ConnectedState(State):
    """Connected state - can send/receive data."""
    
    def handle_request(self, context: NetworkConnection, request: str, *args, **kwargs) -> Any:
        if request == 'connect':
            return f"Already connected to {context.server_address}:{context.port}."
        
        elif request == 'disconnect':
            context.set_state(DisconnectedState())
            return f"Disconnected from {context.server_address}:{context.port}."
        
        elif request == 'send_data':
            data = args[0]
            
            if not data:
                return "Cannot send empty data."
            
            # Update statistics
            context.data_sent += len(data)
            context.last_activity = datetime.now()
            
            return f"Sent {len(data)} bytes: '{data[:50]}{'...' if len(data) > 50 else ''}'"
        
        elif request == 'receive_data':
            # Simulate receiving data
            import random
            received_data = f"Response_{random.randint(1000, 9999)}"
            
            context.data_received += len(received_data)
            context.last_activity = datetime.now()
            
            return f"Received {len(received_data)} bytes: '{received_data}'"
        
        elif request == 'handle_error':
            error = args[0]
            context.error_count += 1
            
            # Determine if error is recoverable
            recoverable_errors = ["timeout", "packet_loss", "retry_needed"]
            
            if any(err in error.lower() for err in recoverable_errors):
                # Recoverable error - stay connected but log it
                return f"Recoverable error handled: {error}"
            else:
                # Fatal error - disconnect
                context.set_state(DisconnectedState())
                return f"Fatal error, disconnected: {error}"
        
        else:
            return f"Invalid request '{request}' in connected state."
    
    def get_state_name(self) -> str:
        return "Connected"
    
    def get_allowed_transitions(self) -> List[str]:
        return ["Disconnected"]


# ============================================================================
# STATE MACHINE FACTORY
# ============================================================================

class StateMachineFactory:
    """Factory for creating different types of state machines."""
    
    @staticmethod
    def create_vending_machine() -> VendingMachine:
        """Create a vending machine state machine."""
        return VendingMachine()
    
    @staticmethod
    def create_document_workflow(title: str, author: str) -> Document:
        """Create a document workflow state machine."""
        return Document(title, author)
    
    @staticmethod
    def create_network_connection(connection_id: str) -> NetworkConnection:
        """Create a network connection state machine."""
        return NetworkConnection(connection_id)
    
    @staticmethod
    def get_available_state_machines() -> List[str]:
        """Get list of available state machine types."""
        return ["VendingMachine", "DocumentWorkflow", "NetworkConnection"]


def demonstrate_state_pattern():
    """
    Demonstrate State pattern implementations.
    """
    print("=== STATE PATTERN DEMONSTRATION ===\n")
    
    # 1. Vending Machine State Machine
    print("1. VENDING MACHINE STATE MACHINE:")
    
    # Create vending machine
    vending_machine = VendingMachine()
    
    print("   Initial state:")
    print(f"   State: {vending_machine.get_current_state_name()}")
    print(f"   Balance: ${vending_machine.balance:.2f}")
    print()
    
    # Test vending machine operations
    operations = [
        ("insert_money", 1.00),
        ("select_item", "A1"),  # Coke for $1.50
        ("insert_money", 0.75),  # Add more money
        ("dispense_item",),
        ("insert_money", 2.00),
        ("select_item", "B1"),  # Chips for $2.00
        ("dispense_item",),
        ("return_money",)
    ]
    
    print("   Testing vending machine operations:")
    for operation in operations:
        method_name = operation[0]
        args = operation[1:] if len(operation) > 1 else ()
        
        result = getattr(vending_machine, method_name)(*args)
        print(f"   {method_name}({', '.join(map(str, args))}): {result}")
        
        status = vending_machine.get_machine_status()
        print(f"     State: {status['current_state']}, Balance: ${status['balance']:.2f}")
        print()
    
    # Show machine statistics
    stats = vending_machine.get_statistics()
    print(f"   Machine Statistics:")
    print(f"     Total transitions: {stats['total_transitions']}")
    print(f"     State usage: {stats['state_counts']}")
    print(f"     Most visited state: {stats['most_visited_state']}")
    
    print()
    
    # 2. Document Workflow State Machine
    print("2. DOCUMENT WORKFLOW STATE MACHINE:")
    
    # Create document
    document = Document("API Documentation", "Alice")
    
    print("   Document workflow simulation:")
    
    # Edit document
    result = document.edit_content("This is the API documentation content.", "Alice")
    print(f"   Edit: {result}")
    print(f"   State: {document.get_current_state_name()}")
    print()
    
    # Submit for review
    result = document.submit_for_review(["Bob", "Charlie"])
    print(f"   Submit for review: {result}")
    print(f"   State: {document.get_current_state_name()}")
    print()
    
    # Add review comments
    result = document.add_review_comment("Bob", "Looks good overall, minor formatting issues.", 4)
    print(f"   Bob's review: {result}")
    
    result = document.add_review_comment("Charlie", "Excellent documentation, very clear.", 5)
    print(f"   Charlie's review: {result}")
    print()
    
    # Try to approve (should work now)
    result = document.approve_document("Bob")
    print(f"   Approve: {result}")
    print(f"   State: {document.get_current_state_name()}")
    print()
    
    # Publish document
    result = document.publish_document("Alice")
    print(f"   Publish: {result}")
    print(f"   State: {document.get_current_state_name()}")
    print()
    
    # Show document info
    doc_info = document.get_document_info()
    print(f"   Document Information:")
    print(f"     Title: {doc_info['title']}")
    print(f"     Status: {doc_info['status']}")
    print(f"     Comments: {doc_info['comments_count']}")
    print(f"     Published: {doc_info['published_at']}")
    
    print()
    
    # 3. Document Rejection Workflow
    print("3. DOCUMENT REJECTION WORKFLOW:")
    
    # Create another document to demonstrate rejection
    doc2 = Document("User Guide", "David")
    
    # Edit and submit
    doc2.edit_content("Basic user guide content.", "David")
    doc2.submit_for_review(["Eve"])
    
    print(f"   Document state: {doc2.get_current_state_name()}")
    
    # Reject document
    result = doc2.reject_document("Eve", "Content is too basic, needs more detail.")
    print(f"   Rejection: {result}")
    print(f"   New state: {doc2.get_current_state_name()}")
    
    # Edit again after rejection
    result = doc2.edit_content("Comprehensive user guide with detailed examples.", "David")
    print(f"   Re-edit: {result}")
    
    print()
    
    # 4. Network Connection State Machine
    print("4. NETWORK CONNECTION STATE MACHINE:")
    
    # Create network connection
    connection = NetworkConnection("conn_001")
    
    print(f"   Initial state: {connection.get_current_state_name()}")
    
    # Test connection operations
    connection_ops = [
        ("connect", "api.example.com", 443),
        ("send_data", "GET /api/users HTTP/1.1"),
        ("receive_data",),
        ("send_data", "POST /api/data HTTP/1.1"),
        ("handle_error", "timeout"),  # Recoverable error
        ("send_data", "GET /api/status HTTP/1.1"),
        ("handle_error", "connection_reset"),  # Fatal error
        ("connect", "backup.example.com", 443)
    ]
    
    print("   Testing network operations:")
    for operation in connection_ops:
        method_name = operation[0]
        args = operation[1:] if len(operation) > 1 else ()
        
        # Add delay to allow connecting state to transition
        if method_name == "send_data" and connection.get_current_state_name() == "Connecting":
            time.sleep(0.15)  # Wait for connection to establish
        
        result = getattr(connection, method_name)(*args)
        print(f"   {method_name}({', '.join(map(str, args))}): {result}")
        
        conn_info = connection.get_connection_info()
        print(f"     State: {conn_info['state']}, Data sent: {conn_info['data_sent']} bytes")
        print()
    
    print()
    
    # 5. State Transition Validation
    print("5. STATE TRANSITION VALIDATION:")
    
    # Create vending machine to test invalid transitions
    vm = VendingMachine()
    
    print("   Testing invalid state transitions:")
    
    # Try invalid operations in idle state
    invalid_ops = [
        ("select_item", "A1"),  # Should fail - no money
        ("dispense_item",),     # Should fail - no selection
        ("return_money",)       # Should fail - no money
    ]
    
    for operation in invalid_ops:
        method_name = operation[0]
        args = operation[1:] if len(operation) > 1 else ()
        
        result = getattr(vm, method_name)(*args)
        print(f"   {method_name}({', '.join(map(str, args))}): {result}")
    
    print()
    
    # 6. State History and Analytics
    print("6. STATE HISTORY AND ANALYTICS:")
    
    # Show state history for vending machine
    history = vending_machine.get_state_history()
    print("   Vending machine state history:")
    for i, record in enumerate(history[-5:], 1):  # Show last 5 transitions
        print(f"     {i}. {record['from_state']} → {record['to_state']} "
              f"(Transition #{record['transition_number']})")
    
    print()
    
    # Show document state history
    doc_history = document.get_state_history()
    print("   Document workflow state history:")
    for i, record in enumerate(doc_history, 1):
        print(f"     {i}. {record['from_state']} → {record['to_state']}")
    
    print()
    
    # 7. State Machine Factory
    print("7. STATE MACHINE FACTORY:")
    
    factory = StateMachineFactory()
    available_machines = factory.get_available_state_machines()
    
    print(f"   Available state machines: {available_machines}")
    
    # Create instances using factory
    vm_factory = factory.create_vending_machine()
    doc_factory = factory.create_document_workflow("Factory Doc", "Factory Author")
    conn_factory = factory.create_network_connection("factory_conn")
    
    print(f"   Created vending machine: {vm_factory.get_current_state_name()}")
    print(f"   Created document: {doc_factory.get_current_state_name()}")
    print(f"   Created connection: {conn_factory.get_current_state_name()}")
    
    print()
    
    # 8. State Pattern Benefits
    print("8. STATE PATTERN BENEFITS:")
    print("   ✓ State Encapsulation: Each state encapsulates its behavior")
    print("   ✓ Clean Transitions: State transitions are explicit and controlled")
    print("   ✓ Eliminates Conditionals: Reduces complex if-else chains")
    print("   ✓ Easy Extension: New states can be added without modifying existing code")
    print("   ✓ State-Specific Validation: Each state validates its own operations")
    print("   ✓ Maintainability: State logic is separated and organized")
    print("   ✓ Debugging: State transitions are trackable and auditable")
    print("   ✓ Consistency: Ensures object behavior is consistent with its state")
    print("   ✓ Flexibility: States can have different implementations of same interface")
    print("   ✓ Reusability: States can be reused across different contexts")
    print()
    
    print("=== STATE PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_state_pattern()
