"""
ABSTRACTION CONCEPTS - Abstract Classes and Interfaces
======================================================

Problem Statement:
Demonstrate abstraction concepts including:
- Abstract base classes and abstract methods
- Interface design and implementation
- Hiding implementation complexity
- Creating clean abstractions
- Template method pattern with abstraction

Learning Objectives:
- Understand abstraction as a design principle
- Create and use abstract base classes
- Design clean interfaces
- Hide implementation details effectively
- Use abstraction for code organization
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import json


# Enumeration for payment status
class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


# Abstract base class for payment processing
class PaymentProcessor(ABC):
    """
    Abstract base class for payment processing.
    Defines the contract that all payment processors must follow.
    """
    
    def __init__(self, merchant_id: str, api_key: str):
        """Initialize payment processor with credentials."""
        self.merchant_id = merchant_id
        self.api_key = api_key
        self.transaction_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def validate_payment_details(self, payment_details: Dict[str, Any]) -> bool:
        """
        Validate payment details (abstract method).
        Each processor has different validation rules.
        """
        pass
    
    @abstractmethod
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process payment (abstract method).
        Each processor has different processing logic.
        """
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Refund payment (abstract method).
        Each processor has different refund mechanisms.
        """
        pass
    
    @abstractmethod
    def get_transaction_status(self, transaction_id: str) -> PaymentStatus:
        """
        Get transaction status (abstract method).
        Each processor has different status checking methods.
        """
        pass
    
    # Concrete methods (template methods)
    def initiate_payment(self, amount: float, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template method for payment initiation.
        Uses abstract methods but provides common workflow.
        """
        print(f"Initiating payment of ${amount:.2f}")
        
        # Step 1: Validate payment details
        if not self.validate_payment_details(payment_details):
            return {
                'success': False,
                'error': 'Invalid payment details',
                'transaction_id': None
            }
        
        # Step 2: Process payment
        result = self.process_payment(amount, payment_details)
        
        # Step 3: Log transaction
        self._log_transaction(result)
        
        return result
    
    def _log_transaction(self, transaction_result: Dict[str, Any]) -> None:
        """Log transaction (private method)."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'processor': self.__class__.__name__,
            'result': transaction_result
        }
        self.transaction_history.append(log_entry)
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """Get transaction history."""
        return self.transaction_history.copy()


# Concrete implementation for Credit Card processing
class CreditCardProcessor(PaymentProcessor):
    """
    Credit card payment processor implementation.
    """
    
    def __init__(self, merchant_id: str, api_key: str, gateway_url: str):
        """Initialize credit card processor."""
        super().__init__(merchant_id, api_key)
        self.gateway_url = gateway_url
        self.supported_cards = ['visa', 'mastercard', 'amex', 'discover']
    
    def validate_payment_details(self, payment_details: Dict[str, Any]) -> bool:
        """Validate credit card details."""
        required_fields = ['card_number', 'expiry_month', 'expiry_year', 'cvv', 'cardholder_name']
        
        # Check required fields
        for field in required_fields:
            if field not in payment_details:
                print(f"Missing required field: {field}")
                return False
        
        # Validate card number (simplified Luhn check)
        card_number = payment_details['card_number'].replace(' ', '').replace('-', '')
        if not self._luhn_check(card_number):
            print("Invalid card number")
            return False
        
        # Validate expiry date
        expiry_month = payment_details['expiry_month']
        expiry_year = payment_details['expiry_year']
        if not (1 <= expiry_month <= 12):
            print("Invalid expiry month")
            return False
        
        current_year = datetime.now().year
        if expiry_year < current_year:
            print("Card has expired")
            return False
        
        # Validate CVV
        cvv = str(payment_details['cvv'])
        if not (3 <= len(cvv) <= 4 and cvv.isdigit()):
            print("Invalid CVV")
            return False
        
        return True
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process credit card payment."""
        transaction_id = f"CC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(payment_details['card_number']) % 10000}"
        
        # Simulate payment processing
        print(f"Processing credit card payment via {self.gateway_url}")
        
        # Simulate success/failure (90% success rate)
        import random
        success = random.random() > 0.1
        
        if success:
            return {
                'success': True,
                'transaction_id': transaction_id,
                'amount': amount,
                'status': PaymentStatus.COMPLETED.value,
                'processor': 'CreditCard',
                'message': 'Payment processed successfully'
            }
        else:
            return {
                'success': False,
                'transaction_id': transaction_id,
                'amount': amount,
                'status': PaymentStatus.FAILED.value,
                'processor': 'CreditCard',
                'error': 'Payment declined by bank'
            }
    
    def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """Refund credit card payment."""
        print(f"Processing credit card refund for transaction {transaction_id}")
        
        # Find original transaction
        original_transaction = None
        for log_entry in self.transaction_history:
            if log_entry['result'].get('transaction_id') == transaction_id:
                original_transaction = log_entry['result']
                break
        
        if not original_transaction:
            return {
                'success': False,
                'error': 'Transaction not found'
            }
        
        refund_amount = amount or original_transaction['amount']
        
        return {
            'success': True,
            'refund_id': f"REF_{transaction_id}",
            'amount': refund_amount,
            'status': PaymentStatus.REFUNDED.value,
            'message': 'Refund processed successfully'
        }
    
    def get_transaction_status(self, transaction_id: str) -> PaymentStatus:
        """Get credit card transaction status."""
        for log_entry in self.transaction_history:
            if log_entry['result'].get('transaction_id') == transaction_id:
                status_str = log_entry['result'].get('status', 'pending')
                return PaymentStatus(status_str)
        
        return PaymentStatus.PENDING
    
    def _luhn_check(self, card_number: str) -> bool:
        """Implement Luhn algorithm for card validation."""
        digits = [int(d) for d in card_number if d.isdigit()]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0


# Concrete implementation for PayPal processing
class PayPalProcessor(PaymentProcessor):
    """
    PayPal payment processor implementation.
    """
    
    def __init__(self, merchant_id: str, api_key: str, client_secret: str):
        """Initialize PayPal processor."""
        super().__init__(merchant_id, api_key)
        self.client_secret = client_secret
        self.sandbox_mode = True
    
    def validate_payment_details(self, payment_details: Dict[str, Any]) -> bool:
        """Validate PayPal payment details."""
        required_fields = ['email', 'amount']
        
        for field in required_fields:
            if field not in payment_details:
                print(f"Missing required field: {field}")
                return False
        
        # Validate email format
        email = payment_details['email']
        if '@' not in email or '.' not in email.split('@')[1]:
            print("Invalid email format")
            return False
        
        return True
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process PayPal payment."""
        transaction_id = f"PP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(payment_details['email']) % 10000}"
        
        print(f"Processing PayPal payment for {payment_details['email']}")
        
        # Simulate PayPal processing
        import random
        success = random.random() > 0.05  # 95% success rate
        
        if success:
            return {
                'success': True,
                'transaction_id': transaction_id,
                'amount': amount,
                'status': PaymentStatus.COMPLETED.value,
                'processor': 'PayPal',
                'payer_email': payment_details['email'],
                'message': 'PayPal payment completed'
            }
        else:
            return {
                'success': False,
                'transaction_id': transaction_id,
                'amount': amount,
                'status': PaymentStatus.FAILED.value,
                'processor': 'PayPal',
                'error': 'Insufficient funds in PayPal account'
            }
    
    def refund_payment(self, transaction_id: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """Refund PayPal payment."""
        print(f"Processing PayPal refund for transaction {transaction_id}")
        
        return {
            'success': True,
            'refund_id': f"PPREF_{transaction_id}",
            'amount': amount,
            'status': PaymentStatus.REFUNDED.value,
            'message': 'PayPal refund initiated'
        }
    
    def get_transaction_status(self, transaction_id: str) -> PaymentStatus:
        """Get PayPal transaction status."""
        for log_entry in self.transaction_history:
            if log_entry['result'].get('transaction_id') == transaction_id:
                status_str = log_entry['result'].get('status', 'pending')
                return PaymentStatus(status_str)
        
        return PaymentStatus.PENDING


# Abstract class for data storage
class DataStorage(ABC):
    """
    Abstract base class for data storage systems.
    Provides abstraction over different storage mechanisms.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to storage system."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from storage system."""
        pass
    
    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """Save data with given key."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Any:
        """Load data by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        """List all keys."""
        pass


# Concrete implementation for file storage
class FileStorage(DataStorage):
    """
    File-based storage implementation.
    """
    
    def __init__(self, base_directory: str = "./data"):
        """Initialize file storage."""
        self.base_directory = base_directory
        self.is_connected = False
        import os
        os.makedirs(base_directory, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to file system."""
        self.is_connected = True
        print(f"Connected to file storage: {self.base_directory}")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from file system."""
        self.is_connected = False
        print("Disconnected from file storage")
        return True
    
    def save(self, key: str, data: Any) -> bool:
        """Save data to file."""
        if not self.is_connected:
            print("Not connected to storage")
            return False
        
        try:
            import os
            file_path = os.path.join(self.base_directory, f"{key}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def load(self, key: str) -> Any:
        """Load data from file."""
        if not self.is_connected:
            print("Not connected to storage")
            return None
        
        try:
            import os
            file_path = os.path.join(self.base_directory, f"{key}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Data loaded from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Key '{key}' not found")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete file."""
        if not self.is_connected:
            print("Not connected to storage")
            return False
        
        try:
            import os
            file_path = os.path.join(self.base_directory, f"{key}.json")
            os.remove(file_path)
            print(f"Deleted {file_path}")
            return True
        except FileNotFoundError:
            print(f"Key '{key}' not found")
            return False
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        import os
        file_path = os.path.join(self.base_directory, f"{key}.json")
        return os.path.exists(file_path)
    
    def list_keys(self) -> List[str]:
        """List all keys (files)."""
        import os
        try:
            files = os.listdir(self.base_directory)
            keys = [f.replace('.json', '') for f in files if f.endswith('.json')]
            return keys
        except Exception as e:
            print(f"Error listing keys: {e}")
            return []


# Concrete implementation for memory storage
class MemoryStorage(DataStorage):
    """
    In-memory storage implementation.
    """
    
    def __init__(self):
        """Initialize memory storage."""
        self.data_store: Dict[str, Any] = {}
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to memory storage."""
        self.is_connected = True
        print("Connected to memory storage")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from memory storage."""
        self.is_connected = False
        print("Disconnected from memory storage")
        return True
    
    def save(self, key: str, data: Any) -> bool:
        """Save data to memory."""
        if not self.is_connected:
            print("Not connected to storage")
            return False
        
        self.data_store[key] = data
        print(f"Data saved to memory with key '{key}'")
        return True
    
    def load(self, key: str) -> Any:
        """Load data from memory."""
        if not self.is_connected:
            print("Not connected to storage")
            return None
        
        if key in self.data_store:
            print(f"Data loaded from memory with key '{key}'")
            return self.data_store[key]
        else:
            print(f"Key '{key}' not found in memory")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from memory."""
        if not self.is_connected:
            print("Not connected to storage")
            return False
        
        if key in self.data_store:
            del self.data_store[key]
            print(f"Deleted key '{key}' from memory")
            return True
        else:
            print(f"Key '{key}' not found in memory")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self.data_store
    
    def list_keys(self) -> List[str]:
        """List all keys in memory."""
        return list(self.data_store.keys())


# High-level abstraction using the storage systems
class DocumentManager:
    """
    Document manager that uses abstract storage.
    Demonstrates how abstraction hides implementation details.
    """
    
    def __init__(self, storage: DataStorage):
        """Initialize with any storage implementation."""
        self.storage = storage
        self.storage.connect()
    
    def save_document(self, doc_id: str, title: str, content: str, author: str) -> bool:
        """Save document using abstract storage."""
        document = {
            'id': doc_id,
            'title': title,
            'content': content,
            'author': author,
            'created_at': datetime.now().isoformat(),
            'version': 1
        }
        
        return self.storage.save(doc_id, document)
    
    def load_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load document using abstract storage."""
        return self.storage.load(doc_id)
    
    def update_document(self, doc_id: str, title: str = None, content: str = None) -> bool:
        """Update existing document."""
        document = self.storage.load(doc_id)
        if not document:
            print(f"Document '{doc_id}' not found")
            return False
        
        if title:
            document['title'] = title
        if content:
            document['content'] = content
        
        document['version'] += 1
        document['updated_at'] = datetime.now().isoformat()
        
        return self.storage.save(doc_id, document)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document using abstract storage."""
        return self.storage.delete(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        doc_ids = self.storage.list_keys()
        documents = []
        
        for doc_id in doc_ids:
            doc = self.storage.load(doc_id)
            if doc:
                documents.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'author': doc['author'],
                    'created_at': doc['created_at'],
                    'version': doc['version']
                })
        
        return documents
    
    def close(self) -> None:
        """Close document manager."""
        self.storage.disconnect()


def demonstrate_abstraction():
    """
    Demonstrate abstraction concepts with practical examples.
    """
    print("=== ABSTRACTION CONCEPTS DEMONSTRATION ===\n")
    
    # 1. Payment Processing Abstraction
    print("1. Payment Processing Abstraction:")
    
    # Create different payment processors
    credit_card = CreditCardProcessor("MERCHANT123", "api_key_cc", "https://gateway.creditcard.com")
    paypal = PayPalProcessor("MERCHANT123", "api_key_pp", "client_secret_pp")
    
    # Credit card payment
    cc_payment_details = {
        'card_number': '4532123456789012',
        'expiry_month': 12,
        'expiry_year': 2025,
        'cvv': '123',
        'cardholder_name': 'John Doe'
    }
    
    cc_result = credit_card.initiate_payment(100.50, cc_payment_details)
    print(f"Credit Card Result: {cc_result}")
    
    # PayPal payment
    pp_payment_details = {
        'email': 'user@example.com',
        'amount': 75.25
    }
    
    pp_result = paypal.initiate_payment(75.25, pp_payment_details)
    print(f"PayPal Result: {pp_result}")
    
    # Refund payments
    if cc_result['success']:
        refund_result = credit_card.refund_payment(cc_result['transaction_id'], 50.0)
        print(f"Credit Card Refund: {refund_result}")
    
    print()
    
    # 2. Storage Abstraction
    print("2. Storage System Abstraction:")
    
    # Test with file storage
    print("Using File Storage:")
    file_storage = FileStorage("./test_data")
    doc_manager_file = DocumentManager(file_storage)
    
    doc_manager_file.save_document("doc1", "My First Document", "This is the content", "Alice")
    doc_manager_file.save_document("doc2", "Another Document", "More content here", "Bob")
    
    # Load and display documents
    doc1 = doc_manager_file.load_document("doc1")
    if doc1:
        print(f"Loaded document: {doc1['title']} by {doc1['author']}")
    
    # List all documents
    all_docs = doc_manager_file.list_documents()
    print(f"All documents: {len(all_docs)} found")
    for doc in all_docs:
        print(f"  - {doc['title']} (v{doc['version']}) by {doc['author']}")
    
    doc_manager_file.close()
    print()
    
    # Test with memory storage
    print("Using Memory Storage:")
    memory_storage = MemoryStorage()
    doc_manager_memory = DocumentManager(memory_storage)
    
    doc_manager_memory.save_document("mem_doc1", "Memory Document", "Stored in memory", "Charlie")
    doc_manager_memory.update_document("mem_doc1", content="Updated content in memory")
    
    mem_doc = doc_manager_memory.load_document("mem_doc1")
    if mem_doc:
        print(f"Memory document: {mem_doc['title']} (v{mem_doc['version']})")
    
    doc_manager_memory.close()
    print()
    
    # 3. Polymorphic Usage of Abstract Classes
    print("3. Polymorphic Usage of Abstract Classes:")
    
    # List of different payment processors
    processors = [credit_card, paypal]
    
    test_payments = [
        (50.0, cc_payment_details),
        (25.0, pp_payment_details)
    ]
    
    for i, processor in enumerate(processors):
        amount, details = test_payments[i]
        print(f"Processing payment with {processor.__class__.__name__}:")
        result = processor.initiate_payment(amount, details)
        print(f"  Result: {'Success' if result['success'] else 'Failed'}")
        
        if result['success']:
            status = processor.get_transaction_status(result['transaction_id'])
            print(f"  Status: {status.value}")
    
    print()
    
    # 4. Storage System Polymorphism
    print("4. Storage System Polymorphism:")
    
    storage_systems = [FileStorage("./poly_test"), MemoryStorage()]
    
    for storage in storage_systems:
        print(f"Testing {storage.__class__.__name__}:")
        storage.connect()
        
        # Test basic operations
        storage.save("test_key", {"message": "Hello from abstraction!"})
        data = storage.load("test_key")
        print(f"  Loaded: {data}")
        
        exists = storage.exists("test_key")
        print(f"  Exists: {exists}")
        
        keys = storage.list_keys()
        print(f"  Keys: {keys}")
        
        storage.disconnect()
        print()
    
    print("=== ABSTRACTION DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_abstraction()
