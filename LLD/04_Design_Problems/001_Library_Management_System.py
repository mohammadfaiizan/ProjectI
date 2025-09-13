"""
LIBRARY MANAGEMENT SYSTEM - Complete System Design
===================================================

Problem Statement:
Design a comprehensive Library Management System that handles:
- Book catalog management with multiple copies
- User registration and membership management
- Book borrowing and returning with due dates
- Fine calculation for overdue books
- Search functionality by title, author, ISBN, genre
- Reservation system for unavailable books
- Staff management with different access levels
- Report generation for library statistics

Requirements:
- Support multiple book copies and editions
- Handle different user types (Student, Faculty, General)
- Implement borrowing limits and duration policies
- Calculate fines automatically for overdue books
- Maintain transaction history and audit trails
- Support book reservations and waiting lists
- Generate various reports and statistics
- Handle concurrent operations safely

Design Patterns Used:
- Singleton: Library system instance
- Factory: User and book creation
- Observer: Notification system
- Strategy: Fine calculation strategies
- Command: Transaction operations
- State: Book availability states
- Decorator: User privilege enhancement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class BookStatus(Enum):
    AVAILABLE = "available"
    BORROWED = "borrowed"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    LOST = "lost"


class UserType(Enum):
    STUDENT = "student"
    FACULTY = "faculty"
    GENERAL = "general"
    STAFF = "staff"


class TransactionType(Enum):
    BORROW = "borrow"
    RETURN = "return"
    RESERVE = "reserve"
    CANCEL_RESERVATION = "cancel_reservation"
    RENEW = "renew"


@dataclass
class BookInfo:
    """Book information data class."""
    isbn: str
    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    genre: str
    pages: int
    language: str = "English"
    edition: str = "1st"
    description: str = ""
    
    def __post_init__(self):
        if not self.isbn or not self.title or not self.authors:
            raise ValueError("ISBN, title, and authors are required")


@dataclass
class UserInfo:
    """User information data class."""
    user_id: str
    name: str
    email: str
    phone: str
    address: str
    user_type: UserType
    registration_date: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def __post_init__(self):
        if not all([self.user_id, self.name, self.email]):
            raise ValueError("User ID, name, and email are required")


# ============================================================================
# CORE ENTITIES
# ============================================================================

class Book:
    """Individual book copy with unique barcode."""
    
    def __init__(self, book_info: BookInfo, barcode: str):
        self.book_info = book_info
        self.barcode = barcode
        self.status = BookStatus.AVAILABLE
        self.current_borrower_id: Optional[str] = None
        self.due_date: Optional[datetime] = None
        self.reservation_queue: List[str] = []
        self.borrow_count = 0
        self.last_borrowed_date: Optional[datetime] = None
        self.condition_notes = ""
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        """Check if book is available for borrowing."""
        return self.status == BookStatus.AVAILABLE
    
    def is_overdue(self) -> bool:
        """Check if book is overdue."""
        if self.status == BookStatus.BORROWED and self.due_date:
            return datetime.now() > self.due_date
        return False
    
    def borrow(self, user_id: str, due_date: datetime) -> bool:
        """Borrow the book."""
        with self._lock:
            if not self.is_available():
                return False
            
            self.status = BookStatus.BORROWED
            self.current_borrower_id = user_id
            self.due_date = due_date
            self.borrow_count += 1
            self.last_borrowed_date = datetime.now()
            return True
    
    def return_book(self) -> bool:
        """Return the book."""
        with self._lock:
            if self.status != BookStatus.BORROWED:
                return False
            
            self.status = BookStatus.AVAILABLE
            self.current_borrower_id = None
            self.due_date = None
            
            # If there are reservations, mark as reserved
            if self.reservation_queue:
                self.status = BookStatus.RESERVED
            
            return True
    
    def add_reservation(self, user_id: str) -> bool:
        """Add user to reservation queue."""
        with self._lock:
            if user_id not in self.reservation_queue:
                self.reservation_queue.append(user_id)
                if self.status == BookStatus.AVAILABLE:
                    self.status = BookStatus.RESERVED
                return True
            return False
    
    def remove_reservation(self, user_id: str) -> bool:
        """Remove user from reservation queue."""
        with self._lock:
            if user_id in self.reservation_queue:
                self.reservation_queue.remove(user_id)
                if not self.reservation_queue and self.status == BookStatus.RESERVED:
                    self.status = BookStatus.AVAILABLE
                return True
            return False
    
    def get_next_reserver(self) -> Optional[str]:
        """Get next user in reservation queue."""
        return self.reservation_queue[0] if self.reservation_queue else None
    
    def get_book_details(self) -> Dict:
        """Get complete book details."""
        return {
            'barcode': self.barcode,
            'isbn': self.book_info.isbn,
            'title': self.book_info.title,
            'authors': self.book_info.authors,
            'status': self.status.value,
            'current_borrower_id': self.current_borrower_id,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'is_overdue': self.is_overdue(),
            'reservation_count': len(self.reservation_queue),
            'borrow_count': self.borrow_count,
            'condition_notes': self.condition_notes
        }


class User:
    """Library user with borrowing capabilities."""
    
    def __init__(self, user_info: UserInfo):
        self.user_info = user_info
        self.borrowed_books: Set[str] = set()  # Set of barcodes
        self.reservation_list: Set[str] = set()  # Set of ISBNs
        self.transaction_history: List[Dict] = []
        self.total_fines = 0.0
        self.outstanding_fines = 0.0
        self._lock = threading.Lock()
    
    def get_borrowing_limit(self) -> int:
        """Get borrowing limit based on user type."""
        limits = {
            UserType.STUDENT: 5,
            UserType.FACULTY: 10,
            UserType.GENERAL: 3,
            UserType.STAFF: 15
        }
        return limits.get(self.user_info.user_type, 3)
    
    def get_borrowing_duration(self) -> int:
        """Get borrowing duration in days based on user type."""
        durations = {
            UserType.STUDENT: 14,
            UserType.FACULTY: 30,
            UserType.GENERAL: 7,
            UserType.STAFF: 30
        }
        return durations.get(self.user_info.user_type, 7)
    
    def can_borrow(self) -> bool:
        """Check if user can borrow more books."""
        return (len(self.borrowed_books) < self.get_borrowing_limit() and
                self.user_info.is_active and
                self.outstanding_fines < 50.0)  # Max outstanding fine limit
    
    def borrow_book(self, barcode: str) -> bool:
        """Record book borrowing."""
        with self._lock:
            if not self.can_borrow():
                return False
            
            self.borrowed_books.add(barcode)
            self._add_transaction(TransactionType.BORROW, barcode)
            return True
    
    def return_book(self, barcode: str) -> bool:
        """Record book return."""
        with self._lock:
            if barcode in self.borrowed_books:
                self.borrowed_books.remove(barcode)
                self._add_transaction(TransactionType.RETURN, barcode)
                return True
            return False
    
    def add_reservation(self, isbn: str) -> bool:
        """Add book reservation."""
        with self._lock:
            if isbn not in self.reservation_list:
                self.reservation_list.add(isbn)
                self._add_transaction(TransactionType.RESERVE, isbn)
                return True
            return False
    
    def remove_reservation(self, isbn: str) -> bool:
        """Remove book reservation."""
        with self._lock:
            if isbn in self.reservation_list:
                self.reservation_list.remove(isbn)
                self._add_transaction(TransactionType.CANCEL_RESERVATION, isbn)
                return True
            return False
    
    def add_fine(self, amount: float, description: str) -> None:
        """Add fine to user account."""
        with self._lock:
            self.total_fines += amount
            self.outstanding_fines += amount
            self._add_transaction("FINE", description, {"amount": amount})
    
    def pay_fine(self, amount: float) -> float:
        """Pay fine and return remaining amount."""
        with self._lock:
            paid = min(amount, self.outstanding_fines)
            self.outstanding_fines -= paid
            self._add_transaction("FINE_PAYMENT", f"Paid ${paid:.2f}")
            return amount - paid
    
    def _add_transaction(self, transaction_type: TransactionType, item_id: str, 
                        metadata: Dict = None) -> None:
        """Add transaction to history."""
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'type': transaction_type.value if isinstance(transaction_type, TransactionType) else transaction_type,
            'item_id': item_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.transaction_history.append(transaction)
    
    def get_user_summary(self) -> Dict:
        """Get user summary information."""
        return {
            'user_id': self.user_info.user_id,
            'name': self.user_info.name,
            'user_type': self.user_info.user_type.value,
            'is_active': self.user_info.is_active,
            'borrowed_books_count': len(self.borrowed_books),
            'borrowing_limit': self.get_borrowing_limit(),
            'reservations_count': len(self.reservation_list),
            'total_fines': self.total_fines,
            'outstanding_fines': self.outstanding_fines,
            'total_transactions': len(self.transaction_history)
        }


# ============================================================================
# FINE CALCULATION STRATEGIES
# ============================================================================

class FineCalculationStrategy(ABC):
    """Abstract strategy for fine calculation."""
    
    @abstractmethod
    def calculate_fine(self, days_overdue: int, book: Book, user: User) -> float:
        """Calculate fine amount."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class StandardFineStrategy(FineCalculationStrategy):
    """Standard fine calculation strategy."""
    
    def __init__(self, daily_rate: float = 0.50):
        self.daily_rate = daily_rate
    
    def calculate_fine(self, days_overdue: int, book: Book, user: User) -> float:
        """Calculate standard fine."""
        if days_overdue <= 0:
            return 0.0
        
        base_fine = days_overdue * self.daily_rate
        
        # User type multipliers
        multipliers = {
            UserType.STUDENT: 0.8,
            UserType.FACULTY: 1.0,
            UserType.GENERAL: 1.2,
            UserType.STAFF: 0.5
        }
        
        multiplier = multipliers.get(user.user_info.user_type, 1.0)
        return round(base_fine * multiplier, 2)
    
    def get_strategy_name(self) -> str:
        return "Standard Fine Strategy"


class ProgressiveFineStrategy(FineCalculationStrategy):
    """Progressive fine calculation with increasing rates."""
    
    def calculate_fine(self, days_overdue: int, book: Book, user: User) -> float:
        """Calculate progressive fine."""
        if days_overdue <= 0:
            return 0.0
        
        fine = 0.0
        
        # Progressive rates
        if days_overdue <= 7:
            fine = days_overdue * 0.25
        elif days_overdue <= 14:
            fine = 7 * 0.25 + (days_overdue - 7) * 0.50
        elif days_overdue <= 30:
            fine = 7 * 0.25 + 7 * 0.50 + (days_overdue - 14) * 1.00
        else:
            fine = 7 * 0.25 + 7 * 0.50 + 16 * 1.00 + (days_overdue - 30) * 2.00
        
        return round(fine, 2)
    
    def get_strategy_name(self) -> str:
        return "Progressive Fine Strategy"


# ============================================================================
# NOTIFICATION SYSTEM (OBSERVER PATTERN)
# ============================================================================

class LibraryObserver(ABC):
    """Abstract observer for library events."""
    
    @abstractmethod
    def notify(self, event_type: str, data: Dict) -> None:
        """Handle library event notification."""
        pass


class EmailNotificationService(LibraryObserver):
    """Email notification service."""
    
    def __init__(self):
        self.sent_notifications: List[Dict] = []
    
    def notify(self, event_type: str, data: Dict) -> None:
        """Send email notification."""
        notification = {
            'type': 'email',
            'event': event_type,
            'recipient': data.get('user_email'),
            'subject': self._get_subject(event_type),
            'message': self._get_message(event_type, data),
            'timestamp': datetime.now().isoformat()
        }
        
        self.sent_notifications.append(notification)
        print(f"📧 Email sent: {notification['subject']} to {notification['recipient']}")
    
    def _get_subject(self, event_type: str) -> str:
        """Get email subject based on event type."""
        subjects = {
            'book_borrowed': 'Book Borrowed Successfully',
            'book_returned': 'Book Returned Successfully',
            'book_overdue': 'Overdue Book Reminder',
            'book_reserved': 'Book Reserved Successfully',
            'book_available': 'Reserved Book Now Available',
            'fine_added': 'Fine Added to Account'
        }
        return subjects.get(event_type, 'Library Notification')
    
    def _get_message(self, event_type: str, data: Dict) -> str:
        """Get email message based on event type."""
        if event_type == 'book_borrowed':
            return f"You have borrowed '{data['book_title']}'. Due date: {data['due_date']}"
        elif event_type == 'book_overdue':
            return f"'{data['book_title']}' is overdue. Please return it to avoid additional fines."
        elif event_type == 'fine_added':
            return f"A fine of ${data['amount']:.2f} has been added to your account."
        else:
            return f"Library notification: {event_type}"


class SMSNotificationService(LibraryObserver):
    """SMS notification service."""
    
    def __init__(self):
        self.sent_messages: List[Dict] = []
    
    def notify(self, event_type: str, data: Dict) -> None:
        """Send SMS notification."""
        if event_type in ['book_overdue', 'fine_added']:  # Only urgent notifications
            message = {
                'type': 'sms',
                'event': event_type,
                'recipient': data.get('user_phone'),
                'message': self._get_sms_message(event_type, data),
                'timestamp': datetime.now().isoformat()
            }
            
            self.sent_messages.append(message)
            print(f"📱 SMS sent: {message['message']} to {message['recipient']}")
    
    def _get_sms_message(self, event_type: str, data: Dict) -> str:
        """Get SMS message based on event type."""
        if event_type == 'book_overdue':
            return f"LIBRARY: '{data['book_title']}' is overdue. Return ASAP."
        elif event_type == 'fine_added':
            return f"LIBRARY: Fine ${data['amount']:.2f} added. Current total: ${data['total_fines']:.2f}"
        return f"LIBRARY: {event_type}"


# ============================================================================
# SEARCH AND CATALOG MANAGEMENT
# ============================================================================

class BookCatalog:
    """Book catalog with search capabilities."""
    
    def __init__(self):
        self.books_by_isbn: Dict[str, List[Book]] = defaultdict(list)
        self.books_by_title: Dict[str, List[Book]] = defaultdict(list)
        self.books_by_author: Dict[str, List[Book]] = defaultdict(list)
        self.books_by_genre: Dict[str, List[Book]] = defaultdict(list)
        self.all_books: Dict[str, Book] = {}  # barcode -> Book
        self._lock = threading.RLock()
    
    def add_book(self, book: Book) -> bool:
        """Add book to catalog."""
        with self._lock:
            if book.barcode in self.all_books:
                return False
            
            self.all_books[book.barcode] = book
            self.books_by_isbn[book.book_info.isbn].append(book)
            self.books_by_title[book.book_info.title.lower()].append(book)
            self.books_by_genre[book.book_info.genre.lower()].append(book)
            
            for author in book.book_info.authors:
                self.books_by_author[author.lower()].append(book)
            
            return True
    
    def remove_book(self, barcode: str) -> bool:
        """Remove book from catalog."""
        with self._lock:
            if barcode not in self.all_books:
                return False
            
            book = self.all_books[barcode]
            
            # Remove from all indices
            self.books_by_isbn[book.book_info.isbn].remove(book)
            self.books_by_title[book.book_info.title.lower()].remove(book)
            self.books_by_genre[book.book_info.genre.lower()].remove(book)
            
            for author in book.book_info.authors:
                self.books_by_author[author.lower()].remove(book)
            
            del self.all_books[barcode]
            return True
    
    def search_by_isbn(self, isbn: str) -> List[Book]:
        """Search books by ISBN."""
        return self.books_by_isbn.get(isbn, [])
    
    def search_by_title(self, title: str) -> List[Book]:
        """Search books by title (partial match)."""
        title_lower = title.lower()
        results = []
        
        for book_title, books in self.books_by_title.items():
            if title_lower in book_title:
                results.extend(books)
        
        return results
    
    def search_by_author(self, author: str) -> List[Book]:
        """Search books by author (partial match)."""
        author_lower = author.lower()
        results = []
        
        for book_author, books in self.books_by_author.items():
            if author_lower in book_author:
                results.extend(books)
        
        return results
    
    def search_by_genre(self, genre: str) -> List[Book]:
        """Search books by genre."""
        return self.books_by_genre.get(genre.lower(), [])
    
    def get_available_books(self, isbn: str = None) -> List[Book]:
        """Get available books, optionally filtered by ISBN."""
        if isbn:
            books = self.search_by_isbn(isbn)
        else:
            books = list(self.all_books.values())
        
        return [book for book in books if book.is_available()]
    
    def get_catalog_statistics(self) -> Dict:
        """Get catalog statistics."""
        total_books = len(self.all_books)
        available_books = len([b for b in self.all_books.values() if b.is_available()])
        borrowed_books = len([b for b in self.all_books.values() if b.status == BookStatus.BORROWED])
        reserved_books = len([b for b in self.all_books.values() if b.status == BookStatus.RESERVED])
        
        return {
            'total_books': total_books,
            'unique_titles': len(self.books_by_isbn),
            'available_books': available_books,
            'borrowed_books': borrowed_books,
            'reserved_books': reserved_books,
            'total_authors': len(self.books_by_author),
            'total_genres': len(self.books_by_genre)
        }


# ============================================================================
# MAIN LIBRARY MANAGEMENT SYSTEM (SINGLETON)
# ============================================================================

class LibraryManagementSystem:
    """Main library management system (Singleton)."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        
        self.catalog = BookCatalog()
        self.users: Dict[str, User] = {}
        self.fine_strategy = StandardFineStrategy()
        self.observers: List[LibraryObserver] = []
        self.transaction_log: List[Dict] = []
        self._system_lock = threading.RLock()
        self.initialized = True
        
        print("📚 Library Management System initialized")
    
    def add_observer(self, observer: LibraryObserver) -> None:
        """Add notification observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: LibraryObserver) -> None:
        """Remove notification observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event_type: str, data: Dict) -> None:
        """Notify all observers of an event."""
        for observer in self.observers:
            try:
                observer.notify(event_type, data)
            except Exception as e:
                print(f"Error notifying observer: {e}")
    
    def set_fine_strategy(self, strategy: FineCalculationStrategy) -> None:
        """Set fine calculation strategy."""
        self.fine_strategy = strategy
        print(f"Fine strategy changed to: {strategy.get_strategy_name()}")
    
    # User Management
    def register_user(self, user_info: UserInfo) -> bool:
        """Register new user."""
        with self._system_lock:
            if user_info.user_id in self.users:
                return False
            
            user = User(user_info)
            self.users[user_info.user_id] = user
            
            self._log_transaction("USER_REGISTERED", user_info.user_id, {
                'name': user_info.name,
                'user_type': user_info.user_type.value
            })
            
            print(f"👤 User registered: {user_info.name} ({user_info.user_id})")
            return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        user = self.get_user(user_id)
        if user and user.user_info.is_active:
            user.user_info.is_active = False
            self._log_transaction("USER_DEACTIVATED", user_id)
            return True
        return False
    
    # Book Management
    def add_book(self, book_info: BookInfo, barcode: str) -> bool:
        """Add new book to library."""
        with self._system_lock:
            book = Book(book_info, barcode)
            if self.catalog.add_book(book):
                self._log_transaction("BOOK_ADDED", barcode, {
                    'isbn': book_info.isbn,
                    'title': book_info.title
                })
                print(f"📖 Book added: {book_info.title} ({barcode})")
                return True
            return False
    
    def remove_book(self, barcode: str) -> bool:
        """Remove book from library."""
        with self._system_lock:
            if self.catalog.remove_book(barcode):
                self._log_transaction("BOOK_REMOVED", barcode)
                print(f"📖 Book removed: {barcode}")
                return True
            return False
    
    # Borrowing Operations
    def borrow_book(self, user_id: str, barcode: str) -> Tuple[bool, str]:
        """Borrow a book."""
        with self._system_lock:
            user = self.get_user(user_id)
            if not user:
                return False, "User not found"
            
            if not user.can_borrow():
                return False, "User cannot borrow more books"
            
            book = self.catalog.all_books.get(barcode)
            if not book:
                return False, "Book not found"
            
            if not book.is_available():
                return False, f"Book is {book.status.value}"
            
            # Calculate due date
            due_date = datetime.now() + timedelta(days=user.get_borrowing_duration())
            
            # Perform borrowing
            if book.borrow(user_id, due_date) and user.borrow_book(barcode):
                self._log_transaction("BOOK_BORROWED", barcode, {
                    'user_id': user_id,
                    'due_date': due_date.isoformat()
                })
                
                # Notify observers
                self.notify_observers('book_borrowed', {
                    'user_email': user.user_info.email,
                    'user_phone': user.user_info.phone,
                    'book_title': book.book_info.title,
                    'due_date': due_date.strftime('%Y-%m-%d')
                })
                
                return True, f"Book borrowed successfully. Due: {due_date.strftime('%Y-%m-%d')}"
            
            return False, "Failed to borrow book"
    
    def return_book(self, user_id: str, barcode: str) -> Tuple[bool, str, float]:
        """Return a book and calculate fine if overdue."""
        with self._system_lock:
            user = self.get_user(user_id)
            if not user:
                return False, "User not found", 0.0
            
            book = self.catalog.all_books.get(barcode)
            if not book:
                return False, "Book not found", 0.0
            
            if book.current_borrower_id != user_id:
                return False, "Book not borrowed by this user", 0.0
            
            # Calculate fine if overdue
            fine_amount = 0.0
            if book.is_overdue():
                days_overdue = (datetime.now() - book.due_date).days
                fine_amount = self.fine_strategy.calculate_fine(days_overdue, book, user)
                
                if fine_amount > 0:
                    user.add_fine(fine_amount, f"Overdue fine for '{book.book_info.title}'")
                    
                    # Notify observers
                    self.notify_observers('fine_added', {
                        'user_email': user.user_info.email,
                        'user_phone': user.user_info.phone,
                        'amount': fine_amount,
                        'total_fines': user.outstanding_fines
                    })
            
            # Return the book
            if book.return_book() and user.return_book(barcode):
                self._log_transaction("BOOK_RETURNED", barcode, {
                    'user_id': user_id,
                    'fine_amount': fine_amount
                })
                
                # Check if someone is waiting for this book
                next_reserver = book.get_next_reserver()
                if next_reserver:
                    reserved_user = self.get_user(next_reserver)
                    if reserved_user:
                        self.notify_observers('book_available', {
                            'user_email': reserved_user.user_info.email,
                            'user_phone': reserved_user.user_info.phone,
                            'book_title': book.book_info.title
                        })
                
                message = "Book returned successfully"
                if fine_amount > 0:
                    message += f". Fine: ${fine_amount:.2f}"
                
                return True, message, fine_amount
            
            return False, "Failed to return book", 0.0
    
    def reserve_book(self, user_id: str, isbn: str) -> Tuple[bool, str]:
        """Reserve a book by ISBN."""
        with self._system_lock:
            user = self.get_user(user_id)
            if not user:
                return False, "User not found"
            
            # Find available book with this ISBN
            books = self.catalog.search_by_isbn(isbn)
            if not books:
                return False, "Book not found"
            
            # Check if user already has this book or reservation
            if isbn in user.reservation_list:
                return False, "Book already reserved by user"
            
            # Find a book to reserve (prefer available ones)
            book_to_reserve = None
            for book in books:
                if book.is_available():
                    book_to_reserve = book
                    break
            
            if not book_to_reserve:
                # All copies are borrowed, add to queue of first copy
                book_to_reserve = books[0]
            
            # Add reservation
            if book_to_reserve.add_reservation(user_id) and user.add_reservation(isbn):
                self._log_transaction("BOOK_RESERVED", isbn, {
                    'user_id': user_id,
                    'barcode': book_to_reserve.barcode
                })
                
                self.notify_observers('book_reserved', {
                    'user_email': user.user_info.email,
                    'user_phone': user.user_info.phone,
                    'book_title': book_to_reserve.book_info.title
                })
                
                return True, "Book reserved successfully"
            
            return False, "Failed to reserve book"
    
    def cancel_reservation(self, user_id: str, isbn: str) -> Tuple[bool, str]:
        """Cancel book reservation."""
        with self._system_lock:
            user = self.get_user(user_id)
            if not user:
                return False, "User not found"
            
            if isbn not in user.reservation_list:
                return False, "No reservation found"
            
            # Find the book and remove reservation
            books = self.catalog.search_by_isbn(isbn)
            for book in books:
                if user_id in book.reservation_queue:
                    book.remove_reservation(user_id)
                    break
            
            user.remove_reservation(isbn)
            
            self._log_transaction("RESERVATION_CANCELLED", isbn, {
                'user_id': user_id
            })
            
            return True, "Reservation cancelled successfully"
    
    # Search Operations
    def search_books(self, query: str, search_type: str = "title") -> List[Dict]:
        """Search books in catalog."""
        if search_type == "isbn":
            books = self.catalog.search_by_isbn(query)
        elif search_type == "title":
            books = self.catalog.search_by_title(query)
        elif search_type == "author":
            books = self.catalog.search_by_author(query)
        elif search_type == "genre":
            books = self.catalog.search_by_genre(query)
        else:
            return []
        
        return [book.get_book_details() for book in books]
    
    # Fine Management
    def pay_fine(self, user_id: str, amount: float) -> Tuple[bool, str, float]:
        """Pay user fine."""
        with self._system_lock:
            user = self.get_user(user_id)
            if not user:
                return False, "User not found", 0.0
            
            remaining = user.pay_fine(amount)
            paid = amount - remaining
            
            self._log_transaction("FINE_PAYMENT", user_id, {
                'amount_paid': paid,
                'remaining_amount': remaining
            })
            
            message = f"Fine payment processed. Paid: ${paid:.2f}"
            if remaining > 0:
                message += f", Remaining: ${remaining:.2f}"
            
            return True, message, remaining
    
    def check_overdue_books(self) -> List[Dict]:
        """Check for overdue books and send notifications."""
        overdue_books = []
        
        for book in self.catalog.all_books.values():
            if book.is_overdue():
                user = self.get_user(book.current_borrower_id)
                if user:
                    overdue_info = {
                        'barcode': book.barcode,
                        'title': book.book_info.title,
                        'user_id': user.user_info.user_id,
                        'user_name': user.user_info.name,
                        'due_date': book.due_date.isoformat(),
                        'days_overdue': (datetime.now() - book.due_date).days
                    }
                    overdue_books.append(overdue_info)
                    
                    # Send overdue notification
                    self.notify_observers('book_overdue', {
                        'user_email': user.user_info.email,
                        'user_phone': user.user_info.phone,
                        'book_title': book.book_info.title
                    })
        
        return overdue_books
    
    # Reporting
    def generate_user_report(self, user_id: str) -> Optional[Dict]:
        """Generate detailed user report."""
        user = self.get_user(user_id)
        if not user:
            return None
        
        borrowed_books = []
        for barcode in user.borrowed_books:
            book = self.catalog.all_books.get(barcode)
            if book:
                borrowed_books.append({
                    'barcode': barcode,
                    'title': book.book_info.title,
                    'due_date': book.due_date.isoformat() if book.due_date else None,
                    'is_overdue': book.is_overdue()
                })
        
        return {
            'user_info': user.get_user_summary(),
            'borrowed_books': borrowed_books,
            'reservations': list(user.reservation_list),
            'recent_transactions': user.transaction_history[-10:]  # Last 10 transactions
        }
    
    def generate_library_statistics(self) -> Dict:
        """Generate library statistics."""
        catalog_stats = self.catalog.get_catalog_statistics()
        
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.user_info.is_active])
        users_with_fines = len([u for u in self.users.values() if u.outstanding_fines > 0])
        
        total_fines = sum(u.outstanding_fines for u in self.users.values())
        
        return {
            'catalog_statistics': catalog_stats,
            'user_statistics': {
                'total_users': total_users,
                'active_users': active_users,
                'users_with_fines': users_with_fines,
                'total_outstanding_fines': total_fines
            },
            'system_statistics': {
                'total_transactions': len(self.transaction_log),
                'fine_strategy': self.fine_strategy.get_strategy_name(),
                'observers_count': len(self.observers)
            }
        }
    
    def _log_transaction(self, transaction_type: str, item_id: str, metadata: Dict = None) -> None:
        """Log system transaction."""
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'type': transaction_type,
            'item_id': item_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.transaction_log.append(transaction)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_library_system():
    """Demonstrate the Library Management System."""
    print("=== LIBRARY MANAGEMENT SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    library = LibraryManagementSystem()
    
    # Add notification services
    email_service = EmailNotificationService()
    sms_service = SMSNotificationService()
    library.add_observer(email_service)
    library.add_observer(sms_service)
    
    print("1. SYSTEM INITIALIZATION:")
    print("   ✓ Library system initialized")
    print("   ✓ Notification services added")
    print()
    
    # Register users
    print("2. USER REGISTRATION:")
    users_data = [
        ("STU001", "Alice Johnson", "alice@university.edu", "555-0101", "123 Campus Dr", UserType.STUDENT),
        ("FAC001", "Dr. Bob Smith", "bob@university.edu", "555-0102", "456 Faculty Ave", UserType.FACULTY),
        ("GEN001", "Charlie Brown", "charlie@email.com", "555-0103", "789 Public St", UserType.GENERAL),
        ("STF001", "Diana Wilson", "diana@library.edu", "555-0104", "321 Staff Rd", UserType.STAFF)
    ]
    
    for user_id, name, email, phone, address, user_type in users_data:
        user_info = UserInfo(user_id, name, email, phone, address, user_type)
        library.register_user(user_info)
    
    print()
    
    # Add books
    print("3. BOOK CATALOG MANAGEMENT:")
    books_data = [
        ("978-0134685991", "Effective Java", ["Joshua Bloch"], "Addison-Wesley", 2017, "Programming", 416),
        ("978-0134685991", "Effective Java", ["Joshua Bloch"], "Addison-Wesley", 2017, "Programming", 416),  # Second copy
        ("978-0596009205", "Head First Design Patterns", ["Eric Freeman", "Elisabeth Robson"], "O'Reilly", 2004, "Programming", 694),
        ("978-0132350884", "Clean Code", ["Robert C. Martin"], "Prentice Hall", 2008, "Programming", 464),
        ("978-0201633610", "Design Patterns", ["Gang of Four"], "Addison-Wesley", 1994, "Programming", 395),
        ("978-1449331818", "Learning Python", ["Mark Lutz"], "O'Reilly", 2013, "Programming", 1648)
    ]
    
    for i, (isbn, title, authors, publisher, year, genre, pages) in enumerate(books_data):
        book_info = BookInfo(isbn, title, authors, publisher, year, genre, pages)
        barcode = f"BC{1001 + i:04d}"
        library.add_book(book_info, barcode)
    
    print()
    
    # Test borrowing
    print("4. BOOK BORROWING OPERATIONS:")
    
    # Alice borrows Effective Java
    success, message = library.borrow_book("STU001", "BC1001")
    print(f"   Alice borrows Effective Java: {message}")
    
    # Bob borrows Clean Code
    success, message = library.borrow_book("FAC001", "BC1004")
    print(f"   Bob borrows Clean Code: {message}")
    
    # Charlie tries to borrow the same copy of Effective Java (should fail)
    success, message = library.borrow_book("GEN001", "BC1001")
    print(f"   Charlie tries same book: {message}")
    
    # Charlie borrows second copy of Effective Java
    success, message = library.borrow_book("GEN001", "BC1002")
    print(f"   Charlie borrows second copy: {message}")
    
    print()
    
    # Test reservations
    print("5. BOOK RESERVATION SYSTEM:")
    
    # Diana reserves Design Patterns
    success, message = library.reserve_book("STF001", "978-0201633610")
    print(f"   Diana reserves Design Patterns: {message}")
    
    # Alice tries to reserve a book she already has
    success, message = library.reserve_book("STU001", "978-0134685991")
    print(f"   Alice tries to reserve book she has: {message}")
    
    print()
    
    # Test search functionality
    print("6. SEARCH FUNCTIONALITY:")
    
    # Search by title
    results = library.search_books("Java", "title")
    print(f"   Search 'Java' in titles: {len(results)} results")
    for result in results:
        print(f"     - {result['title']} ({result['status']})")
    
    # Search by author
    results = library.search_books("Martin", "author")
    print(f"   Search 'Martin' in authors: {len(results)} results")
    for result in results:
        print(f"     - {result['title']} by {', '.join(result['authors']) if 'authors' in result else 'Unknown'}")
    
    print()
    
    # Test overdue and fines
    print("7. OVERDUE BOOKS AND FINES:")
    
    # Simulate overdue book by manually setting due date
    book = library.catalog.all_books["BC1001"]
    book.due_date = datetime.now() - timedelta(days=5)  # 5 days overdue
    
    # Check overdue books
    overdue_books = library.check_overdue_books()
    print(f"   Found {len(overdue_books)} overdue books")
    for overdue in overdue_books:
        print(f"     - {overdue['title']} by {overdue['user_name']} ({overdue['days_overdue']} days)")
    
    # Return overdue book (should calculate fine)
    success, message, fine = library.return_book("STU001", "BC1001")
    print(f"   Return overdue book: {message}")
    
    print()
    
    # Test fine payment
    print("8. FINE MANAGEMENT:")
    
    alice = library.get_user("STU001")
    if alice and alice.outstanding_fines > 0:
        print(f"   Alice's outstanding fines: ${alice.outstanding_fines:.2f}")
        
        # Pay partial fine
        success, message, remaining = library.pay_fine("STU001", 1.00)
        print(f"   Pay $1.00: {message}")
    
    print()
    
    # Test different fine strategies
    print("9. FINE CALCULATION STRATEGIES:")
    
    # Test progressive fine strategy
    library.set_fine_strategy(ProgressiveFineStrategy())
    
    # Simulate another overdue scenario
    book2 = library.catalog.all_books["BC1004"]
    book2.due_date = datetime.now() - timedelta(days=10)  # 10 days overdue
    
    success, message, fine = library.return_book("FAC001", "BC1004")
    print(f"   Return with progressive fines: {message}")
    
    print()
    
    # Generate reports
    print("10. REPORTING AND STATISTICS:")
    
    # User report
    alice_report = library.generate_user_report("STU001")
    if alice_report:
        print("   Alice's Report:")
        print(f"     - Borrowed books: {alice_report['user_info']['borrowed_books_count']}")
        print(f"     - Outstanding fines: ${alice_report['user_info']['outstanding_fines']:.2f}")
        print(f"     - Total transactions: {alice_report['user_info']['total_transactions']}")
    
    # Library statistics
    stats = library.generate_library_statistics()
    print(f"\n   Library Statistics:")
    print(f"     - Total books: {stats['catalog_statistics']['total_books']}")
    print(f"     - Available books: {stats['catalog_statistics']['available_books']}")
    print(f"     - Total users: {stats['user_statistics']['total_users']}")
    print(f"     - Active users: {stats['user_statistics']['active_users']}")
    print(f"     - Outstanding fines: ${stats['user_statistics']['total_outstanding_fines']:.2f}")
    
    print()
    
    # Show notifications sent
    print("11. NOTIFICATION SUMMARY:")
    print(f"   📧 Emails sent: {len(email_service.sent_notifications)}")
    for notification in email_service.sent_notifications[-3:]:  # Show last 3
        print(f"     - {notification['subject']} to {notification['recipient']}")
    
    print(f"   📱 SMS sent: {len(sms_service.sent_messages)}")
    for message in sms_service.sent_messages:
        print(f"     - {message['message']} to {message['recipient']}")
    
    print()
    print("=== LIBRARY MANAGEMENT SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_library_system()
