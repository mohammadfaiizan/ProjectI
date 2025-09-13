"""
STATIC VS INSTANCE MEMBERS - Class vs Instance Variables/Methods
===============================================================

Problem Statement:
Demonstrate the differences between static and instance members:
- Instance variables vs class variables
- Instance methods vs class methods vs static methods
- When to use each type of member
- Memory implications and sharing behavior
- Design patterns using static members

Learning Objectives:
- Understand static vs instance member differences
- Choose appropriate member types for different scenarios
- Implement class-level functionality effectively
- Use static methods and class methods appropriately
- Design efficient memory usage patterns
"""

from typing import List, Dict, Any, Optional, ClassVar
from datetime import datetime
from functools import wraps
import threading
import time


class BankAccount:
    """
    Bank Account class demonstrating static vs instance members.
    """
    
    # Class variables (static members) - shared by all instances
    bank_name: ClassVar[str] = "Global Bank"
    interest_rate: ClassVar[float] = 0.02  # 2% annual interest
    total_accounts: ClassVar[int] = 0
    total_balance: ClassVar[float] = 0.0
    account_types: ClassVar[List[str]] = ["SAVINGS", "CHECKING", "BUSINESS"]
    
    # Class-level data structures
    all_accounts: ClassVar[Dict[str, 'BankAccount']] = {}
    transaction_log: ClassVar[List[Dict[str, Any]]] = []
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float = 0.0):
        """
        Initialize instance with instance variables.
        """
        # Instance variables (unique to each object)
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = initial_balance
        self.account_type = "SAVINGS"
        self.created_at = datetime.now()
        self.transaction_history: List[Dict[str, Any]] = []
        self.is_active = True
        
        # Update class variables
        BankAccount.total_accounts += 1
        BankAccount.total_balance += initial_balance
        BankAccount.all_accounts[account_number] = self
        
        # Log account creation
        self._log_transaction("ACCOUNT_CREATED", initial_balance, "Account opened")
    
    # Instance method - operates on specific instance
    def deposit(self, amount: float) -> bool:
        """Instance method to deposit money."""
        if amount <= 0:
            return False
        
        self.balance += amount
        BankAccount.total_balance += amount  # Update class variable
        
        self._log_transaction("DEPOSIT", amount, f"Deposited ${amount:.2f}")
        return True
    
    # Instance method
    def withdraw(self, amount: float) -> bool:
        """Instance method to withdraw money."""
        if amount <= 0 or amount > self.balance:
            return False
        
        self.balance -= amount
        BankAccount.total_balance -= amount  # Update class variable
        
        self._log_transaction("WITHDRAWAL", -amount, f"Withdrew ${amount:.2f}")
        return True
    
    # Instance method
    def calculate_interest(self) -> float:
        """Calculate interest for this account using class variable."""
        return self.balance * BankAccount.interest_rate
    
    # Instance method
    def apply_interest(self) -> float:
        """Apply interest to this account."""
        interest = self.calculate_interest()
        self.deposit(interest)
        return interest
    
    # Private instance method
    def _log_transaction(self, transaction_type: str, amount: float, description: str) -> None:
        """Log transaction to both instance and class logs."""
        transaction = {
            'account_number': self.account_number,
            'timestamp': datetime.now().isoformat(),
            'type': transaction_type,
            'amount': amount,
            'description': description,
            'balance_after': self.balance
        }
        
        # Add to instance transaction history
        self.transaction_history.append(transaction)
        
        # Add to class-level transaction log
        BankAccount.transaction_log.append(transaction)
    
    # Class method - operates on the class itself
    @classmethod
    def get_bank_info(cls) -> Dict[str, Any]:
        """Class method to get bank-wide information."""
        return {
            'bank_name': cls.bank_name,
            'total_accounts': cls.total_accounts,
            'total_balance': cls.total_balance,
            'interest_rate': cls.interest_rate,
            'account_types': cls.account_types.copy()
        }
    
    # Class method
    @classmethod
    def set_interest_rate(cls, new_rate: float) -> None:
        """Class method to change interest rate for all accounts."""
        if 0 <= new_rate <= 0.1:  # Max 10% interest
            cls.interest_rate = new_rate
            print(f"Interest rate updated to {new_rate:.2%}")
        else:
            print("Invalid interest rate. Must be between 0% and 10%")
    
    # Class method
    @classmethod
    def find_account(cls, account_number: str) -> Optional['BankAccount']:
        """Class method to find account by number."""
        return cls.all_accounts.get(account_number)
    
    # Class method
    @classmethod
    def get_accounts_by_holder(cls, holder_name: str) -> List['BankAccount']:
        """Class method to find accounts by holder name."""
        return [account for account in cls.all_accounts.values() 
                if account.account_holder == holder_name]
    
    # Class method for alternative constructor
    @classmethod
    def create_business_account(cls, account_number: str, business_name: str, 
                              initial_balance: float = 1000.0) -> 'BankAccount':
        """Class method as alternative constructor for business accounts."""
        account = cls(account_number, business_name, initial_balance)
        account.account_type = "BUSINESS"
        return account
    
    # Static method - doesn't access instance or class
    @staticmethod
    def validate_account_number(account_number: str) -> bool:
        """Static method to validate account number format."""
        return (len(account_number) == 10 and 
                account_number.isdigit() and 
                account_number.startswith('1'))
    
    # Static method
    @staticmethod
    def calculate_compound_interest(principal: float, rate: float, 
                                  time_years: int, compounds_per_year: int = 12) -> float:
        """Static method to calculate compound interest."""
        return principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time_years)
    
    # Static method
    @staticmethod
    def format_currency(amount: float) -> str:
        """Static method to format currency."""
        return f"${amount:,.2f}"
    
    # Static method for validation
    @staticmethod
    def is_valid_transaction_amount(amount: float) -> bool:
        """Static method to validate transaction amounts."""
        return 0 < amount <= 1000000  # Max $1M per transaction
    
    # Instance method using static method
    def get_formatted_balance(self) -> str:
        """Instance method using static method."""
        return BankAccount.format_currency(self.balance)
    
    def __str__(self) -> str:
        return f"Account({self.account_number}, {self.account_holder}, {self.get_formatted_balance()})"


class Counter:
    """
    Counter class demonstrating class variables for shared state.
    """
    
    # Class variable - shared counter
    count: ClassVar[int] = 0
    instances: ClassVar[List['Counter']] = []
    
    def __init__(self, name: str):
        self.name = name  # Instance variable
        self.local_count = 0  # Instance variable
        
        # Update class variables
        Counter.count += 1
        Counter.instances.append(self)
    
    def increment(self) -> None:
        """Increment both class and instance counters."""
        Counter.count += 1  # Class variable
        self.local_count += 1  # Instance variable
    
    def decrement(self) -> None:
        """Decrement both counters."""
        Counter.count -= 1
        self.local_count -= 1
    
    @classmethod
    def get_total_count(cls) -> int:
        """Get total count across all instances."""
        return cls.count
    
    @classmethod
    def reset_global_count(cls) -> None:
        """Reset global counter."""
        cls.count = 0
    
    @classmethod
    def get_instance_count(cls) -> int:
        """Get number of Counter instances created."""
        return len(cls.instances)
    
    @staticmethod
    def is_even(number: int) -> bool:
        """Static utility method."""
        return number % 2 == 0
    
    def __str__(self) -> str:
        return f"Counter({self.name}: local={self.local_count}, global={Counter.count})"


class DatabaseConnection:
    """
    Database connection class demonstrating singleton pattern with class variables.
    """
    
    # Class variables for singleton pattern
    _instance: ClassVar[Optional['DatabaseConnection']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    connection_count: ClassVar[int] = 0
    
    def __new__(cls, *args, **kwargs):
        """Singleton implementation using class variables."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, connection_string: str = "default"):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.connection_string = connection_string
            self.is_connected = False
            self.initialized = True
            DatabaseConnection.connection_count += 1
    
    def connect(self) -> bool:
        """Connect to database."""
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        self.is_connected = False
        return True
    
    @classmethod
    def get_instance(cls) -> 'DatabaseConnection':
        """Class method to get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton for testing purposes."""
        cls._instance = None


class MathUtils:
    """
    Utility class with only static methods (no instance needed).
    """
    
    # Class variable for constants
    PI: ClassVar[float] = 3.14159265359
    E: ClassVar[float] = 2.71828182846
    
    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    @staticmethod
    def power(base: float, exponent: float) -> float:
        """Calculate power."""
        return base ** exponent
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    @classmethod
    def circle_area(cls, radius: float) -> float:
        """Calculate circle area using class constant."""
        return cls.PI * radius ** 2
    
    @classmethod
    def circle_circumference(cls, radius: float) -> float:
        """Calculate circle circumference using class constant."""
        return 2 * cls.PI * radius


# Decorator using static methods
class Timer:
    """Timer utility class with static methods."""
    
    @staticmethod
    def time_function(func):
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def current_timestamp() -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()


def demonstrate_static_vs_instance_members():
    """
    Demonstrate static vs instance members with practical examples.
    """
    print("=== STATIC VS INSTANCE MEMBERS DEMONSTRATION ===\n")
    
    # 1. Bank Account Example
    print("1. Bank Account - Instance vs Class Variables:")
    
    # Create accounts
    account1 = BankAccount("1234567890", "Alice Johnson", 1000.0)
    account2 = BankAccount("1234567891", "Bob Smith", 500.0)
    account3 = BankAccount.create_business_account("1234567892", "Tech Corp", 5000.0)
    
    print(f"Created 3 accounts:")
    print(f"  {account1}")
    print(f"  {account2}")
    print(f"  {account3}")
    
    # Show class-level information
    bank_info = BankAccount.get_bank_info()
    print(f"\nBank Info: {bank_info}")
    
    # Perform transactions
    account1.deposit(200)
    account2.withdraw(100)
    account3.deposit(1000)
    
    # Show updated class-level information
    updated_info = BankAccount.get_bank_info()
    print(f"Updated Bank Info: {updated_info}")
    
    # Use class method to change interest rate
    BankAccount.set_interest_rate(0.03)  # 3%
    
    # Apply interest to all accounts
    for account in BankAccount.all_accounts.values():
        interest = account.apply_interest()
        print(f"Applied ${interest:.2f} interest to {account.account_holder}")
    
    print()
    
    # 2. Static Methods Usage
    print("2. Static Methods Usage:")
    
    # Validate account numbers
    test_numbers = ["1234567890", "0123456789", "12345", "abcd567890"]
    for number in test_numbers:
        is_valid = BankAccount.validate_account_number(number)
        print(f"Account number '{number}' is {'valid' if is_valid else 'invalid'}")
    
    # Use static utility methods
    principal = 1000.0
    rate = 0.05
    years = 5
    compound_amount = BankAccount.calculate_compound_interest(principal, rate, years)
    print(f"${principal} at {rate:.1%} for {years} years = {BankAccount.format_currency(compound_amount)}")
    
    # Validate transaction amounts
    amounts = [100.0, -50.0, 1500000.0, 0.0]
    for amount in amounts:
        is_valid = BankAccount.is_valid_transaction_amount(amount)
        print(f"Transaction amount ${amount} is {'valid' if is_valid else 'invalid'}")
    
    print()
    
    # 3. Counter Example - Shared State
    print("3. Counter Example - Shared vs Instance State:")
    
    counter1 = Counter("Counter1")
    counter2 = Counter("Counter2")
    counter3 = Counter("Counter3")
    
    print(f"Initial state:")
    print(f"  {counter1}")
    print(f"  {counter2}")
    print(f"  {counter3}")
    print(f"  Total instances: {Counter.get_instance_count()}")
    
    # Increment counters
    counter1.increment()
    counter1.increment()
    counter2.increment()
    counter3.increment()
    counter3.increment()
    counter3.increment()
    
    print(f"\nAfter increments:")
    print(f"  {counter1}")
    print(f"  {counter2}")
    print(f"  {counter3}")
    print(f"  Global count: {Counter.get_total_count()}")
    
    # Test static method
    print(f"  Is global count even? {Counter.is_even(Counter.get_total_count())}")
    
    print()
    
    # 4. Singleton Pattern with Class Variables
    print("4. Singleton Pattern with Class Variables:")
    
    # Try to create multiple database connections
    db1 = DatabaseConnection("connection1")
    db2 = DatabaseConnection("connection2")
    db3 = DatabaseConnection.get_instance()
    
    print(f"db1 is db2: {db1 is db2}")
    print(f"db2 is db3: {db2 is db3}")
    print(f"All are same instance: {db1 is db2 is db3}")
    print(f"Connection count: {DatabaseConnection.connection_count}")
    
    # Connect and check state
    db1.connect()
    print(f"db2 connected: {db2.is_connected}")  # Should be True
    print()
    
    # 5. Utility Class with Static Methods
    print("5. Utility Class with Static Methods:")
    
    # Use static methods without creating instance
    print(f"MathUtils.add(5, 3): {MathUtils.add(5, 3)}")
    print(f"MathUtils.multiply(4, 7): {MathUtils.multiply(4, 7)}")
    print(f"MathUtils.power(2, 8): {MathUtils.power(2, 8)}")
    print(f"MathUtils.factorial(5): {MathUtils.factorial(5)}")
    print(f"MathUtils.is_prime(17): {MathUtils.is_prime(17)}")
    print(f"MathUtils.gcd(48, 18): {MathUtils.gcd(48, 18)}")
    
    # Use class methods with class constants
    radius = 5.0
    print(f"Circle area (r={radius}): {MathUtils.circle_area(radius):.2f}")
    print(f"Circle circumference (r={radius}): {MathUtils.circle_circumference(radius):.2f}")
    print()
    
    # 6. Memory Usage Comparison
    print("6. Memory Usage and Sharing:")
    
    # Show that class variables are shared
    print("Class variables are shared:")
    print(f"  BankAccount.bank_name: {BankAccount.bank_name}")
    print(f"  account1.bank_name: {account1.bank_name}")
    print(f"  account2.bank_name: {account2.bank_name}")
    
    # Change class variable
    BankAccount.bank_name = "New Global Bank"
    print(f"After changing class variable:")
    print(f"  BankAccount.bank_name: {BankAccount.bank_name}")
    print(f"  account1.bank_name: {account1.bank_name}")
    print(f"  account2.bank_name: {account2.bank_name}")
    
    # Instance variables are separate
    print(f"\nInstance variables are separate:")
    print(f"  account1.balance: {account1.balance}")
    print(f"  account2.balance: {account2.balance}")
    
    account1.balance = 9999.99
    print(f"After changing account1.balance:")
    print(f"  account1.balance: {account1.balance}")
    print(f"  account2.balance: {account2.balance}")  # Unchanged
    print()
    
    # 7. Using Timer Decorator (Static Method)
    print("7. Static Method as Decorator:")
    
    @Timer.time_function
    def slow_function():
        """Function that takes some time."""
        time.sleep(0.1)
        return "Done"
    
    result = slow_function()
    print(f"Function result: {result}")
    print(f"Current timestamp: {Timer.current_timestamp()}")
    print()
    
    print("=== STATIC VS INSTANCE MEMBERS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_static_vs_instance_members()
