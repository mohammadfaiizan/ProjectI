"""
SINGLE RESPONSIBILITY PRINCIPLE - SRP with Practical Examples
=============================================================

Problem Statement:
Demonstrate the Single Responsibility Principle (SRP):
- A class should have only one reason to change
- Each class should have only one job or responsibility
- Separation of concerns in class design
- Identifying and fixing SRP violations
- Benefits of following SRP

Learning Objectives:
- Understand what constitutes a single responsibility
- Identify SRP violations in code
- Refactor code to follow SRP
- Design classes with clear, single purposes
- Apply SRP in real-world scenarios
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib


# VIOLATION EXAMPLE - Class with multiple responsibilities
class BadUserManager:
    """
    BAD EXAMPLE: This class violates SRP by having multiple responsibilities:
    1. User data management
    2. User validation
    3. Password hashing
    4. Email sending
    5. Database operations
    6. Logging
    """
    
    def __init__(self):
        self.users = {}
        self.log_entries = []
    
    def create_user(self, username: str, email: str, password: str) -> bool:
        """Creates user - but does too many things!"""
        
        # Responsibility 1: Logging
        self.log_entries.append(f"Creating user: {username}")
        
        # Responsibility 2: Validation
        if not self.validate_email(email):
            self.log_entries.append(f"Invalid email: {email}")
            return False
        
        if not self.validate_password(password):
            self.log_entries.append(f"Invalid password for user: {username}")
            return False
        
        # Responsibility 3: Password hashing
        hashed_password = self.hash_password(password)
        
        # Responsibility 4: User data management
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.now().isoformat()
        }
        
        # Responsibility 5: Database operations
        self.save_to_database(user_data)
        
        # Responsibility 6: Email sending
        self.send_welcome_email(email, username)
        
        self.users[username] = user_data
        return True
    
    def validate_email(self, email: str) -> bool:
        """Email validation logic"""
        return "@" in email and "." in email.split("@")[1]
    
    def validate_password(self, password: str) -> bool:
        """Password validation logic"""
        return len(password) >= 8
    
    def hash_password(self, password: str) -> str:
        """Password hashing logic"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def save_to_database(self, user_data: dict) -> None:
        """Database save logic"""
        print(f"Saving to database: {user_data['username']}")
    
    def send_welcome_email(self, email: str, username: str) -> None:
        """Email sending logic"""
        print(f"Sending welcome email to {email} for user {username}")


# GOOD EXAMPLE - Following SRP by separating responsibilities

# 1. Single Responsibility: Email Validation
class EmailValidator:
    """Responsible ONLY for email validation."""
    
    @staticmethod
    def is_valid(email: str) -> bool:
        """Validate email format."""
        if not email or "@" not in email:
            return False
        
        parts = email.split("@")
        if len(parts) != 2:
            return False
        
        local, domain = parts
        return len(local) > 0 and "." in domain and len(domain.split(".")[1]) > 0
    
    @staticmethod
    def normalize(email: str) -> str:
        """Normalize email to lowercase."""
        return email.lower().strip()


# 2. Single Responsibility: Password Operations
class PasswordManager:
    """Responsible ONLY for password-related operations."""
    
    @staticmethod
    def is_strong(password: str) -> bool:
        """Check if password meets strength requirements."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    @staticmethod
    def hash_password(password: str, salt: str = "default_salt") -> str:
        """Hash password with salt."""
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str = "default_salt") -> bool:
        """Verify password against hash."""
        return PasswordManager.hash_password(password, salt) == hashed


# 3. Single Responsibility: User Data Model
class User:
    """Responsible ONLY for representing user data."""
    
    def __init__(self, user_id: str, username: str, email: str, password_hash: str):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = datetime.now()
        self.is_active = True
        self.last_login = None
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
    
    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __str__(self) -> str:
        return f"User(id={self.user_id}, username={self.username}, email={self.email})"


# 4. Single Responsibility: User Repository (Data Access)
class UserRepository:
    """Responsible ONLY for user data persistence."""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._next_id = 1
    
    def save(self, user: User) -> bool:
        """Save user to storage."""
        try:
            self._users[user.user_id] = user
            print(f"User {user.username} saved to repository")
            return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID."""
        return self._users.get(user_id)
    
    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None
    
    def get_all_users(self) -> List[User]:
        """Get all users."""
        return list(self._users.values())
    
    def delete(self, user_id: str) -> bool:
        """Delete user from storage."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
    
    def generate_user_id(self) -> str:
        """Generate unique user ID."""
        user_id = f"USER_{self._next_id:06d}"
        self._next_id += 1
        return user_id


# 5. Single Responsibility: Email Service
class EmailService:
    """Responsible ONLY for sending emails."""
    
    def __init__(self, smtp_server: str = "localhost", port: int = 587):
        self.smtp_server = smtp_server
        self.port = port
        self.sent_emails = []  # For demonstration
    
    def send_welcome_email(self, email: str, username: str) -> bool:
        """Send welcome email to new user."""
        subject = "Welcome to Our Platform!"
        body = f"""
        Dear {username},
        
        Welcome to our platform! Your account has been successfully created.
        
        Best regards,
        The Team
        """
        
        return self.send_email(email, subject, body)
    
    def send_password_reset_email(self, email: str, reset_token: str) -> bool:
        """Send password reset email."""
        subject = "Password Reset Request"
        body = f"""
        A password reset was requested for your account.
        
        Reset token: {reset_token}
        
        If you didn't request this, please ignore this email.
        """
        
        return self.send_email(email, subject, body)
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email (simulated)."""
        try:
            email_data = {
                'to': to,
                'subject': subject,
                'body': body,
                'sent_at': datetime.now().isoformat()
            }
            
            self.sent_emails.append(email_data)
            print(f"Email sent to {to}: {subject}")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False


# 6. Single Responsibility: Logging
class Logger:
    """Responsible ONLY for logging operations."""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.log_entries = []
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._log("INFO", message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log("WARNING", message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self._log("ERROR", message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._log("DEBUG", message)
    
    def _log(self, level: str, message: str) -> None:
        """Internal logging method."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(log_entry)
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.log_entries.copy()


# 7. Single Responsibility: User Service (Business Logic)
class UserService:
    """
    Responsible ONLY for user-related business logic.
    Coordinates other services but doesn't implement their functionality.
    """
    
    def __init__(self, user_repository: UserRepository, email_service: EmailService, logger: Logger):
        self.user_repository = user_repository
        self.email_service = email_service
        self.logger = logger
    
    def create_user(self, username: str, email: str, password: str) -> Optional[User]:
        """Create new user with proper validation and coordination."""
        
        self.logger.info(f"Attempting to create user: {username}")
        
        # Validate email
        if not EmailValidator.is_valid(email):
            self.logger.error(f"Invalid email format: {email}")
            return None
        
        # Normalize email
        email = EmailValidator.normalize(email)
        
        # Check if user already exists
        if self.user_repository.find_by_username(username):
            self.logger.error(f"Username already exists: {username}")
            return None
        
        if self.user_repository.find_by_email(email):
            self.logger.error(f"Email already registered: {email}")
            return None
        
        # Validate password
        if not PasswordManager.is_strong(password):
            self.logger.error(f"Password too weak for user: {username}")
            return None
        
        # Hash password
        password_hash = PasswordManager.hash_password(password)
        
        # Create user
        user_id = self.user_repository.generate_user_id()
        user = User(user_id, username, email, password_hash)
        
        # Save user
        if self.user_repository.save(user):
            self.logger.info(f"User created successfully: {username}")
            
            # Send welcome email
            if self.email_service.send_welcome_email(email, username):
                self.logger.info(f"Welcome email sent to: {email}")
            else:
                self.logger.warning(f"Failed to send welcome email to: {email}")
            
            return user
        else:
            self.logger.error(f"Failed to save user: {username}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user login."""
        
        self.logger.info(f"Authentication attempt for user: {username}")
        
        user = self.user_repository.find_by_username(username)
        if not user:
            self.logger.warning(f"User not found: {username}")
            return None
        
        if not user.is_active:
            self.logger.warning(f"Inactive user login attempt: {username}")
            return None
        
        if PasswordManager.verify_password(password, user.password_hash):
            user.update_last_login()
            self.user_repository.save(user)  # Update last login
            self.logger.info(f"User authenticated successfully: {username}")
            return user
        else:
            self.logger.warning(f"Invalid password for user: {username}")
            return None
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        
        user = self.user_repository.find_by_id(user_id)
        if not user:
            self.logger.error(f"User not found for deactivation: {user_id}")
            return False
        
        user.deactivate()
        if self.user_repository.save(user):
            self.logger.info(f"User deactivated: {user.username}")
            return True
        else:
            self.logger.error(f"Failed to deactivate user: {user.username}")
            return False
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        all_users = self.user_repository.get_all_users()
        active_users = [u for u in all_users if u.is_active]
        
        return {
            'total_users': len(all_users),
            'active_users': len(active_users),
            'inactive_users': len(all_users) - len(active_users)
        }


# Example of another domain following SRP
class Product:
    """Single responsibility: Product data representation."""
    
    def __init__(self, product_id: str, name: str, price: float, category: str):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.created_at = datetime.now()
        self.is_available = True
    
    def update_price(self, new_price: float) -> None:
        """Update product price."""
        if new_price > 0:
            self.price = new_price
    
    def set_availability(self, available: bool) -> None:
        """Set product availability."""
        self.is_available = available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary."""
        return {
            'product_id': self.product_id,
            'name': self.name,
            'price': self.price,
            'category': self.category,
            'is_available': self.is_available,
            'created_at': self.created_at.isoformat()
        }


class PriceCalculator:
    """Single responsibility: Price calculations."""
    
    @staticmethod
    def calculate_discount(original_price: float, discount_percentage: float) -> float:
        """Calculate discounted price."""
        if discount_percentage < 0 or discount_percentage > 100:
            raise ValueError("Discount percentage must be between 0 and 100")
        
        return original_price * (1 - discount_percentage / 100)
    
    @staticmethod
    def calculate_tax(price: float, tax_rate: float) -> float:
        """Calculate tax amount."""
        return price * tax_rate
    
    @staticmethod
    def calculate_total_with_tax(price: float, tax_rate: float) -> float:
        """Calculate total price including tax."""
        return price + PriceCalculator.calculate_tax(price, tax_rate)


class InventoryManager:
    """Single responsibility: Inventory management."""
    
    def __init__(self):
        self.inventory: Dict[str, int] = {}
    
    def add_stock(self, product_id: str, quantity: int) -> None:
        """Add stock for product."""
        if product_id in self.inventory:
            self.inventory[product_id] += quantity
        else:
            self.inventory[product_id] = quantity
    
    def remove_stock(self, product_id: str, quantity: int) -> bool:
        """Remove stock for product."""
        if product_id not in self.inventory:
            return False
        
        if self.inventory[product_id] < quantity:
            return False
        
        self.inventory[product_id] -= quantity
        return True
    
    def get_stock_level(self, product_id: str) -> int:
        """Get current stock level."""
        return self.inventory.get(product_id, 0)
    
    def is_in_stock(self, product_id: str, required_quantity: int = 1) -> bool:
        """Check if product is in stock."""
        return self.get_stock_level(product_id) >= required_quantity


def demonstrate_single_responsibility_principle():
    """
    Demonstrate Single Responsibility Principle with practical examples.
    """
    print("=== SINGLE RESPONSIBILITY PRINCIPLE DEMONSTRATION ===\n")
    
    # 1. Show the problem with violating SRP
    print("1. SRP VIOLATION EXAMPLE:")
    print("   BadUserManager class has multiple responsibilities:")
    print("   - User data management")
    print("   - Validation")
    print("   - Password hashing")
    print("   - Email sending")
    print("   - Database operations")
    print("   - Logging")
    print("   This makes it hard to maintain, test, and extend.\n")
    
    # 2. Demonstrate SRP-compliant design
    print("2. SRP-COMPLIANT DESIGN:")
    
    # Create services with single responsibilities
    logger = Logger()
    user_repository = UserRepository()
    email_service = EmailService()
    user_service = UserService(user_repository, email_service, logger)
    
    print("   Created separate classes for:")
    print("   - Logger: Only handles logging")
    print("   - UserRepository: Only handles data persistence")
    print("   - EmailService: Only handles email sending")
    print("   - UserService: Only handles business logic coordination")
    print()
    
    # 3. Create users using SRP-compliant design
    print("3. CREATING USERS WITH SRP DESIGN:")
    
    # Valid user creation
    user1 = user_service.create_user("alice_johnson", "alice@example.com", "SecurePass123!")
    if user1:
        print(f"   Successfully created: {user1}")
    
    # Invalid email
    user2 = user_service.create_user("bob_smith", "invalid-email", "SecurePass123!")
    if not user2:
        print("   Failed to create user with invalid email (as expected)")
    
    # Weak password
    user3 = user_service.create_user("charlie_brown", "charlie@example.com", "weak")
    if not user3:
        print("   Failed to create user with weak password (as expected)")
    
    # Duplicate username
    user4 = user_service.create_user("alice_johnson", "alice2@example.com", "SecurePass123!")
    if not user4:
        print("   Failed to create user with duplicate username (as expected)")
    
    print()
    
    # 4. User authentication
    print("4. USER AUTHENTICATION:")
    
    authenticated_user = user_service.authenticate_user("alice_johnson", "SecurePass123!")
    if authenticated_user:
        print(f"   Authentication successful: {authenticated_user.username}")
        print(f"   Last login: {authenticated_user.last_login}")
    
    failed_auth = user_service.authenticate_user("alice_johnson", "wrong_password")
    if not failed_auth:
        print("   Authentication failed with wrong password (as expected)")
    
    print()
    
    # 5. Show individual component functionality
    print("5. INDIVIDUAL COMPONENT FUNCTIONALITY:")
    
    # Email validation
    emails = ["valid@example.com", "invalid-email", "another@valid.org"]
    print("   Email Validation:")
    for email in emails:
        is_valid = EmailValidator.is_valid(email)
        normalized = EmailValidator.normalize(email) if is_valid else "N/A"
        print(f"     {email}: {'Valid' if is_valid else 'Invalid'} -> {normalized}")
    
    # Password strength checking
    passwords = ["weak", "StrongPass123!", "NoNumbers!", "nonumbers123"]
    print("\n   Password Strength:")
    for password in passwords:
        is_strong = PasswordManager.is_strong(password)
        print(f"     '{password}': {'Strong' if is_strong else 'Weak'}")
    
    print()
    
    # 6. Product domain example
    print("6. PRODUCT DOMAIN EXAMPLE (Also Following SRP):")
    
    # Create product
    product = Product("PROD001", "Laptop", 999.99, "Electronics")
    print(f"   Created product: {product.name} - ${product.price}")
    
    # Price calculations
    discounted_price = PriceCalculator.calculate_discount(product.price, 10)  # 10% off
    tax_amount = PriceCalculator.calculate_tax(discounted_price, 0.08)  # 8% tax
    total_price = PriceCalculator.calculate_total_with_tax(discounted_price, 0.08)
    
    print(f"   Original price: ${product.price:.2f}")
    print(f"   After 10% discount: ${discounted_price:.2f}")
    print(f"   Tax (8%): ${tax_amount:.2f}")
    print(f"   Total with tax: ${total_price:.2f}")
    
    # Inventory management
    inventory = InventoryManager()
    inventory.add_stock("PROD001", 50)
    print(f"   Added 50 units to inventory")
    print(f"   Current stock: {inventory.get_stock_level('PROD001')}")
    print(f"   In stock (10 units): {inventory.is_in_stock('PROD001', 10)}")
    
    inventory.remove_stock("PROD001", 5)
    print(f"   After removing 5 units: {inventory.get_stock_level('PROD001')}")
    
    print()
    
    # 7. Show user statistics
    print("7. USER STATISTICS:")
    stats = user_service.get_user_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print()
    
    # 8. Show logs
    print("8. SYSTEM LOGS:")
    logs = logger.get_logs()
    for log in logs[-5:]:  # Show last 5 logs
        print(f"   {log}")
    
    print()
    
    # 9. SRP Benefits Summary
    print("9. SRP BENEFITS:")
    print("   ✓ Each class has a single, well-defined purpose")
    print("   ✓ Changes to one responsibility don't affect others")
    print("   ✓ Classes are easier to understand and maintain")
    print("   ✓ Better testability - can test each responsibility in isolation")
    print("   ✓ Higher cohesion within classes")
    print("   ✓ Easier to extend and modify individual components")
    print("   ✓ Follows the principle of separation of concerns")
    print()
    
    print("=== SINGLE RESPONSIBILITY PRINCIPLE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_single_responsibility_principle()
