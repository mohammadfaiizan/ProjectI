"""
ACCESS MODIFIERS DESIGN - Public, Private, Protected Design
===========================================================

Problem Statement:
Demonstrate access modifier concepts and design patterns:
- Public, private, and protected members in Python
- Name mangling and conventions
- Property decorators for controlled access
- Access control design patterns
- Information hiding best practices

Learning Objectives:
- Understand Python's access control mechanisms
- Design proper access levels for class members
- Use properties for controlled access
- Implement access control patterns
- Follow Python naming conventions
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class User:
    """
    User class demonstrating access modifiers and controlled access.
    """
    
    def __init__(self, username: str, email: str, password: str):
        # Public attributes (no underscore prefix)
        self.username = username
        self.email = email
        self.created_at = datetime.now()
        self.is_active = True
        
        # Protected attributes (single underscore - convention only)
        self._user_id = self._generate_user_id()
        self._login_attempts = 0
        self._last_login = None
        
        # Private attributes (double underscore - name mangling)
        self.__password_hash = self._hash_password(password)
        self.__salt = "user_salt_2023"
        self.__session_token = None
    
    # Public methods
    def get_user_info(self) -> Dict[str, Any]:
        """Public method to get user information."""
        return {
            'username': self.username,
            'email': self.email,
            'user_id': self._user_id,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'last_login': self._last_login.isoformat() if self._last_login else None
        }
    
    def update_email(self, new_email: str) -> bool:
        """Public method to update email with validation."""
        if self._validate_email(new_email):
            self.email = new_email
            return True
        return False
    
    def login(self, password: str) -> bool:
        """Public method for user login."""
        if self._verify_password(password):
            self._last_login = datetime.now()
            self._login_attempts = 0
            self.__session_token = self._generate_session_token()
            return True
        else:
            self._login_attempts += 1
            if self._login_attempts >= 3:
                self.is_active = False
                print("Account locked due to multiple failed login attempts")
            return False
    
    def logout(self) -> None:
        """Public method for user logout."""
        self.__session_token = None
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """Public method to change password."""
        if self._verify_password(old_password):
            if self._validate_password_strength(new_password):
                self.__password_hash = self._hash_password(new_password)
                self.__session_token = None  # Invalidate session
                return True
            else:
                print("New password doesn't meet strength requirements")
                return False
        else:
            print("Current password is incorrect")
            return False
    
    # Protected methods (intended for subclasses)
    def _generate_user_id(self) -> str:
        """Protected method to generate user ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _validate_email(self, email: str) -> bool:
        """Protected method to validate email format."""
        return "@" in email and "." in email.split("@")[1]
    
    def _validate_password_strength(self, password: str) -> bool:
        """Protected method to validate password strength."""
        return (len(password) >= 8 and
                any(c.isupper() for c in password) and
                any(c.islower() for c in password) and
                any(c.isdigit() for c in password))
    
    def _hash_password(self, password: str) -> str:
        """Protected method to hash password."""
        return hashlib.sha256((password + self.__salt).encode()).hexdigest()
    
    def _verify_password(self, password: str) -> bool:
        """Protected method to verify password."""
        return self._hash_password(password) == self.__password_hash
    
    def _generate_session_token(self) -> str:
        """Protected method to generate session token."""
        import uuid
        return str(uuid.uuid4())
    
    def _reset_login_attempts(self) -> None:
        """Protected method to reset login attempts."""
        self._login_attempts = 0
        self.is_active = True
    
    # Private methods (name mangled)
    def __validate_session(self) -> bool:
        """Private method to validate session."""
        return self.__session_token is not None
    
    def __get_password_hash(self) -> str:
        """Private method to get password hash."""
        return self.__password_hash
    
    # Property decorators for controlled access
    @property
    def user_id(self) -> str:
        """Read-only property for user ID."""
        return self._user_id
    
    @property
    def login_attempts(self) -> int:
        """Read-only property for login attempts."""
        return self._login_attempts
    
    @property
    def is_logged_in(self) -> bool:
        """Read-only property to check login status."""
        return self.__validate_session()
    
    def __str__(self) -> str:
        return f"User({self.username}, {self.email})"


class AdminUser(User):
    """
    Admin user class demonstrating protected member access in inheritance.
    """
    
    def __init__(self, username: str, email: str, password: str, admin_level: int):
        super().__init__(username, email, password)
        self.admin_level = admin_level
        self._permissions = self._get_default_permissions()
    
    def _get_default_permissions(self) -> List[str]:
        """Protected method to get default admin permissions."""
        base_permissions = ["read", "write"]
        if self.admin_level >= 2:
            base_permissions.extend(["delete", "modify_users"])
        if self.admin_level >= 3:
            base_permissions.extend(["system_admin", "backup"])
        return base_permissions
    
    def reset_user_password(self, target_user: User, new_password: str) -> bool:
        """Admin method to reset another user's password."""
        if "modify_users" in self._permissions:
            # Access protected method from parent class
            if self._validate_password_strength(new_password):
                # Cannot access private method directly - would need public interface
                print(f"Admin {self.username} reset password for {target_user.username}")
                return True
            else:
                print("Password doesn't meet strength requirements")
                return False
        else:
            print("Insufficient permissions to reset user password")
            return False
    
    def unlock_user_account(self, target_user: User) -> bool:
        """Admin method to unlock user account."""
        if "modify_users" in self._permissions:
            # Access protected method
            target_user._reset_login_attempts()
            print(f"Admin {self.username} unlocked account for {target_user.username}")
            return True
        else:
            print("Insufficient permissions to unlock user account")
            return False
    
    def get_user_details(self, target_user: User) -> Dict[str, Any]:
        """Admin method to get detailed user information."""
        if "read" in self._permissions:
            details = target_user.get_user_info()
            # Can access protected attributes
            details.update({
                'login_attempts': target_user._login_attempts,
                'user_id': target_user._user_id
            })
            # Cannot access private attributes directly
            # details['password_hash'] = target_user.__password_hash  # This would fail
            return details
        else:
            print("Insufficient permissions to view user details")
            return {}


class BankAccount:
    """
    Bank account class with strict access control.
    """
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float):
        # Public attributes
        self.account_holder = account_holder
        self.account_type = "SAVINGS"
        self.created_at = datetime.now()
        
        # Protected attributes
        self._account_number = account_number
        self._transaction_history = []
        
        # Private attributes (sensitive data)
        self.__balance = initial_balance
        self.__pin = None
        self.__security_questions = {}
    
    # Properties for controlled access
    @property
    def balance(self) -> Optional[float]:
        """Get balance (requires authentication in real implementation)."""
        # In real implementation, this would require PIN verification
        return self.__balance
    
    @property
    def account_number(self) -> str:
        """Get masked account number."""
        return f"****{self._account_number[-4:]}"
    
    @property
    def full_account_number(self) -> str:
        """Get full account number (protected access)."""
        return self._account_number
    
    # Setter with validation
    @balance.setter
    def balance(self, value: float) -> None:
        """Set balance with validation."""
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self.__balance = value
    
    # Public methods
    def deposit(self, amount: float) -> bool:
        """Public method to deposit money."""
        if amount <= 0:
            return False
        
        self.__balance += amount
        self._log_transaction("DEPOSIT", amount)
        return True
    
    def withdraw(self, amount: float, pin: str) -> bool:
        """Public method to withdraw money with PIN verification."""
        if not self._verify_pin(pin):
            print("Invalid PIN")
            return False
        
        if amount <= 0 or amount > self.__balance:
            print("Invalid withdrawal amount")
            return False
        
        self.__balance -= amount
        self._log_transaction("WITHDRAWAL", -amount)
        return True
    
    def set_pin(self, new_pin: str, old_pin: Optional[str] = None) -> bool:
        """Public method to set PIN."""
        if self.__pin is not None and not self._verify_pin(old_pin):
            print("Current PIN verification failed")
            return False
        
        if self._validate_pin(new_pin):
            self.__pin = self._hash_pin(new_pin)
            return True
        else:
            print("Invalid PIN format")
            return False
    
    # Protected methods
    def _log_transaction(self, transaction_type: str, amount: float) -> None:
        """Protected method to log transactions."""
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': transaction_type,
            'amount': amount,
            'balance_after': self.__balance
        }
        self._transaction_history.append(transaction)
    
    def _validate_pin(self, pin: str) -> bool:
        """Protected method to validate PIN format."""
        return len(pin) == 4 and pin.isdigit()
    
    def _hash_pin(self, pin: str) -> str:
        """Protected method to hash PIN."""
        return hashlib.sha256(pin.encode()).hexdigest()
    
    def _verify_pin(self, pin: str) -> bool:
        """Protected method to verify PIN."""
        if self.__pin is None or pin is None:
            return False
        return self._hash_pin(pin) == self.__pin
    
    # Private methods
    def __calculate_interest(self) -> float:
        """Private method to calculate interest."""
        return self.__balance * 0.02  # 2% interest
    
    def __apply_monthly_fee(self) -> None:
        """Private method to apply monthly fee."""
        if self.__balance < 1000:
            self.__balance -= 10  # $10 monthly fee for low balance
    
    def get_statement(self, pin: str) -> Optional[Dict[str, Any]]:
        """Get account statement with PIN verification."""
        if not self._verify_pin(pin):
            print("Invalid PIN for statement access")
            return None
        
        return {
            'account_holder': self.account_holder,
            'account_number': self.account_number,
            'balance': self.__balance,
            'transactions': self._transaction_history[-10:]  # Last 10 transactions
        }


class SecureDocument:
    """
    Document class with multiple access levels.
    """
    
    def __init__(self, title: str, content: str, classification: str = "PUBLIC"):
        # Public attributes
        self.title = title
        self.created_at = datetime.now()
        self.classification = classification
        
        # Protected attributes
        self._document_id = self._generate_document_id()
        self._access_log = []
        
        # Private attributes
        self.__content = content
        self.__encryption_key = self._generate_encryption_key()
        self.__authorized_users = set()
    
    # Properties with different access levels
    @property
    def document_id(self) -> str:
        """Public read-only property."""
        return self._document_id
    
    @property
    def content(self) -> Optional[str]:
        """Controlled access to content based on classification."""
        if self.classification == "PUBLIC":
            self._log_access("PUBLIC_READ")
            return self.__content
        else:
            print("Access denied: Document is classified")
            return None
    
    def get_content_with_authorization(self, user_id: str) -> Optional[str]:
        """Get content with user authorization."""
        if user_id in self.__authorized_users or self.classification == "PUBLIC":
            self._log_access(f"AUTHORIZED_READ_{user_id}")
            return self.__content
        else:
            self._log_access(f"UNAUTHORIZED_ACCESS_ATTEMPT_{user_id}")
            print("Access denied: User not authorized")
            return None
    
    def authorize_user(self, user_id: str, admin_key: str) -> bool:
        """Authorize user to access classified document."""
        if self._verify_admin_key(admin_key):
            self.__authorized_users.add(user_id)
            self._log_access(f"USER_AUTHORIZED_{user_id}")
            return True
        else:
            print("Invalid admin key")
            return False
    
    def revoke_user_access(self, user_id: str, admin_key: str) -> bool:
        """Revoke user access to document."""
        if self._verify_admin_key(admin_key):
            self.__authorized_users.discard(user_id)
            self._log_access(f"USER_ACCESS_REVOKED_{user_id}")
            return True
        else:
            print("Invalid admin key")
            return False
    
    # Protected methods
    def _generate_document_id(self) -> str:
        """Protected method to generate document ID."""
        import uuid
        return f"DOC_{str(uuid.uuid4())[:8]}"
    
    def _generate_encryption_key(self) -> str:
        """Protected method to generate encryption key."""
        import uuid
        return str(uuid.uuid4())
    
    def _log_access(self, action: str) -> None:
        """Protected method to log access attempts."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'document_id': self._document_id
        }
        self._access_log.append(log_entry)
    
    def _verify_admin_key(self, admin_key: str) -> bool:
        """Protected method to verify admin key."""
        return admin_key == "admin_master_key_2023"  # Simplified for demo
    
    # Private methods
    def __encrypt_content(self) -> str:
        """Private method to encrypt content."""
        # Simplified encryption for demo
        return "ENCRYPTED_" + self.__content
    
    def __decrypt_content(self, encrypted_content: str) -> str:
        """Private method to decrypt content."""
        return encrypted_content.replace("ENCRYPTED_", "")
    
    def get_access_log(self, admin_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get access log (admin only)."""
        if self._verify_admin_key(admin_key):
            return self._access_log.copy()
        else:
            print("Access denied: Invalid admin key")
            return None


def demonstrate_access_modifiers():
    """
    Demonstrate access modifiers and controlled access patterns.
    """
    print("=== ACCESS MODIFIERS DESIGN DEMONSTRATION ===\n")
    
    # 1. Basic Access Control
    print("1. Basic Access Control with User Class:")
    
    user = User("john_doe", "john@example.com", "SecurePass123")
    print(f"User created: {user}")
    print(f"User info: {user.get_user_info()}")
    
    # Access public attributes
    print(f"Public - Username: {user.username}")
    print(f"Public - Email: {user.email}")
    
    # Access protected attributes (by convention)
    print(f"Protected - User ID: {user._user_id}")
    print(f"Protected - Login attempts: {user._login_attempts}")
    
    # Try to access private attributes (will show name mangling)
    try:
        print(f"Private - Password hash: {user.__password_hash}")
    except AttributeError as e:
        print(f"Cannot access private attribute: {e}")
    
    # Access private attribute through name mangling (not recommended)
    print(f"Name mangled access: {user._User__password_hash[:20]}...")
    print()
    
    # 2. Property-based Access Control
    print("2. Property-based Access Control:")
    
    print(f"User ID (property): {user.user_id}")
    print(f"Login attempts (property): {user.login_attempts}")
    print(f"Is logged in (property): {user.is_logged_in}")
    
    # Login and check status
    login_success = user.login("SecurePass123")
    print(f"Login successful: {login_success}")
    print(f"Is logged in after login: {user.is_logged_in}")
    
    # Try wrong password
    user.logout()
    wrong_login = user.login("WrongPassword")
    print(f"Login with wrong password: {wrong_login}")
    print(f"Login attempts after failure: {user.login_attempts}")
    print()
    
    # 3. Inheritance and Protected Access
    print("3. Inheritance and Protected Access:")
    
    admin = AdminUser("admin_user", "admin@example.com", "AdminPass123", 3)
    regular_user = User("regular_user", "user@example.com", "UserPass123")
    
    print(f"Admin created: {admin}")
    print(f"Admin permissions: {admin._permissions}")
    
    # Admin can access protected methods and attributes
    admin_details = admin.get_user_details(regular_user)
    print(f"Admin viewing user details: {admin_details}")
    
    # Admin can unlock accounts
    regular_user._login_attempts = 3
    regular_user.is_active = False
    admin.unlock_user_account(regular_user)
    print(f"User active after admin unlock: {regular_user.is_active}")
    print()
    
    # 4. Bank Account with Strict Access Control
    print("4. Bank Account with Strict Access Control:")
    
    account = BankAccount("1234567890", "Alice Johnson", 1000.0)
    print(f"Account created for: {account.account_holder}")
    print(f"Masked account number: {account.account_number}")
    
    # Set PIN
    pin_set = account.set_pin("1234")
    print(f"PIN set successfully: {pin_set}")
    
    # Access balance (property)
    print(f"Current balance: ${account.balance:.2f}")
    
    # Perform transactions
    deposit_success = account.deposit(200.0)
    print(f"Deposit successful: {deposit_success}")
    print(f"Balance after deposit: ${account.balance:.2f}")
    
    withdraw_success = account.withdraw(150.0, "1234")
    print(f"Withdrawal successful: {withdraw_success}")
    print(f"Balance after withdrawal: ${account.balance:.2f}")
    
    # Try withdrawal with wrong PIN
    wrong_pin_withdrawal = account.withdraw(100.0, "0000")
    print(f"Withdrawal with wrong PIN: {wrong_pin_withdrawal}")
    
    # Get statement with PIN
    statement = account.get_statement("1234")
    if statement:
        print(f"Statement transactions: {len(statement['transactions'])}")
    print()
    
    # 5. Document Security with Multiple Access Levels
    print("5. Document Security with Multiple Access Levels:")
    
    # Public document
    public_doc = SecureDocument("Public Announcement", "This is public information", "PUBLIC")
    print(f"Public document: {public_doc.title}")
    print(f"Public content: {public_doc.content}")
    
    # Classified document
    classified_doc = SecureDocument("Secret Project", "Classified information here", "CLASSIFIED")
    print(f"Classified document: {classified_doc.title}")
    print(f"Classified content (unauthorized): {classified_doc.content}")
    
    # Authorize user for classified document
    admin_key = "admin_master_key_2023"
    auth_success = classified_doc.authorize_user("user123", admin_key)
    print(f"User authorization successful: {auth_success}")
    
    # Access with authorization
    authorized_content = classified_doc.get_content_with_authorization("user123")
    print(f"Authorized content: {authorized_content}")
    
    # Try unauthorized access
    unauthorized_content = classified_doc.get_content_with_authorization("hacker456")
    print(f"Unauthorized access result: {unauthorized_content}")
    
    # Get access log
    access_log = classified_doc.get_access_log(admin_key)
    if access_log:
        print(f"Access log entries: {len(access_log)}")
        for entry in access_log[-3:]:  # Show last 3 entries
            print(f"  {entry['timestamp'][:19]}: {entry['action']}")
    print()
    
    # 6. Access Control Best Practices Summary
    print("6. Access Control Best Practices:")
    print("✓ Use single underscore (_) for protected members (convention)")
    print("✓ Use double underscore (__) for private members (name mangling)")
    print("✓ Provide public interfaces for controlled access")
    print("✓ Use properties for computed or validated attributes")
    print("✓ Implement authentication/authorization for sensitive operations")
    print("✓ Log access attempts for security auditing")
    print("✓ Follow principle of least privilege")
    print("✓ Validate all inputs in public methods")
    print()
    
    print("=== ACCESS MODIFIERS DESIGN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_access_modifiers()
