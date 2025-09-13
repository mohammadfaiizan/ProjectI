"""
ENCAPSULATION PRINCIPLES - Data Hiding and Access Control
========================================================

Problem Statement:
Demonstrate encapsulation concepts including:
- Private, protected, and public access modifiers
- Data hiding and information security
- Getter and setter methods (properties)
- Controlled access to object state
- Validation and data integrity

Learning Objectives:
- Understand access control mechanisms
- Implement proper data hiding
- Create controlled interfaces to object data
- Maintain data integrity through validation
"""

from typing import Optional, List
from datetime import datetime
import re


class BankAccount:
    """
    Bank Account class demonstrating encapsulation principles.
    Shows proper data hiding and controlled access to sensitive information.
    """
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float = 0.0):
        """
        Initialize bank account with proper encapsulation.
        
        Args:
            account_number: Unique account identifier
            account_holder: Name of account holder
            initial_balance: Starting balance (default 0.0)
        """
        # Private attributes (name mangling with __)
        self.__account_number = account_number
        self.__account_holder = account_holder
        self.__balance = 0.0
        self.__pin = None
        self.__is_active = True
        self.__transaction_history: List[dict] = []
        
        # Protected attributes (convention with _)
        self._creation_date = datetime.now()
        self._account_type = "SAVINGS"
        self._daily_withdrawal_limit = 5000.0
        self._daily_withdrawn = 0.0
        
        # Public attributes
        self.bank_name = "SecureBank"
        self.branch_code = "001"
        
        # Use setter to validate initial balance
        self.deposit(initial_balance)
    
    # Property decorators for controlled access
    @property
    def account_number(self) -> str:
        """Get account number (read-only)."""
        return self.__account_number
    
    @property
    def account_holder(self) -> str:
        """Get account holder name (read-only)."""
        return self.__account_holder
    
    @property
    def balance(self) -> float:
        """Get current balance (read-only)."""
        return self.__balance
    
    @property
    def is_active(self) -> bool:
        """Check if account is active."""
        return self.__is_active
    
    @property
    def account_type(self) -> str:
        """Get account type."""
        return self._account_type
    
    @account_type.setter
    def account_type(self, new_type: str) -> None:
        """
        Set account type with validation.
        
        Args:
            new_type: New account type
        """
        valid_types = ["SAVINGS", "CHECKING", "BUSINESS"]
        if new_type.upper() in valid_types:
            self._account_type = new_type.upper()
            self._add_transaction("ACCOUNT_TYPE_CHANGE", 0, f"Changed to {new_type}")
        else:
            raise ValueError(f"Invalid account type. Must be one of {valid_types}")
    
    @property
    def daily_withdrawal_limit(self) -> float:
        """Get daily withdrawal limit."""
        return self._daily_withdrawal_limit
    
    @daily_withdrawal_limit.setter
    def daily_withdrawal_limit(self, new_limit: float) -> None:
        """
        Set daily withdrawal limit with validation.
        
        Args:
            new_limit: New withdrawal limit
        """
        if new_limit > 0:
            self._daily_withdrawal_limit = new_limit
            self._add_transaction("LIMIT_CHANGE", 0, f"Withdrawal limit set to {new_limit}")
        else:
            raise ValueError("Withdrawal limit must be positive")
    
    def set_pin(self, new_pin: str, current_pin: Optional[str] = None) -> bool:
        """
        Set or change PIN with validation.
        
        Args:
            new_pin: New 4-digit PIN
            current_pin: Current PIN (required for changes)
            
        Returns:
            bool: True if PIN set successfully
        """
        # Validate PIN format
        if not self.__validate_pin_format(new_pin):
            print("Invalid PIN format. Must be 4 digits.")
            return False
        
        # Check current PIN if changing existing PIN
        if self.__pin is not None:
            if current_pin is None or not self.__verify_pin(current_pin):
                print("Current PIN verification failed.")
                return False
        
        self.__pin = new_pin
        self._add_transaction("PIN_CHANGE", 0, "PIN changed successfully")
        print("PIN set successfully.")
        return True
    
    def deposit(self, amount: float) -> bool:
        """
        Deposit money into account.
        
        Args:
            amount: Amount to deposit
            
        Returns:
            bool: True if deposit successful
        """
        if not self.__is_active:
            print("Account is inactive. Cannot perform deposit.")
            return False
        
        if amount <= 0:
            print("Deposit amount must be positive.")
            return False
        
        self.__balance += amount
        self._add_transaction("DEPOSIT", amount, f"Deposited {amount}")
        print(f"Deposited ${amount:.2f}. New balance: ${self.__balance:.2f}")
        return True
    
    def withdraw(self, amount: float, pin: str) -> bool:
        """
        Withdraw money from account with PIN verification.
        
        Args:
            amount: Amount to withdraw
            pin: Account PIN for verification
            
        Returns:
            bool: True if withdrawal successful
        """
        if not self.__is_active:
            print("Account is inactive. Cannot perform withdrawal.")
            return False
        
        if not self.__verify_pin(pin):
            print("Invalid PIN. Withdrawal denied.")
            return False
        
        if amount <= 0:
            print("Withdrawal amount must be positive.")
            return False
        
        if amount > self.__balance:
            print("Insufficient funds.")
            return False
        
        if self._daily_withdrawn + amount > self._daily_withdrawal_limit:
            print(f"Daily withdrawal limit exceeded. Limit: ${self._daily_withdrawal_limit:.2f}")
            return False
        
        self.__balance -= amount
        self._daily_withdrawn += amount
        self._add_transaction("WITHDRAWAL", -amount, f"Withdrew {amount}")
        print(f"Withdrew ${amount:.2f}. New balance: ${self.__balance:.2f}")
        return True
    
    def transfer(self, amount: float, target_account: 'BankAccount', pin: str) -> bool:
        """
        Transfer money to another account.
        
        Args:
            amount: Amount to transfer
            target_account: Destination account
            pin: Account PIN for verification
            
        Returns:
            bool: True if transfer successful
        """
        if not self.__verify_pin(pin):
            print("Invalid PIN. Transfer denied.")
            return False
        
        if self.withdraw(amount, pin):
            if target_account.deposit(amount):
                self._add_transaction("TRANSFER_OUT", -amount, 
                                    f"Transferred to {target_account.account_number}")
                target_account._add_transaction("TRANSFER_IN", amount, 
                                              f"Received from {self.__account_number}")
                print(f"Transfer of ${amount:.2f} completed successfully.")
                return True
            else:
                # Rollback withdrawal if deposit fails
                self.__balance += amount
                self._daily_withdrawn -= amount
                print("Transfer failed. Amount refunded.")
                return False
        return False
    
    def get_balance(self, pin: str) -> Optional[float]:
        """
        Get account balance with PIN verification.
        
        Args:
            pin: Account PIN for verification
            
        Returns:
            Optional[float]: Account balance if PIN is correct, None otherwise
        """
        if self.__verify_pin(pin):
            return self.__balance
        else:
            print("Invalid PIN. Balance inquiry denied.")
            return None
    
    def get_transaction_history(self, pin: str, limit: int = 10) -> Optional[List[dict]]:
        """
        Get transaction history with PIN verification.
        
        Args:
            pin: Account PIN for verification
            limit: Maximum number of transactions to return
            
        Returns:
            Optional[List[dict]]: Transaction history if PIN is correct
        """
        if self.__verify_pin(pin):
            return self.__transaction_history[-limit:]
        else:
            print("Invalid PIN. Transaction history access denied.")
            return None
    
    def freeze_account(self, pin: str) -> bool:
        """
        Freeze account to prevent transactions.
        
        Args:
            pin: Account PIN for verification
            
        Returns:
            bool: True if account frozen successfully
        """
        if self.__verify_pin(pin):
            self.__is_active = False
            self._add_transaction("ACCOUNT_FREEZE", 0, "Account frozen by user")
            print("Account has been frozen.")
            return True
        else:
            print("Invalid PIN. Cannot freeze account.")
            return False
    
    def unfreeze_account(self, pin: str) -> bool:
        """
        Unfreeze account to allow transactions.
        
        Args:
            pin: Account PIN for verification
            
        Returns:
            bool: True if account unfrozen successfully
        """
        if self.__verify_pin(pin):
            self.__is_active = True
            self._add_transaction("ACCOUNT_UNFREEZE", 0, "Account unfrozen by user")
            print("Account has been unfrozen.")
            return True
        else:
            print("Invalid PIN. Cannot unfreeze account.")
            return False
    
    # Private methods (internal implementation)
    def __validate_pin_format(self, pin: str) -> bool:
        """
        Validate PIN format (private method).
        
        Args:
            pin: PIN to validate
            
        Returns:
            bool: True if PIN format is valid
        """
        return len(pin) == 4 and pin.isdigit()
    
    def __verify_pin(self, pin: str) -> bool:
        """
        Verify PIN against stored PIN (private method).
        
        Args:
            pin: PIN to verify
            
        Returns:
            bool: True if PIN is correct
        """
        return self.__pin is not None and self.__pin == pin
    
    # Protected method (for internal use and subclasses)
    def _add_transaction(self, transaction_type: str, amount: float, description: str) -> None:
        """
        Add transaction to history (protected method).
        
        Args:
            transaction_type: Type of transaction
            amount: Transaction amount
            description: Transaction description
        """
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': transaction_type,
            'amount': amount,
            'description': description,
            'balance_after': self.__balance
        }
        self.__transaction_history.append(transaction)
    
    def _reset_daily_limit(self) -> None:
        """Reset daily withdrawal counter (protected method)."""
        self._daily_withdrawn = 0.0
        self._add_transaction("DAILY_RESET", 0, "Daily withdrawal limit reset")
    
    # Public method for account information
    def get_account_info(self) -> dict:
        """
        Get non-sensitive account information.
        
        Returns:
            dict: Public account information
        """
        return {
            'account_number': self.__account_number[-4:].rjust(len(self.__account_number), '*'),
            'account_holder': self.__account_holder,
            'account_type': self._account_type,
            'bank_name': self.bank_name,
            'branch_code': self.branch_code,
            'is_active': self.__is_active,
            'creation_date': self._creation_date.strftime("%Y-%m-%d")
        }
    
    def __str__(self) -> str:
        """String representation hiding sensitive information."""
        masked_number = self.__account_number[-4:].rjust(len(self.__account_number), '*')
        return f"BankAccount({masked_number}, {self.__account_holder}, Active: {self.__is_active})"
    
    def __repr__(self) -> str:
        """Developer representation hiding sensitive information."""
        return f"BankAccount('{self.__account_number[-4:]}', '{self.__account_holder}')"


class SecureCreditCard:
    """
    Credit Card class demonstrating advanced encapsulation.
    """
    
    def __init__(self, card_number: str, cardholder_name: str, credit_limit: float):
        # Private attributes with validation
        self.__card_number = self.__validate_and_set_card_number(card_number)
        self.__cardholder_name = cardholder_name
        self.__credit_limit = credit_limit
        self.__current_balance = 0.0
        self.__cvv = None
        self.__expiry_date = None
        self.__is_blocked = False
        
        # Protected attributes
        self._card_type = self.__determine_card_type()
        self._transaction_history: List[dict] = []
    
    @property
    def masked_card_number(self) -> str:
        """Get masked card number for display."""
        return f"****-****-****-{self.__card_number[-4:]}"
    
    @property
    def cardholder_name(self) -> str:
        """Get cardholder name."""
        return self.__cardholder_name
    
    @property
    def available_credit(self) -> float:
        """Get available credit amount."""
        return self.__credit_limit - self.__current_balance
    
    @property
    def current_balance(self) -> float:
        """Get current balance."""
        return self.__current_balance
    
    @property
    def credit_limit(self) -> float:
        """Get credit limit."""
        return self.__credit_limit
    
    @credit_limit.setter
    def credit_limit(self, new_limit: float) -> None:
        """Set new credit limit with validation."""
        if new_limit > 0 and new_limit >= self.__current_balance:
            self.__credit_limit = new_limit
            print(f"Credit limit updated to ${new_limit:.2f}")
        else:
            raise ValueError("Invalid credit limit")
    
    def __validate_and_set_card_number(self, card_number: str) -> str:
        """Validate card number using Luhn algorithm."""
        # Remove spaces and dashes
        clean_number = re.sub(r'[\s-]', '', card_number)
        
        if not clean_number.isdigit() or len(clean_number) != 16:
            raise ValueError("Invalid card number format")
        
        # Luhn algorithm validation
        if not self.__luhn_check(clean_number):
            raise ValueError("Invalid card number (failed Luhn check)")
        
        return clean_number
    
    def __luhn_check(self, card_number: str) -> bool:
        """Implement Luhn algorithm for card validation."""
        digits = [int(d) for d in card_number]
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0
    
    def __determine_card_type(self) -> str:
        """Determine card type based on number."""
        if self.__card_number.startswith('4'):
            return 'VISA'
        elif self.__card_number.startswith('5'):
            return 'MASTERCARD'
        elif self.__card_number.startswith('3'):
            return 'AMEX'
        else:
            return 'UNKNOWN'
    
    def make_purchase(self, amount: float, merchant: str) -> bool:
        """Make a purchase with the card."""
        if self.__is_blocked:
            print("Card is blocked. Transaction denied.")
            return False
        
        if amount <= 0:
            print("Invalid purchase amount.")
            return False
        
        if self.__current_balance + amount > self.__credit_limit:
            print("Credit limit exceeded. Transaction denied.")
            return False
        
        self.__current_balance += amount
        self._add_transaction("PURCHASE", amount, f"Purchase at {merchant}")
        print(f"Purchase of ${amount:.2f} at {merchant} approved.")
        return True
    
    def make_payment(self, amount: float) -> bool:
        """Make a payment towards the card balance."""
        if amount <= 0:
            print("Payment amount must be positive.")
            return False
        
        if amount > self.__current_balance:
            amount = self.__current_balance
        
        self.__current_balance -= amount
        self._add_transaction("PAYMENT", -amount, f"Payment of ${amount:.2f}")
        print(f"Payment of ${amount:.2f} processed. New balance: ${self.__current_balance:.2f}")
        return True
    
    def _add_transaction(self, transaction_type: str, amount: float, description: str) -> None:
        """Add transaction to history."""
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'type': transaction_type,
            'amount': amount,
            'description': description,
            'balance_after': self.__current_balance
        }
        self._transaction_history.append(transaction)
    
    def get_statement(self) -> dict:
        """Get card statement with transaction history."""
        return {
            'card_number': self.masked_card_number,
            'cardholder': self.__cardholder_name,
            'current_balance': self.__current_balance,
            'credit_limit': self.__credit_limit,
            'available_credit': self.available_credit,
            'transactions': self._transaction_history[-10:]  # Last 10 transactions
        }


def demonstrate_encapsulation():
    """
    Demonstrate encapsulation principles with practical examples.
    """
    print("=== ENCAPSULATION PRINCIPLES DEMONSTRATION ===\n")
    
    # 1. Creating Bank Account with Encapsulation
    print("1. Creating Bank Account:")
    account = BankAccount("ACC123456789", "John Doe", 1000.0)
    print(f"Account created: {account}")
    print(f"Account info: {account.get_account_info()}")
    print()
    
    # 2. Setting PIN (required for sensitive operations)
    print("2. Setting Account PIN:")
    account.set_pin("1234")
    print()
    
    # 3. Accessing Properties (Controlled Access)
    print("3. Accessing Account Properties:")
    print(f"Account Number: {account.account_number}")
    print(f"Account Holder: {account.account_holder}")
    print(f"Balance: ${account.balance:.2f}")
    print(f"Account Type: {account.account_type}")
    print(f"Is Active: {account.is_active}")
    print()
    
    # 4. Using Setters with Validation
    print("4. Using Property Setters:")
    try:
        account.account_type = "CHECKING"
        print(f"Account type changed to: {account.account_type}")
        
        account.daily_withdrawal_limit = 3000.0
        print(f"Daily withdrawal limit set to: ${account.daily_withdrawal_limit:.2f}")
    except ValueError as e:
        print(f"Error: {e}")
    print()
    
    # 5. Secure Operations with PIN
    print("5. Secure Operations:")
    account.withdraw(200.0, "1234")  # Correct PIN
    account.withdraw(100.0, "0000")  # Wrong PIN
    
    balance = account.get_balance("1234")
    if balance is not None:
        print(f"Current balance: ${balance:.2f}")
    print()
    
    # 6. Transfer Between Accounts
    print("6. Account Transfer:")
    account2 = BankAccount("ACC987654321", "Jane Smith", 500.0)
    account2.set_pin("5678")
    
    account.transfer(150.0, account2, "1234")
    print(f"Account 1 balance: ${account.get_balance('1234'):.2f}")
    print(f"Account 2 balance: ${account2.get_balance('5678'):.2f}")
    print()
    
    # 7. Transaction History (Secure Access)
    print("7. Transaction History:")
    history = account.get_transaction_history("1234", 5)
    if history:
        for transaction in history:
            print(f"  {transaction['timestamp'][:19]}: {transaction['type']} - {transaction['description']}")
    print()
    
    # 8. Credit Card Encapsulation
    print("8. Credit Card Encapsulation:")
    try:
        card = SecureCreditCard("4532123456789012", "John Doe", 5000.0)
        print(f"Card created: {card.masked_card_number}")
        print(f"Available credit: ${card.available_credit:.2f}")
        
        card.make_purchase(250.0, "Amazon")
        card.make_purchase(100.0, "Gas Station")
        card.make_payment(150.0)
        
        statement = card.get_statement()
        print(f"Current balance: ${statement['current_balance']:.2f}")
        print(f"Available credit: ${statement['available_credit']:.2f}")
        
    except ValueError as e:
        print(f"Error creating card: {e}")
    print()
    
    # 9. Attempting to Access Private Members (Will Fail)
    print("9. Attempting Direct Access to Private Members:")
    try:
        # This will cause AttributeError
        print(account.__balance)
    except AttributeError:
        print("Cannot access private attribute __balance directly")
    
    try:
        # This will also fail
        print(account.__account_number)
    except AttributeError:
        print("Cannot access private attribute __account_number directly")
    print()
    
    print("=== ENCAPSULATION DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_encapsulation()
