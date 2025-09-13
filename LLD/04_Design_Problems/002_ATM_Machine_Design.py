"""
ATM MACHINE DESIGN - Complete System Design
============================================

Problem Statement:
Design a comprehensive ATM (Automated Teller Machine) system that handles:
- Multiple account types (Checking, Savings, Credit)
- Various transaction types (Withdraw, Deposit, Transfer, Balance Inquiry)
- Card authentication and PIN verification
- Cash dispensing with denomination management
- Receipt printing and transaction logging
- Daily withdrawal limits and security features
- Network communication with bank systems
- Error handling and recovery mechanisms

Requirements:
- Support multiple card types (Debit, Credit, ATM)
- Implement secure PIN verification with lockout
- Handle cash management and denomination tracking
- Enforce daily withdrawal limits per account
- Generate detailed transaction receipts
- Maintain audit logs for all operations
- Handle network failures gracefully
- Support multiple languages and accessibility
- Implement fraud detection mechanisms

Design Patterns Used:
- State: ATM operational states
- Strategy: Transaction processing strategies
- Command: Transaction operations
- Observer: Transaction monitoring
- Factory: Card and account creation
- Singleton: ATM machine instance
- Decorator: Security enhancements
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
from enum import Enum
import uuid
import threading
import time
from dataclasses import dataclass, field
import json
import hashlib


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ATMState(Enum):
    IDLE = "idle"
    CARD_INSERTED = "card_inserted"
    PIN_VERIFICATION = "pin_verification"
    MENU_SELECTION = "menu_selection"
    TRANSACTION_PROCESSING = "transaction_processing"
    CASH_DISPENSING = "cash_dispensing"
    RECEIPT_PRINTING = "receipt_printing"
    OUT_OF_SERVICE = "out_of_service"
    MAINTENANCE = "maintenance"


class TransactionType(Enum):
    WITHDRAW = "withdraw"
    DEPOSIT = "deposit"
    TRANSFER = "transfer"
    BALANCE_INQUIRY = "balance_inquiry"
    PIN_CHANGE = "pin_change"
    MINI_STATEMENT = "mini_statement"


class AccountType(Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"


class CardType(Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    ATM = "atm"


class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CashDenomination:
    """Cash denomination with count."""
    value: int
    count: int
    
    @property
    def total_value(self) -> int:
        return self.value * self.count


@dataclass
class TransactionResult:
    """Transaction result data."""
    success: bool
    message: str
    transaction_id: str
    amount: float = 0.0
    balance: float = 0.0
    receipt_data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CARD AND ACCOUNT CLASSES
# ============================================================================

class Card:
    """ATM card with authentication capabilities."""
    
    def __init__(self, card_number: str, card_type: CardType, pin_hash: str, 
                 account_numbers: List[str], expiry_date: date):
        self.card_number = card_number
        self.card_type = card_type
        self.pin_hash = pin_hash
        self.account_numbers = account_numbers
        self.expiry_date = expiry_date
        self.is_blocked = False
        self.failed_attempts = 0
        self.last_used = None
        self.daily_limit = self._get_daily_limit()
        self.daily_used = 0.0
        self.last_limit_reset = date.today()
    
    def _get_daily_limit(self) -> float:
        """Get daily withdrawal limit based on card type."""
        limits = {
            CardType.DEBIT: 1000.0,
            CardType.CREDIT: 2000.0,
            CardType.ATM: 500.0
        }
        return limits.get(self.card_type, 500.0)
    
    def verify_pin(self, pin: str) -> bool:
        """Verify PIN and handle failed attempts."""
        if self.is_blocked:
            return False
        
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()
        
        if pin_hash == self.pin_hash:
            self.failed_attempts = 0
            self.last_used = datetime.now()
            return True
        else:
            self.failed_attempts += 1
            if self.failed_attempts >= 3:
                self.is_blocked = True
            return False
    
    def is_expired(self) -> bool:
        """Check if card is expired."""
        return date.today() > self.expiry_date
    
    def can_withdraw(self, amount: float) -> bool:
        """Check if withdrawal amount is within daily limit."""
        self._reset_daily_limit_if_needed()
        return (self.daily_used + amount) <= self.daily_limit
    
    def record_withdrawal(self, amount: float) -> None:
        """Record withdrawal amount for daily limit tracking."""
        self._reset_daily_limit_if_needed()
        self.daily_used += amount
    
    def _reset_daily_limit_if_needed(self) -> None:
        """Reset daily limit if new day."""
        if self.last_limit_reset < date.today():
            self.daily_used = 0.0
            self.last_limit_reset = date.today()
    
    def get_card_info(self) -> Dict[str, Any]:
        """Get card information."""
        return {
            'card_number': f"****{self.card_number[-4:]}",
            'card_type': self.card_type.value,
            'is_blocked': self.is_blocked,
            'is_expired': self.is_expired(),
            'daily_limit': self.daily_limit,
            'daily_used': self.daily_used,
            'remaining_limit': self.daily_limit - self.daily_used
        }


class Account:
    """Bank account with transaction capabilities."""
    
    def __init__(self, account_number: str, account_type: AccountType, 
                 balance: float, customer_name: str):
        self.account_number = account_number
        self.account_type = account_type
        self.balance = balance
        self.customer_name = customer_name
        self.is_active = True
        self.transaction_history: List[Dict] = []
        self.daily_withdrawal_limit = self._get_daily_withdrawal_limit()
        self.daily_withdrawn = 0.0
        self.last_limit_reset = date.today()
        self._lock = threading.Lock()
    
    def _get_daily_withdrawal_limit(self) -> float:
        """Get daily withdrawal limit based on account type."""
        limits = {
            AccountType.CHECKING: 1500.0,
            AccountType.SAVINGS: 1000.0,
            AccountType.CREDIT: 3000.0
        }
        return limits.get(self.account_type, 1000.0)
    
    def can_withdraw(self, amount: float) -> Tuple[bool, str]:
        """Check if withdrawal is possible."""
        with self._lock:
            if not self.is_active:
                return False, "Account is inactive"
            
            self._reset_daily_limit_if_needed()
            
            if self.daily_withdrawn + amount > self.daily_withdrawal_limit:
                return False, "Daily withdrawal limit exceeded"
            
            if self.account_type == AccountType.CREDIT:
                # Credit accounts can go negative up to credit limit
                credit_limit = 5000.0  # Simplified credit limit
                if self.balance - amount < -credit_limit:
                    return False, "Credit limit exceeded"
            else:
                if self.balance < amount:
                    return False, "Insufficient funds"
            
            return True, "Withdrawal allowed"
    
    def withdraw(self, amount: float, transaction_id: str) -> bool:
        """Withdraw amount from account."""
        with self._lock:
            can_withdraw, message = self.can_withdraw(amount)
            if not can_withdraw:
                return False
            
            self.balance -= amount
            self.daily_withdrawn += amount
            
            self._add_transaction(TransactionType.WITHDRAW, -amount, transaction_id)
            return True
    
    def deposit(self, amount: float, transaction_id: str) -> bool:
        """Deposit amount to account."""
        with self._lock:
            if not self.is_active or amount <= 0:
                return False
            
            self.balance += amount
            self._add_transaction(TransactionType.DEPOSIT, amount, transaction_id)
            return True
    
    def transfer_out(self, amount: float, to_account: str, transaction_id: str) -> bool:
        """Transfer amount out of account."""
        with self._lock:
            can_withdraw, message = self.can_withdraw(amount)
            if not can_withdraw:
                return False
            
            self.balance -= amount
            self.daily_withdrawn += amount
            
            self._add_transaction(TransactionType.TRANSFER, -amount, transaction_id, 
                                {'to_account': to_account})
            return True
    
    def transfer_in(self, amount: float, from_account: str, transaction_id: str) -> bool:
        """Transfer amount into account."""
        with self._lock:
            if not self.is_active or amount <= 0:
                return False
            
            self.balance += amount
            self._add_transaction(TransactionType.TRANSFER, amount, transaction_id,
                                {'from_account': from_account})
            return True
    
    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance
    
    def get_mini_statement(self, count: int = 5) -> List[Dict]:
        """Get recent transactions."""
        return self.transaction_history[-count:] if self.transaction_history else []
    
    def _add_transaction(self, transaction_type: TransactionType, amount: float,
                        transaction_id: str, metadata: Dict = None) -> None:
        """Add transaction to history."""
        transaction = {
            'transaction_id': transaction_id,
            'type': transaction_type.value,
            'amount': amount,
            'balance_after': self.balance,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.transaction_history.append(transaction)
    
    def _reset_daily_limit_if_needed(self) -> None:
        """Reset daily withdrawal limit if new day."""
        if self.last_limit_reset < date.today():
            self.daily_withdrawn = 0.0
            self.last_limit_reset = date.today()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return {
            'account_number': f"****{self.account_number[-4:]}",
            'account_type': self.account_type.value,
            'balance': self.balance,
            'customer_name': self.customer_name,
            'is_active': self.is_active,
            'daily_limit': self.daily_withdrawal_limit,
            'daily_withdrawn': self.daily_withdrawn
        }


# ============================================================================
# TRANSACTION STRATEGIES
# ============================================================================

class TransactionStrategy(ABC):
    """Abstract strategy for transaction processing."""
    
    @abstractmethod
    def execute(self, atm: 'ATMMachine', account: Account, amount: float, 
                **kwargs) -> TransactionResult:
        """Execute transaction."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class WithdrawalStrategy(TransactionStrategy):
    """Strategy for withdrawal transactions."""
    
    def execute(self, atm: 'ATMMachine', account: Account, amount: float, 
                **kwargs) -> TransactionResult:
        """Execute withdrawal transaction."""
        transaction_id = str(uuid.uuid4())
        
        # Check if ATM has enough cash
        if not atm.cash_dispenser.can_dispense(amount):
            return TransactionResult(
                success=False,
                message="ATM has insufficient cash",
                transaction_id=transaction_id
            )
        
        # Check account withdrawal capability
        can_withdraw, message = account.can_withdraw(amount)
        if not can_withdraw:
            return TransactionResult(
                success=False,
                message=message,
                transaction_id=transaction_id
            )
        
        # Perform withdrawal
        if account.withdraw(amount, transaction_id):
            # Dispense cash
            denominations = atm.cash_dispenser.dispense_cash(amount)
            
            if denominations:
                receipt_data = {
                    'transaction_type': 'Withdrawal',
                    'amount': amount,
                    'account_number': f"****{account.account_number[-4:]}",
                    'balance': account.balance,
                    'denominations': denominations,
                    'location': atm.location,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return TransactionResult(
                    success=True,
                    message=f"Withdrawal successful. Please take your cash.",
                    transaction_id=transaction_id,
                    amount=amount,
                    balance=account.balance,
                    receipt_data=receipt_data
                )
            else:
                # Rollback account withdrawal if cash dispensing failed
                account.deposit(amount, transaction_id + "_rollback")
                return TransactionResult(
                    success=False,
                    message="Cash dispensing failed",
                    transaction_id=transaction_id
                )
        
        return TransactionResult(
            success=False,
            message="Withdrawal failed",
            transaction_id=transaction_id
        )
    
    def get_strategy_name(self) -> str:
        return "Withdrawal Strategy"


class DepositStrategy(TransactionStrategy):
    """Strategy for deposit transactions."""
    
    def execute(self, atm: 'ATMMachine', account: Account, amount: float, 
                **kwargs) -> TransactionResult:
        """Execute deposit transaction."""
        transaction_id = str(uuid.uuid4())
        
        if amount <= 0:
            return TransactionResult(
                success=False,
                message="Invalid deposit amount",
                transaction_id=transaction_id
            )
        
        # Perform deposit
        if account.deposit(amount, transaction_id):
            receipt_data = {
                'transaction_type': 'Deposit',
                'amount': amount,
                'account_number': f"****{account.account_number[-4:]}",
                'balance': account.balance,
                'location': atm.location,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return TransactionResult(
                success=True,
                message="Deposit successful",
                transaction_id=transaction_id,
                amount=amount,
                balance=account.balance,
                receipt_data=receipt_data
            )
        
        return TransactionResult(
            success=False,
            message="Deposit failed",
            transaction_id=transaction_id
        )
    
    def get_strategy_name(self) -> str:
        return "Deposit Strategy"


class TransferStrategy(TransactionStrategy):
    """Strategy for transfer transactions."""
    
    def execute(self, atm: 'ATMMachine', account: Account, amount: float, 
                **kwargs) -> TransactionResult:
        """Execute transfer transaction."""
        transaction_id = str(uuid.uuid4())
        to_account_number = kwargs.get('to_account_number')
        
        if not to_account_number:
            return TransactionResult(
                success=False,
                message="Destination account number required",
                transaction_id=transaction_id
            )
        
        # Get destination account
        to_account = atm.bank_network.get_account(to_account_number)
        if not to_account:
            return TransactionResult(
                success=False,
                message="Destination account not found",
                transaction_id=transaction_id
            )
        
        # Perform transfer
        if account.transfer_out(amount, to_account_number, transaction_id):
            if to_account.transfer_in(amount, account.account_number, transaction_id):
                receipt_data = {
                    'transaction_type': 'Transfer',
                    'amount': amount,
                    'from_account': f"****{account.account_number[-4:]}",
                    'to_account': f"****{to_account_number[-4:]}",
                    'balance': account.balance,
                    'location': atm.location,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return TransactionResult(
                    success=True,
                    message="Transfer successful",
                    transaction_id=transaction_id,
                    amount=amount,
                    balance=account.balance,
                    receipt_data=receipt_data
                )
            else:
                # Rollback if destination transfer failed
                account.deposit(amount, transaction_id + "_rollback")
                return TransactionResult(
                    success=False,
                    message="Transfer to destination account failed",
                    transaction_id=transaction_id
                )
        
        return TransactionResult(
            success=False,
            message="Transfer failed - insufficient funds or limit exceeded",
            transaction_id=transaction_id
        )
    
    def get_strategy_name(self) -> str:
        return "Transfer Strategy"


class BalanceInquiryStrategy(TransactionStrategy):
    """Strategy for balance inquiry transactions."""
    
    def execute(self, atm: 'ATMMachine', account: Account, amount: float = 0, 
                **kwargs) -> TransactionResult:
        """Execute balance inquiry transaction."""
        transaction_id = str(uuid.uuid4())
        
        receipt_data = {
            'transaction_type': 'Balance Inquiry',
            'account_number': f"****{account.account_number[-4:]}",
            'balance': account.balance,
            'account_type': account.account_type.value,
            'location': atm.location,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return TransactionResult(
            success=True,
            message=f"Current balance: ${account.balance:.2f}",
            transaction_id=transaction_id,
            balance=account.balance,
            receipt_data=receipt_data
        )
    
    def get_strategy_name(self) -> str:
        return "Balance Inquiry Strategy"


# ============================================================================
# CASH MANAGEMENT
# ============================================================================

class CashDispenser:
    """Cash dispenser with denomination management."""
    
    def __init__(self):
        self.denominations = {
            100: CashDenomination(100, 50),  # $100 bills
            50: CashDenomination(50, 100),   # $50 bills
            20: CashDenomination(20, 200),   # $20 bills
            10: CashDenomination(10, 100),   # $10 bills
            5: CashDenomination(5, 50),      # $5 bills
            1: CashDenomination(1, 100)      # $1 bills
        }
        self._lock = threading.Lock()
    
    def get_total_cash(self) -> int:
        """Get total cash available."""
        return sum(denom.total_value for denom in self.denominations.values())
    
    def can_dispense(self, amount: float) -> bool:
        """Check if amount can be dispensed."""
        with self._lock:
            return self._calculate_denominations(int(amount)) is not None
    
    def dispense_cash(self, amount: float) -> Optional[Dict[int, int]]:
        """Dispense cash and return denominations used."""
        with self._lock:
            amount_int = int(amount)
            denominations_needed = self._calculate_denominations(amount_int)
            
            if denominations_needed:
                # Deduct from available cash
                for value, count in denominations_needed.items():
                    self.denominations[value].count -= count
                
                return denominations_needed
            
            return None
    
    def _calculate_denominations(self, amount: int) -> Optional[Dict[int, int]]:
        """Calculate denominations needed for amount."""
        result = {}
        remaining = amount
        
        # Sort denominations in descending order
        for value in sorted(self.denominations.keys(), reverse=True):
            available_count = self.denominations[value].count
            needed_count = min(remaining // value, available_count)
            
            if needed_count > 0:
                result[value] = needed_count
                remaining -= value * needed_count
        
        # Check if exact amount can be dispensed
        if remaining == 0:
            return result
        else:
            return None
    
    def add_cash(self, denominations: Dict[int, int]) -> None:
        """Add cash to dispenser."""
        with self._lock:
            for value, count in denominations.items():
                if value in self.denominations:
                    self.denominations[value].count += count
    
    def get_cash_status(self) -> Dict[str, Any]:
        """Get cash dispenser status."""
        return {
            'total_cash': self.get_total_cash(),
            'denominations': {
                value: {'count': denom.count, 'total_value': denom.total_value}
                for value, denom in self.denominations.items()
            },
            'low_cash_warning': self.get_total_cash() < 1000
        }


# ============================================================================
# BANK NETWORK SIMULATION
# ============================================================================

class BankNetwork:
    """Simulated bank network for account operations."""
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.cards: Dict[str, Card] = {}
        self._lock = threading.Lock()
    
    def add_account(self, account: Account) -> None:
        """Add account to network."""
        with self._lock:
            self.accounts[account.account_number] = account
    
    def add_card(self, card: Card) -> None:
        """Add card to network."""
        with self._lock:
            self.cards[card.card_number] = card
    
    def get_account(self, account_number: str) -> Optional[Account]:
        """Get account by number."""
        return self.accounts.get(account_number)
    
    def get_card(self, card_number: str) -> Optional[Card]:
        """Get card by number."""
        return self.cards.get(card_number)
    
    def authenticate_card(self, card_number: str, pin: str) -> Tuple[bool, Optional[Card]]:
        """Authenticate card with PIN."""
        card = self.get_card(card_number)
        
        if not card:
            return False, None
        
        if card.is_blocked or card.is_expired():
            return False, card
        
        if card.verify_pin(pin):
            return True, card
        
        return False, card
    
    def get_accounts_for_card(self, card_number: str) -> List[Account]:
        """Get all accounts associated with a card."""
        card = self.get_card(card_number)
        if not card:
            return []
        
        accounts = []
        for account_number in card.account_numbers:
            account = self.get_account(account_number)
            if account:
                accounts.append(account)
        
        return accounts


# ============================================================================
# ATM STATES
# ============================================================================

class ATMStateHandler(ABC):
    """Abstract ATM state handler."""
    
    @abstractmethod
    def handle_card_insert(self, atm: 'ATMMachine', card_number: str) -> None:
        """Handle card insertion."""
        pass
    
    @abstractmethod
    def handle_pin_entry(self, atm: 'ATMMachine', pin: str) -> None:
        """Handle PIN entry."""
        pass
    
    @abstractmethod
    def handle_menu_selection(self, atm: 'ATMMachine', selection: str) -> None:
        """Handle menu selection."""
        pass
    
    @abstractmethod
    def handle_transaction(self, atm: 'ATMMachine', transaction_data: Dict) -> None:
        """Handle transaction processing."""
        pass
    
    @abstractmethod
    def handle_card_eject(self, atm: 'ATMMachine') -> None:
        """Handle card ejection."""
        pass


class IdleState(ATMStateHandler):
    """ATM idle state."""
    
    def handle_card_insert(self, atm: 'ATMMachine', card_number: str) -> None:
        """Handle card insertion in idle state."""
        atm.current_card_number = card_number
        atm.pin_attempts = 0
        atm.set_state(ATMState.CARD_INSERTED)
        atm.display_message("Card inserted. Please enter your PIN.")
    
    def handle_pin_entry(self, atm: 'ATMMachine', pin: str) -> None:
        """PIN entry not allowed in idle state."""
        atm.display_message("Please insert your card first.")
    
    def handle_menu_selection(self, atm: 'ATMMachine', selection: str) -> None:
        """Menu selection not allowed in idle state."""
        atm.display_message("Please insert your card first.")
    
    def handle_transaction(self, atm: 'ATMMachine', transaction_data: Dict) -> None:
        """Transaction not allowed in idle state."""
        atm.display_message("Please insert your card first.")
    
    def handle_card_eject(self, atm: 'ATMMachine') -> None:
        """No card to eject in idle state."""
        atm.display_message("No card inserted.")


class CardInsertedState(ATMStateHandler):
    """ATM card inserted state."""
    
    def handle_card_insert(self, atm: 'ATMMachine', card_number: str) -> None:
        """Card already inserted."""
        atm.display_message("Card already inserted. Please enter your PIN.")
    
    def handle_pin_entry(self, atm: 'ATMMachine', pin: str) -> None:
        """Handle PIN entry."""
        success, card = atm.bank_network.authenticate_card(atm.current_card_number, pin)
        
        if success:
            atm.current_card = card
            atm.current_accounts = atm.bank_network.get_accounts_for_card(card.card_number)
            atm.set_state(ATMState.MENU_SELECTION)
            atm.display_main_menu()
        else:
            atm.pin_attempts += 1
            if atm.pin_attempts >= 3:
                atm.display_message("Too many failed attempts. Card retained.")
                atm.eject_card()
            else:
                remaining = 3 - atm.pin_attempts
                atm.display_message(f"Invalid PIN. {remaining} attempts remaining.")
    
    def handle_menu_selection(self, atm: 'ATMMachine', selection: str) -> None:
        """Menu selection not allowed without PIN verification."""
        atm.display_message("Please enter your PIN first.")
    
    def handle_transaction(self, atm: 'ATMMachine', transaction_data: Dict) -> None:
        """Transaction not allowed without PIN verification."""
        atm.display_message("Please enter your PIN first.")
    
    def handle_card_eject(self, atm: 'ATMMachine') -> None:
        """Eject card."""
        atm.eject_card()


class MenuSelectionState(ATMStateHandler):
    """ATM menu selection state."""
    
    def handle_card_insert(self, atm: 'ATMMachine', card_number: str) -> None:
        """Card already inserted."""
        atm.display_message("Card already inserted. Please make a selection.")
    
    def handle_pin_entry(self, atm: 'ATMMachine', pin: str) -> None:
        """PIN already verified."""
        atm.display_message("PIN already verified. Please make a selection.")
    
    def handle_menu_selection(self, atm: 'ATMMachine', selection: str) -> None:
        """Handle menu selection."""
        if selection == "withdraw":
            atm.display_message("Enter withdrawal amount:")
            atm.current_transaction_type = TransactionType.WITHDRAW
            atm.set_state(ATMState.TRANSACTION_PROCESSING)
        elif selection == "deposit":
            atm.display_message("Enter deposit amount:")
            atm.current_transaction_type = TransactionType.DEPOSIT
            atm.set_state(ATMState.TRANSACTION_PROCESSING)
        elif selection == "transfer":
            atm.display_message("Enter transfer details:")
            atm.current_transaction_type = TransactionType.TRANSFER
            atm.set_state(ATMState.TRANSACTION_PROCESSING)
        elif selection == "balance":
            atm.current_transaction_type = TransactionType.BALANCE_INQUIRY
            atm.process_transaction({'amount': 0})
        elif selection == "exit":
            atm.eject_card()
        else:
            atm.display_message("Invalid selection. Please try again.")
    
    def handle_transaction(self, atm: 'ATMMachine', transaction_data: Dict) -> None:
        """Transaction processing from menu selection."""
        atm.process_transaction(transaction_data)
    
    def handle_card_eject(self, atm: 'ATMMachine') -> None:
        """Eject card."""
        atm.eject_card()


class TransactionProcessingState(ATMStateHandler):
    """ATM transaction processing state."""
    
    def handle_card_insert(self, atm: 'ATMMachine', card_number: str) -> None:
        """Card insertion not allowed during transaction."""
        atm.display_message("Transaction in progress. Please wait.")
    
    def handle_pin_entry(self, atm: 'ATMMachine', pin: str) -> None:
        """PIN entry not allowed during transaction."""
        atm.display_message("Transaction in progress. Please wait.")
    
    def handle_menu_selection(self, atm: 'ATMMachine', selection: str) -> None:
        """Menu selection not allowed during transaction."""
        atm.display_message("Transaction in progress. Please wait.")
    
    def handle_transaction(self, atm: 'ATMMachine', transaction_data: Dict) -> None:
        """Process the transaction."""
        atm.process_transaction(transaction_data)
    
    def handle_card_eject(self, atm: 'ATMMachine') -> None:
        """Cancel transaction and eject card."""
        atm.display_message("Transaction cancelled.")
        atm.eject_card()


# ============================================================================
# MAIN ATM MACHINE
# ============================================================================

class ATMMachine:
    """Main ATM machine with state management."""
    
    def __init__(self, atm_id: str, location: str, bank_network: BankNetwork):
        self.atm_id = atm_id
        self.location = location
        self.bank_network = bank_network
        self.cash_dispenser = CashDispenser()
        
        # State management
        self.current_state = ATMState.IDLE
        self.state_handlers = {
            ATMState.IDLE: IdleState(),
            ATMState.CARD_INSERTED: CardInsertedState(),
            ATMState.MENU_SELECTION: MenuSelectionState(),
            ATMState.TRANSACTION_PROCESSING: TransactionProcessingState()
        }
        
        # Transaction strategies
        self.transaction_strategies = {
            TransactionType.WITHDRAW: WithdrawalStrategy(),
            TransactionType.DEPOSIT: DepositStrategy(),
            TransactionType.TRANSFER: TransferStrategy(),
            TransactionType.BALANCE_INQUIRY: BalanceInquiryStrategy()
        }
        
        # Current session data
        self.current_card_number: Optional[str] = None
        self.current_card: Optional[Card] = None
        self.current_accounts: List[Account] = []
        self.current_transaction_type: Optional[TransactionType] = None
        self.pin_attempts = 0
        
        # Logging and monitoring
        self.transaction_log: List[Dict] = []
        self.session_log: List[Dict] = []
        self.is_operational = True
        
        self._lock = threading.Lock()
        
        print(f"🏧 ATM {atm_id} initialized at {location}")
    
    def set_state(self, new_state: ATMState) -> None:
        """Set ATM state."""
        with self._lock:
            old_state = self.current_state
            self.current_state = new_state
            print(f"ATM state changed: {old_state.value} -> {new_state.value}")
    
    def get_current_handler(self) -> ATMStateHandler:
        """Get current state handler."""
        return self.state_handlers.get(self.current_state, IdleState())
    
    def insert_card(self, card_number: str) -> None:
        """Insert card into ATM."""
        if not self.is_operational:
            self.display_message("ATM is out of service.")
            return
        
        self.get_current_handler().handle_card_insert(self, card_number)
    
    def enter_pin(self, pin: str) -> None:
        """Enter PIN."""
        self.get_current_handler().handle_pin_entry(self, pin)
    
    def select_menu_option(self, selection: str) -> None:
        """Select menu option."""
        self.get_current_handler().handle_menu_selection(self, selection)
    
    def process_transaction(self, transaction_data: Dict) -> None:
        """Process transaction."""
        if not self.current_accounts:
            self.display_message("No accounts available.")
            return
        
        # Use first account for simplicity (in real system, user would select)
        account = self.current_accounts[0]
        
        strategy = self.transaction_strategies.get(self.current_transaction_type)
        if not strategy:
            self.display_message("Invalid transaction type.")
            return
        
        # Execute transaction
        result = strategy.execute(self, account, **transaction_data)
        
        # Log transaction
        self._log_transaction(result)
        
        # Display result
        self.display_message(result.message)
        
        if result.success:
            # Print receipt if requested
            if transaction_data.get('print_receipt', True):
                self.print_receipt(result.receipt_data)
            
            # Update card withdrawal tracking for withdrawals
            if (self.current_transaction_type == TransactionType.WITHDRAW and 
                self.current_card):
                self.current_card.record_withdrawal(result.amount)
        
        # Return to menu or eject card based on user choice
        if transaction_data.get('continue_session', True):
            self.set_state(ATMState.MENU_SELECTION)
            self.display_main_menu()
        else:
            self.eject_card()
    
    def eject_card(self) -> None:
        """Eject card and reset session."""
        with self._lock:
            if self.current_card_number:
                self.display_message("Please take your card. Thank you!")
                
                # Log session end
                self._log_session_end()
                
                # Reset session data
                self.current_card_number = None
                self.current_card = None
                self.current_accounts = []
                self.current_transaction_type = None
                self.pin_attempts = 0
                
                self.set_state(ATMState.IDLE)
                self.display_message("Welcome! Please insert your card.")
    
    def display_message(self, message: str) -> None:
        """Display message on ATM screen."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] 🏧 {message}")
    
    def display_main_menu(self) -> None:
        """Display main menu options."""
        menu = """
        === MAIN MENU ===
        1. Withdraw Cash
        2. Deposit Cash
        3. Transfer Funds
        4. Check Balance
        5. Exit
        
        Please select an option:
        """
        self.display_message(menu)
    
    def print_receipt(self, receipt_data: Dict[str, Any]) -> None:
        """Print transaction receipt."""
        if not receipt_data:
            return
        
        receipt = f"""
        ================================
        {receipt_data.get('transaction_type', 'Transaction').upper()} RECEIPT
        ================================
        ATM ID: {self.atm_id}
        Location: {self.location}
        Date/Time: {receipt_data.get('timestamp', 'N/A')}
        
        Account: {receipt_data.get('account_number', 'N/A')}
        Amount: ${receipt_data.get('amount', 0):.2f}
        Balance: ${receipt_data.get('balance', 0):.2f}
        
        Thank you for using our ATM!
        ================================
        """
        
        print("🧾 Receipt printed:")
        print(receipt)
    
    def _log_transaction(self, result: TransactionResult) -> None:
        """Log transaction details."""
        log_entry = {
            'transaction_id': result.transaction_id,
            'atm_id': self.atm_id,
            'card_number': f"****{self.current_card_number[-4:]}" if self.current_card_number else None,
            'transaction_type': self.current_transaction_type.value if self.current_transaction_type else None,
            'amount': result.amount,
            'success': result.success,
            'message': result.message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.transaction_log.append(log_entry)
    
    def _log_session_end(self) -> None:
        """Log session end."""
        if self.current_card_number:
            session_entry = {
                'session_id': str(uuid.uuid4()),
                'atm_id': self.atm_id,
                'card_number': f"****{self.current_card_number[-4:]}",
                'start_time': getattr(self, 'session_start_time', datetime.now()).isoformat(),
                'end_time': datetime.now().isoformat(),
                'transactions_count': len([t for t in self.transaction_log 
                                         if t.get('card_number') == f"****{self.current_card_number[-4:]}"])
            }
            
            self.session_log.append(session_entry)
    
    def get_atm_status(self) -> Dict[str, Any]:
        """Get ATM status information."""
        return {
            'atm_id': self.atm_id,
            'location': self.location,
            'current_state': self.current_state.value,
            'is_operational': self.is_operational,
            'cash_status': self.cash_dispenser.get_cash_status(),
            'transactions_today': len([t for t in self.transaction_log 
                                     if t['timestamp'].startswith(date.today().isoformat())]),
            'total_transactions': len(self.transaction_log),
            'total_sessions': len(self.session_log)
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_atm_system():
    """Demonstrate the ATM system."""
    print("=== ATM MACHINE DESIGN DEMONSTRATION ===\n")
    
    # Initialize bank network
    bank_network = BankNetwork()
    
    print("1. BANK NETWORK SETUP:")
    
    # Create accounts
    accounts_data = [
        ("ACC001", AccountType.CHECKING, 1500.0, "Alice Johnson"),
        ("ACC002", AccountType.SAVINGS, 5000.0, "Alice Johnson"),
        ("ACC003", AccountType.CHECKING, 750.0, "Bob Smith"),
        ("ACC004", AccountType.CREDIT, -200.0, "Charlie Brown")
    ]
    
    for acc_num, acc_type, balance, name in accounts_data:
        account = Account(acc_num, acc_type, balance, name)
        bank_network.add_account(account)
        print(f"   ✓ Account created: {name} - {acc_type.value} (${balance:.2f})")
    
    # Create cards
    cards_data = [
        ("1234567890123456", CardType.DEBIT, "1234", ["ACC001", "ACC002"]),
        ("2345678901234567", CardType.DEBIT, "5678", ["ACC003"]),
        ("3456789012345678", CardType.CREDIT, "9999", ["ACC004"])
    ]
    
    for card_num, card_type, pin, account_nums in cards_data:
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()
        expiry_date = date(2025, 12, 31)
        card = Card(card_num, card_type, pin_hash, account_nums, expiry_date)
        bank_network.add_card(card)
        print(f"   ✓ Card created: ****{card_num[-4:]} - {card_type.value}")
    
    print()
    
    # Initialize ATM
    print("2. ATM INITIALIZATION:")
    atm = ATMMachine("ATM001", "Main Street Branch", bank_network)
    
    # Add cash to dispenser
    initial_cash = {100: 20, 50: 30, 20: 50, 10: 40, 5: 20, 1: 50}
    atm.cash_dispenser.add_cash(initial_cash)
    
    cash_status = atm.cash_dispenser.get_cash_status()
    print(f"   ✓ Cash loaded: ${cash_status['total_cash']}")
    print()
    
    # Simulate ATM operations
    print("3. ATM OPERATIONS SIMULATION:")
    
    # Session 1: Alice's transactions
    print("\n   === Alice's Session ===")
    
    # Insert card
    atm.insert_card("1234567890123456")
    
    # Enter correct PIN
    atm.enter_pin("1234")
    
    # Check balance
    atm.select_menu_option("balance")
    
    # Withdraw cash
    atm.select_menu_option("withdraw")
    atm.process_transaction({'amount': 200.0, 'continue_session': True})
    
    # Make a deposit
    atm.select_menu_option("deposit")
    atm.process_transaction({'amount': 100.0, 'continue_session': True})
    
    # Transfer funds
    atm.select_menu_option("transfer")
    atm.process_transaction({
        'amount': 50.0, 
        'to_account_number': 'ACC003',
        'continue_session': False
    })
    
    print("\n   === Bob's Session ===")
    
    # Session 2: Bob's transactions
    atm.insert_card("2345678901234567")
    
    # Wrong PIN first
    atm.enter_pin("0000")
    
    # Correct PIN
    atm.enter_pin("5678")
    
    # Check balance (should show transfer from Alice)
    atm.select_menu_option("balance")
    
    # Try to withdraw more than available
    atm.select_menu_option("withdraw")
    atm.process_transaction({'amount': 1000.0, 'continue_session': True})
    
    # Successful withdrawal
    atm.process_transaction({'amount': 100.0, 'continue_session': False})
    
    print("\n   === Charlie's Session (Credit Account) ===")
    
    # Session 3: Charlie's credit account
    atm.insert_card("3456789012345678")
    atm.enter_pin("9999")
    
    # Check balance
    atm.select_menu_option("balance")
    
    # Withdraw from credit (should work within credit limit)
    atm.select_menu_option("withdraw")
    atm.process_transaction({'amount': 300.0, 'continue_session': False})
    
    print()
    
    # Test error scenarios
    print("4. ERROR SCENARIO TESTING:")
    
    print("\n   === Invalid PIN Attempts ===")
    atm.insert_card("1234567890123456")
    atm.enter_pin("0000")  # Wrong PIN 1
    atm.enter_pin("1111")  # Wrong PIN 2
    atm.enter_pin("2222")  # Wrong PIN 3 - should block
    
    print("\n   === Insufficient Cash Test ===")
    # Try to withdraw large amount to test cash dispenser limits
    atm.insert_card("1234567890123456")
    atm.enter_pin("1234")
    atm.select_menu_option("withdraw")
    atm.process_transaction({'amount': 10000.0, 'continue_session': False})
    
    print()
    
    # Show final status
    print("5. FINAL STATUS REPORTS:")
    
    # ATM status
    atm_status = atm.get_atm_status()
    print(f"\n   ATM Status:")
    print(f"   - State: {atm_status['current_state']}")
    print(f"   - Operational: {atm_status['is_operational']}")
    print(f"   - Cash remaining: ${atm_status['cash_status']['total_cash']}")
    print(f"   - Transactions today: {atm_status['transactions_today']}")
    print(f"   - Total sessions: {atm_status['total_sessions']}")
    
    # Account balances after all transactions
    print(f"\n   Final Account Balances:")
    for acc_num in ["ACC001", "ACC002", "ACC003", "ACC004"]:
        account = bank_network.get_account(acc_num)
        if account:
            print(f"   - {acc_num} ({account.account_type.value}): ${account.balance:.2f}")
    
    # Transaction log summary
    print(f"\n   Transaction Log Summary:")
    successful_transactions = [t for t in atm.transaction_log if t['success']]
    failed_transactions = [t for t in atm.transaction_log if not t['success']]
    
    print(f"   - Successful: {len(successful_transactions)}")
    print(f"   - Failed: {len(failed_transactions)}")
    
    if failed_transactions:
        print(f"   - Failed transaction reasons:")
        for transaction in failed_transactions:
            print(f"     • {transaction['message']}")
    
    # Cash dispenser final status
    final_cash_status = atm.cash_dispenser.get_cash_status()
    print(f"\n   Cash Dispenser Status:")
    print(f"   - Total cash: ${final_cash_status['total_cash']}")
    print(f"   - Low cash warning: {final_cash_status['low_cash_warning']}")
    print(f"   - Denominations:")
    for value, info in final_cash_status['denominations'].items():
        print(f"     ${value}: {info['count']} bills (${info['total_value']})")
    
    print()
    print("=== ATM MACHINE DESIGN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_atm_system()
