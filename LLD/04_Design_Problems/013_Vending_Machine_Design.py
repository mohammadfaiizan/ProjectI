"""
VENDING MACHINE DESIGN - Complete System Design
===============================================

Problem Statement:
Design a comprehensive vending machine system that handles:
- Product inventory management with different categories
- Multiple payment methods (cash, card, mobile payments)
- Change dispensing and cash management
- Product selection and dispensing mechanisms
- Temperature control for different product types
- User interface and display management
- Maintenance and restocking operations
- Sales reporting and analytics
- Remote monitoring and management
- Multi-location vending machine network

Requirements:
- Support various product types (snacks, beverages, hot drinks)
- Handle multiple currencies and payment methods
- Manage inventory levels and expiration dates
- Provide real-time status monitoring
- Support promotional pricing and discounts
- Handle maintenance scheduling and alerts
- Generate comprehensive sales reports
- Support remote configuration updates
- Implement security features for cash and products
- Provide user-friendly interface with accessibility features

Design Patterns Used:
- State: Machine operational states
- Strategy: Payment processing strategies
- Observer: Inventory and status monitoring
- Command: Transaction operations
- Factory: Product and payment method creation
- Singleton: Machine configuration manager
- Template Method: Transaction processing flow
- Decorator: Product pricing with promotions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import decimal
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MachineState(Enum):
    IDLE = "idle"
    PRODUCT_SELECTION = "product_selection"
    PAYMENT_PROCESSING = "payment_processing"
    DISPENSING = "dispensing"
    CHANGE_DISPENSING = "change_dispensing"
    OUT_OF_ORDER = "out_of_order"
    MAINTENANCE = "maintenance"


class ProductCategory(Enum):
    SNACKS = "snacks"
    BEVERAGES = "beverages"
    HOT_DRINKS = "hot_drinks"
    ICE_CREAM = "ice_cream"
    HEALTHY = "healthy"


class PaymentMethod(Enum):
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOBILE_PAYMENT = "mobile_payment"
    CONTACTLESS = "contactless"


class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


@dataclass
class Money:
    """Money representation with currency."""
    amount: decimal.Decimal
    currency: Currency = Currency.USD
    
    def __post_init__(self):
        self.amount = decimal.Decimal(str(self.amount))
    
    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, multiplier: float) -> 'Money':
        return Money(self.amount * decimal.Decimal(str(multiplier)), self.currency)
    
    def __eq__(self, other: 'Money') -> bool:
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other: 'Money') -> bool:
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount < other.amount
    
    def __str__(self) -> str:
        return f"{self.currency.value} {self.amount:.2f}"


@dataclass
class Product:
    """Product information."""
    product_id: str
    name: str
    category: ProductCategory
    base_price: Money
    size: str
    weight: float  # in grams
    calories: int
    ingredients: List[str]
    allergens: List[str]
    expiration_days: int  # days from manufacture
    temperature_range: Tuple[float, float]  # min, max celsius
    barcode: str
    manufacturer: str
    
    def __post_init__(self):
        if not self.product_id:
            self.product_id = str(uuid.uuid4())


@dataclass
class InventoryItem:
    """Inventory item with stock information."""
    product: Product
    slot_id: str
    quantity: int
    manufacture_date: datetime
    restock_date: datetime
    minimum_stock_level: int = 5
    maximum_capacity: int = 20
    
    @property
    def expiration_date(self) -> datetime:
        return self.manufacture_date + timedelta(days=self.product.expiration_days)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expiration_date
    
    @property
    def days_until_expiration(self) -> int:
        delta = self.expiration_date - datetime.now()
        return max(0, delta.days)
    
    @property
    def needs_restocking(self) -> bool:
        return self.quantity <= self.minimum_stock_level


@dataclass
class CashDenomination:
    """Cash denomination information."""
    value: Money
    count: int
    denomination_type: str  # "coin" or "bill"
    
    @property
    def total_value(self) -> Money:
        return self.value * self.count


@dataclass
class Transaction:
    """Vending machine transaction."""
    transaction_id: str
    machine_id: str
    product: Product
    quantity: int
    total_amount: Money
    payment_method: PaymentMethod
    payment_details: Dict[str, Any]
    status: TransactionStatus
    timestamp: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    change_given: Optional[Money] = None
    
    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = str(uuid.uuid4())


# ============================================================================
# PAYMENT STRATEGIES
# ============================================================================

class PaymentProcessor(ABC):
    """Abstract payment processor."""
    
    @abstractmethod
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Process payment and return success status and message."""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: Money) -> Tuple[bool, str]:
        """Process refund and return success status and message."""
        pass
    
    @abstractmethod
    def get_payment_method(self) -> PaymentMethod:
        """Get payment method type."""
        pass


class CashProcessor(PaymentProcessor):
    """Cash payment processor."""
    
    def __init__(self, cash_manager: 'CashManager'):
        self.cash_manager = cash_manager
    
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Process cash payment."""
        try:
            inserted_cash = payment_details.get('inserted_cash', [])
            total_inserted = Money(decimal.Decimal('0'), amount.currency)
            
            # Calculate total inserted cash
            for denomination in inserted_cash:
                coin_value = Money(denomination['value'], amount.currency)
                coin_count = denomination['count']
                total_inserted = total_inserted + (coin_value * coin_count)
            
            if total_inserted < amount:
                return False, f"Insufficient cash. Need {amount}, inserted {total_inserted}"
            
            # Add cash to machine
            for denomination in inserted_cash:
                self.cash_manager.add_cash(
                    Money(denomination['value'], amount.currency),
                    denomination['count'],
                    denomination['type']
                )
            
            return True, "Cash payment successful"
            
        except Exception as e:
            return False, f"Cash processing error: {str(e)}"
    
    def refund_payment(self, transaction_id: str, amount: Money) -> Tuple[bool, str]:
        """Refund cash payment."""
        return self.cash_manager.dispense_change(amount)
    
    def get_payment_method(self) -> PaymentMethod:
        return PaymentMethod.CASH


class CardProcessor(PaymentProcessor):
    """Card payment processor."""
    
    def __init__(self, payment_gateway: str):
        self.payment_gateway = payment_gateway
        self.processed_transactions: Dict[str, Dict[str, Any]] = {}
    
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Process card payment."""
        try:
            card_number = payment_details.get('card_number', '')
            cvv = payment_details.get('cvv', '')
            expiry = payment_details.get('expiry', '')
            
            # Simulate card validation
            if not self._validate_card(card_number, cvv, expiry):
                return False, "Invalid card details"
            
            # Simulate payment processing
            if self._simulate_payment_gateway(amount, card_number):
                # Store transaction for potential refund
                transaction_id = str(uuid.uuid4())
                self.processed_transactions[transaction_id] = {
                    'amount': amount,
                    'card_number': card_number[-4:],  # Store last 4 digits only
                    'timestamp': datetime.now()
                }
                return True, f"Card payment successful. Transaction ID: {transaction_id}"
            else:
                return False, "Payment declined by bank"
                
        except Exception as e:
            return False, f"Card processing error: {str(e)}"
    
    def refund_payment(self, transaction_id: str, amount: Money) -> Tuple[bool, str]:
        """Refund card payment."""
        if transaction_id in self.processed_transactions:
            # Simulate refund processing
            return True, f"Refund of {amount} processed to card ending in {self.processed_transactions[transaction_id]['card_number']}"
        return False, "Transaction not found for refund"
    
    def _validate_card(self, card_number: str, cvv: str, expiry: str) -> bool:
        """Validate card details (simplified)."""
        return (len(card_number) >= 13 and 
                len(cvv) >= 3 and 
                len(expiry) == 5 and 
                '/' in expiry)
    
    def _simulate_payment_gateway(self, amount: Money, card_number: str) -> bool:
        """Simulate payment gateway response."""
        # Simulate 95% success rate
        import random
        return random.random() < 0.95
    
    def get_payment_method(self) -> PaymentMethod:
        return PaymentMethod.CREDIT_CARD


class MobilePaymentProcessor(PaymentProcessor):
    """Mobile payment processor."""
    
    def __init__(self):
        self.processed_payments: Dict[str, Dict[str, Any]] = {}
    
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Process mobile payment."""
        try:
            payment_app = payment_details.get('app', '')
            user_id = payment_details.get('user_id', '')
            auth_token = payment_details.get('auth_token', '')
            
            if not all([payment_app, user_id, auth_token]):
                return False, "Missing mobile payment credentials"
            
            # Simulate mobile payment processing
            if self._validate_mobile_payment(payment_app, user_id, auth_token):
                payment_id = str(uuid.uuid4())
                self.processed_payments[payment_id] = {
                    'amount': amount,
                    'app': payment_app,
                    'user_id': user_id,
                    'timestamp': datetime.now()
                }
                return True, f"Mobile payment successful via {payment_app}"
            else:
                return False, "Mobile payment authentication failed"
                
        except Exception as e:
            return False, f"Mobile payment error: {str(e)}"
    
    def refund_payment(self, transaction_id: str, amount: Money) -> Tuple[bool, str]:
        """Refund mobile payment."""
        if transaction_id in self.processed_payments:
            payment_info = self.processed_payments[transaction_id]
            return True, f"Refund of {amount} processed via {payment_info['app']}"
        return False, "Mobile payment transaction not found"
    
    def _validate_mobile_payment(self, app: str, user_id: str, auth_token: str) -> bool:
        """Validate mobile payment credentials."""
        # Simulate validation
        return len(auth_token) > 10 and user_id.isalnum()
    
    def get_payment_method(self) -> PaymentMethod:
        return PaymentMethod.MOBILE_PAYMENT


# ============================================================================
# CASH MANAGEMENT
# ============================================================================

class CashManager:
    """Cash management system."""
    
    def __init__(self, currency: Currency = Currency.USD):
        self.currency = currency
        self.cash_inventory: Dict[str, CashDenomination] = {}
        self._initialize_denominations()
        
        # Cash limits
        self.maximum_cash_capacity = Money(decimal.Decimal('1000'), currency)
        self.minimum_change_reserve = Money(decimal.Decimal('50'), currency)
        
        self._lock = threading.Lock()
    
    def _initialize_denominations(self) -> None:
        """Initialize cash denominations."""
        if self.currency == Currency.USD:
            denominations = [
                # Bills
                (Money(decimal.Decimal('20'), self.currency), 10, "bill"),
                (Money(decimal.Decimal('10'), self.currency), 10, "bill"),
                (Money(decimal.Decimal('5'), self.currency), 20, "bill"),
                (Money(decimal.Decimal('1'), self.currency), 50, "bill"),
                # Coins
                (Money(decimal.Decimal('0.25'), self.currency), 40, "coin"),
                (Money(decimal.Decimal('0.10'), self.currency), 50, "coin"),
                (Money(decimal.Decimal('0.05'), self.currency), 60, "coin"),
                (Money(decimal.Decimal('0.01'), self.currency), 100, "coin"),
            ]
        else:
            # Default denominations for other currencies
            denominations = [
                (Money(decimal.Decimal('10'), self.currency), 10, "bill"),
                (Money(decimal.Decimal('5'), self.currency), 20, "bill"),
                (Money(decimal.Decimal('1'), self.currency), 50, "coin"),
                (Money(decimal.Decimal('0.50'), self.currency), 40, "coin"),
                (Money(decimal.Decimal('0.20'), self.currency), 50, "coin"),
                (Money(decimal.Decimal('0.10'), self.currency), 60, "coin"),
                (Money(decimal.Decimal('0.05'), self.currency), 80, "coin"),
                (Money(decimal.Decimal('0.01'), self.currency), 100, "coin"),
            ]
        
        for value, count, denom_type in denominations:
            key = f"{value.amount}_{denom_type}"
            self.cash_inventory[key] = CashDenomination(value, count, denom_type)
    
    def add_cash(self, denomination: Money, count: int, denom_type: str) -> bool:
        """Add cash to inventory."""
        with self._lock:
            key = f"{denomination.amount}_{denom_type}"
            
            if key in self.cash_inventory:
                self.cash_inventory[key].count += count
            else:
                self.cash_inventory[key] = CashDenomination(denomination, count, denom_type)
            
            return True
    
    def get_total_cash(self) -> Money:
        """Get total cash in machine."""
        total = Money(decimal.Decimal('0'), self.currency)
        
        for denomination in self.cash_inventory.values():
            total = total + denomination.total_value
        
        return total
    
    def can_make_change(self, amount: Money) -> bool:
        """Check if machine can make change for given amount."""
        return len(self._calculate_change(amount)) > 0
    
    def dispense_change(self, amount: Money) -> Tuple[bool, str]:
        """Dispense change and return success status."""
        with self._lock:
            if amount.amount <= 0:
                return True, "No change required"
            
            change_breakdown = self._calculate_change(amount)
            
            if not change_breakdown:
                return False, f"Cannot make exact change for {amount}"
            
            # Dispense the change
            for key, count in change_breakdown.items():
                self.cash_inventory[key].count -= count
            
            change_description = self._format_change_description(change_breakdown)
            return True, f"Change dispensed: {change_description}"
    
    def _calculate_change(self, amount: Money) -> Dict[str, int]:
        """Calculate optimal change breakdown."""
        remaining = amount.amount
        change_breakdown = {}
        
        # Sort denominations by value (descending)
        sorted_denominations = sorted(
            self.cash_inventory.items(),
            key=lambda x: x[1].value.amount,
            reverse=True
        )
        
        for key, denomination in sorted_denominations:
            if remaining <= 0:
                break
            
            if denomination.count > 0 and denomination.value.amount <= remaining:
                max_coins = min(
                    denomination.count,
                    int(remaining // denomination.value.amount)
                )
                
                if max_coins > 0:
                    change_breakdown[key] = max_coins
                    remaining -= denomination.value.amount * max_coins
        
        # Check if exact change was made
        if remaining > decimal.Decimal('0.001'):  # Allow for small rounding errors
            return {}
        
        return change_breakdown
    
    def _format_change_description(self, change_breakdown: Dict[str, int]) -> str:
        """Format change breakdown for display."""
        descriptions = []
        
        for key, count in change_breakdown.items():
            denomination = self.cash_inventory[key]
            descriptions.append(f"{count}x {denomination.value}")
        
        return ", ".join(descriptions)
    
    def get_cash_status(self) -> Dict[str, Any]:
        """Get cash inventory status."""
        status = {
            'total_cash': str(self.get_total_cash()),
            'denominations': {},
            'can_make_change': True,
            'needs_refill': False
        }
        
        for key, denomination in self.cash_inventory.items():
            status['denominations'][key] = {
                'value': str(denomination.value),
                'count': denomination.count,
                'total_value': str(denomination.total_value),
                'type': denomination.denomination_type
            }
            
            # Check if any denomination is running low
            if denomination.count < 5:
                status['needs_refill'] = True
        
        # Test if we can make common change amounts
        test_amounts = [Money(decimal.Decimal('0.25'), self.currency),
                       Money(decimal.Decimal('0.75'), self.currency),
                       Money(decimal.Decimal('1.00'), self.currency)]
        
        for test_amount in test_amounts:
            if not self.can_make_change(test_amount):
                status['can_make_change'] = False
                break
        
        return status


# ============================================================================
# PRODUCT MANAGEMENT
# ============================================================================

class ProductPricing:
    """Product pricing with promotions."""
    
    def __init__(self):
        self.base_prices: Dict[str, Money] = {}
        self.promotions: Dict[str, Dict[str, Any]] = {}
    
    def set_base_price(self, product_id: str, price: Money) -> None:
        """Set base price for product."""
        self.base_prices[product_id] = price
    
    def add_promotion(self, product_id: str, promotion_type: str, 
                     discount_percent: float = 0, fixed_discount: Money = None,
                     start_date: datetime = None, end_date: datetime = None) -> None:
        """Add promotion for product."""
        self.promotions[product_id] = {
            'type': promotion_type,
            'discount_percent': discount_percent,
            'fixed_discount': fixed_discount,
            'start_date': start_date or datetime.now(),
            'end_date': end_date or datetime.now() + timedelta(days=30),
            'active': True
        }
    
    def get_current_price(self, product_id: str) -> Money:
        """Get current price including promotions."""
        base_price = self.base_prices.get(product_id)
        if not base_price:
            return Money(decimal.Decimal('0'))
        
        promotion = self.promotions.get(product_id)
        if not promotion or not self._is_promotion_active(promotion):
            return base_price
        
        # Apply promotion
        if promotion['discount_percent'] > 0:
            discount_multiplier = 1 - (promotion['discount_percent'] / 100)
            return base_price * discount_multiplier
        elif promotion['fixed_discount']:
            discounted_price = base_price - promotion['fixed_discount']
            return Money(max(decimal.Decimal('0'), discounted_price.amount), base_price.currency)
        
        return base_price
    
    def _is_promotion_active(self, promotion: Dict[str, Any]) -> bool:
        """Check if promotion is currently active."""
        now = datetime.now()
        return (promotion['active'] and 
                promotion['start_date'] <= now <= promotion['end_date'])


class InventoryManager:
    """Inventory management system."""
    
    def __init__(self):
        self.inventory: Dict[str, InventoryItem] = {}  # slot_id -> InventoryItem
        self.products: Dict[str, Product] = {}  # product_id -> Product
        self.pricing = ProductPricing()
        self._lock = threading.Lock()
    
    def add_product(self, product: Product) -> None:
        """Add product to catalog."""
        self.products[product.product_id] = product
        self.pricing.set_base_price(product.product_id, product.base_price)
    
    def stock_product(self, slot_id: str, product_id: str, quantity: int,
                     manufacture_date: datetime = None) -> bool:
        """Stock product in specific slot."""
        with self._lock:
            if product_id not in self.products:
                return False
            
            product = self.products[product_id]
            manufacture_date = manufacture_date or datetime.now()
            
            if slot_id in self.inventory:
                # Add to existing stock
                self.inventory[slot_id].quantity += quantity
                # Update manufacture date if newer
                if manufacture_date > self.inventory[slot_id].manufacture_date:
                    self.inventory[slot_id].manufacture_date = manufacture_date
            else:
                # Create new inventory item
                self.inventory[slot_id] = InventoryItem(
                    product=product,
                    slot_id=slot_id,
                    quantity=quantity,
                    manufacture_date=manufacture_date,
                    restock_date=datetime.now()
                )
            
            return True
    
    def dispense_product(self, slot_id: str, quantity: int = 1) -> Tuple[bool, Optional[Product]]:
        """Dispense product from slot."""
        with self._lock:
            if slot_id not in self.inventory:
                return False, None
            
            inventory_item = self.inventory[slot_id]
            
            if inventory_item.quantity < quantity:
                return False, None
            
            if inventory_item.is_expired:
                return False, None
            
            inventory_item.quantity -= quantity
            return True, inventory_item.product
    
    def get_product_availability(self, product_id: str) -> List[str]:
        """Get available slots for product."""
        available_slots = []
        
        for slot_id, inventory_item in self.inventory.items():
            if (inventory_item.product.product_id == product_id and
                inventory_item.quantity > 0 and
                not inventory_item.is_expired):
                available_slots.append(slot_id)
        
        return available_slots
    
    def get_slot_info(self, slot_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific slot."""
        if slot_id not in self.inventory:
            return None
        
        inventory_item = self.inventory[slot_id]
        current_price = self.pricing.get_current_price(inventory_item.product.product_id)
        
        return {
            'slot_id': slot_id,
            'product': {
                'id': inventory_item.product.product_id,
                'name': inventory_item.product.name,
                'category': inventory_item.product.category.value,
                'size': inventory_item.product.size,
                'calories': inventory_item.product.calories
            },
            'quantity': inventory_item.quantity,
            'price': str(current_price),
            'available': inventory_item.quantity > 0 and not inventory_item.is_expired,
            'expires_in_days': inventory_item.days_until_expiration,
            'needs_restocking': inventory_item.needs_restocking
        }
    
    def get_inventory_status(self) -> Dict[str, Any]:
        """Get complete inventory status."""
        status = {
            'total_slots': len(self.inventory),
            'occupied_slots': 0,
            'empty_slots': 0,
            'expired_products': 0,
            'low_stock_slots': 0,
            'slots': {}
        }
        
        for slot_id, inventory_item in self.inventory.items():
            slot_info = self.get_slot_info(slot_id)
            status['slots'][slot_id] = slot_info
            
            if inventory_item.quantity > 0:
                status['occupied_slots'] += 1
            else:
                status['empty_slots'] += 1
            
            if inventory_item.is_expired:
                status['expired_products'] += 1
            
            if inventory_item.needs_restocking:
                status['low_stock_slots'] += 1
        
        return status


# ============================================================================
# VENDING MACHINE CORE
# ============================================================================

class VendingMachine:
    """Main vending machine system."""
    
    def __init__(self, machine_id: str, location: str, currency: Currency = Currency.USD):
        self.machine_id = machine_id
        self.location = location
        self.currency = currency
        
        # Core components
        self.state = MachineState.IDLE
        self.inventory_manager = InventoryManager()
        self.cash_manager = CashManager(currency)
        
        # Payment processors
        self.payment_processors: Dict[PaymentMethod, PaymentProcessor] = {
            PaymentMethod.CASH: CashProcessor(self.cash_manager),
            PaymentMethod.CREDIT_CARD: CardProcessor("payment_gateway"),
            PaymentMethod.MOBILE_PAYMENT: MobilePaymentProcessor()
        }
        
        # Transaction management
        self.current_transaction: Optional[Transaction] = None
        self.transaction_history: List[Transaction] = []
        
        # Machine configuration
        self.temperature = 4.0  # Celsius
        self.target_temperature = 4.0
        self.max_transaction_amount = Money(decimal.Decimal('50'), currency)
        
        # Status tracking
        self.last_maintenance = datetime.now()
        self.total_sales = Money(decimal.Decimal('0'), currency)
        self.error_log: List[Dict[str, Any]] = []
        
        # Threading
        self._lock = threading.Lock()
        
        print(f"🏪 Vending Machine initialized: {machine_id} at {location}")
    
    def select_product(self, slot_id: str) -> Tuple[bool, str]:
        """Select product for purchase."""
        with self._lock:
            if self.state != MachineState.IDLE:
                return False, f"Machine is {self.state.value}, cannot select product"
            
            slot_info = self.inventory_manager.get_slot_info(slot_id)
            if not slot_info:
                return False, f"Invalid slot: {slot_id}"
            
            if not slot_info['available']:
                return False, f"Product not available in slot {slot_id}"
            
            # Create transaction
            product = self.inventory_manager.inventory[slot_id].product
            price = self.inventory_manager.pricing.get_current_price(product.product_id)
            
            self.current_transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                machine_id=self.machine_id,
                product=product,
                quantity=1,
                total_amount=price,
                payment_method=PaymentMethod.CASH,  # Default, will be updated
                payment_details={},
                status=TransactionStatus.PENDING
            )
            
            self.state = MachineState.PRODUCT_SELECTION
            
            return True, f"Selected {product.name} - Price: {price}"
    
    def process_payment(self, payment_method: PaymentMethod, 
                       payment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Process payment for selected product."""
        with self._lock:
            if not self.current_transaction:
                return False, "No product selected"
            
            if self.state != MachineState.PRODUCT_SELECTION:
                return False, f"Invalid state for payment: {self.state.value}"
            
            if payment_method not in self.payment_processors:
                return False, f"Payment method {payment_method.value} not supported"
            
            self.state = MachineState.PAYMENT_PROCESSING
            
            # Update transaction
            self.current_transaction.payment_method = payment_method
            self.current_transaction.payment_details = payment_details
            
            # Process payment
            processor = self.payment_processors[payment_method]
            success, message = processor.process_payment(
                self.current_transaction.total_amount,
                payment_details
            )
            
            if success:
                self.current_transaction.status = TransactionStatus.COMPLETED
                
                # Calculate change for cash payments
                if payment_method == PaymentMethod.CASH:
                    inserted_amount = self._calculate_inserted_cash(payment_details)
                    change_amount = inserted_amount - self.current_transaction.total_amount
                    
                    if change_amount.amount > 0:
                        change_success, change_message = self.cash_manager.dispense_change(change_amount)
                        if change_success:
                            self.current_transaction.change_given = change_amount
                            self.state = MachineState.CHANGE_DISPENSING
                        else:
                            # Refund if can't make change
                            self._refund_transaction()
                            return False, f"Cannot make change: {change_message}"
                
                # Proceed to dispensing
                return self._dispense_product()
            else:
                self.current_transaction.status = TransactionStatus.FAILED
                self.state = MachineState.IDLE
                self.current_transaction = None
                return False, f"Payment failed: {message}"
    
    def _calculate_inserted_cash(self, payment_details: Dict[str, Any]) -> Money:
        """Calculate total inserted cash."""
        inserted_cash = payment_details.get('inserted_cash', [])
        total = Money(decimal.Decimal('0'), self.currency)
        
        for denomination in inserted_cash:
            coin_value = Money(denomination['value'], self.currency)
            coin_count = denomination['count']
            total = total + (coin_value * coin_count)
        
        return total
    
    def _dispense_product(self) -> Tuple[bool, str]:
        """Dispense the purchased product."""
        if not self.current_transaction:
            return False, "No active transaction"
        
        self.state = MachineState.DISPENSING
        
        # Find slot with the product
        available_slots = self.inventory_manager.get_product_availability(
            self.current_transaction.product.product_id
        )
        
        if not available_slots:
            self._refund_transaction()
            return False, "Product no longer available"
        
        # Dispense from first available slot
        slot_id = available_slots[0]
        success, product = self.inventory_manager.dispense_product(slot_id, 1)
        
        if success:
            # Complete transaction
            self.current_transaction.completion_time = datetime.now()
            self.transaction_history.append(self.current_transaction)
            
            # Update sales total
            self.total_sales = self.total_sales + self.current_transaction.total_amount
            
            # Log successful transaction
            self._log_transaction(self.current_transaction)
            
            product_name = self.current_transaction.product.name
            change_info = ""
            if self.current_transaction.change_given:
                change_info = f" Change: {self.current_transaction.change_given}"
            
            # Reset state
            self.current_transaction = None
            self.state = MachineState.IDLE
            
            return True, f"Dispensed {product_name}.{change_info} Thank you!"
        else:
            self._refund_transaction()
            return False, "Product dispensing failed"
    
    def _refund_transaction(self) -> None:
        """Refund current transaction."""
        if not self.current_transaction:
            return
        
        processor = self.payment_processors[self.current_transaction.payment_method]
        processor.refund_payment(
            self.current_transaction.transaction_id,
            self.current_transaction.total_amount
        )
        
        self.current_transaction.status = TransactionStatus.REFUNDED
        self.transaction_history.append(self.current_transaction)
        
        self.current_transaction = None
        self.state = MachineState.IDLE
    
    def cancel_transaction(self) -> Tuple[bool, str]:
        """Cancel current transaction."""
        with self._lock:
            if not self.current_transaction:
                return False, "No active transaction to cancel"
            
            if self.current_transaction.status == TransactionStatus.COMPLETED:
                return False, "Cannot cancel completed transaction"
            
            self._refund_transaction()
            return True, "Transaction cancelled and refunded"
    
    def get_machine_status(self) -> Dict[str, Any]:
        """Get comprehensive machine status."""
        inventory_status = self.inventory_manager.get_inventory_status()
        cash_status = self.cash_manager.get_cash_status()
        
        return {
            'machine_id': self.machine_id,
            'location': self.location,
            'state': self.state.value,
            'temperature': self.temperature,
            'target_temperature': self.target_temperature,
            'currency': self.currency.value,
            'total_sales': str(self.total_sales),
            'transactions_today': len([t for t in self.transaction_history 
                                     if t.timestamp.date() == datetime.now().date()]),
            'inventory': inventory_status,
            'cash': cash_status,
            'current_transaction': {
                'active': self.current_transaction is not None,
                'product': self.current_transaction.product.name if self.current_transaction else None,
                'amount': str(self.current_transaction.total_amount) if self.current_transaction else None
            },
            'maintenance_required': self._needs_maintenance(),
            'last_maintenance': self.last_maintenance.isoformat(),
            'error_count': len(self.error_log)
        }
    
    def _needs_maintenance(self) -> bool:
        """Check if machine needs maintenance."""
        # Check if maintenance is overdue (30 days)
        if datetime.now() - self.last_maintenance > timedelta(days=30):
            return True
        
        # Check for critical issues
        inventory_status = self.inventory_manager.get_inventory_status()
        cash_status = self.cash_manager.get_cash_status()
        
        return (inventory_status['low_stock_slots'] > 5 or
                inventory_status['expired_products'] > 0 or
                not cash_status['can_make_change'] or
                len(self.error_log) > 10)
    
    def perform_maintenance(self, maintenance_type: str, notes: str = "") -> bool:
        """Perform maintenance on machine."""
        with self._lock:
            if self.state == MachineState.DISPENSING:
                return False  # Cannot perform maintenance during dispensing
            
            previous_state = self.state
            self.state = MachineState.MAINTENANCE
            
            # Perform maintenance tasks
            if maintenance_type == "restock":
                self._perform_restocking()
            elif maintenance_type == "cash_refill":
                self._refill_cash()
            elif maintenance_type == "cleaning":
                self._perform_cleaning()
            elif maintenance_type == "full":
                self._perform_restocking()
                self._refill_cash()
                self._perform_cleaning()
            
            # Update maintenance record
            self.last_maintenance = datetime.now()
            
            # Clear error log
            self.error_log.clear()
            
            # Return to previous state or idle
            self.state = previous_state if previous_state != MachineState.OUT_OF_ORDER else MachineState.IDLE
            
            print(f"Maintenance completed: {maintenance_type}")
            return True
    
    def _perform_restocking(self) -> None:
        """Perform restocking maintenance."""
        # This would interface with inventory management system
        # For demo, we'll simulate restocking low items
        for slot_id, inventory_item in self.inventory_manager.inventory.items():
            if inventory_item.needs_restocking:
                additional_stock = inventory_item.maximum_capacity - inventory_item.quantity
                inventory_item.quantity += additional_stock
                inventory_item.restock_date = datetime.now()
    
    def _refill_cash(self) -> None:
        """Refill cash denominations."""
        # Simulate cash refill
        for denomination in self.cash_manager.cash_inventory.values():
            if denomination.count < 20:
                denomination.count = 50  # Refill to standard level
    
    def _perform_cleaning(self) -> None:
        """Perform cleaning maintenance."""
        # Reset temperature to optimal
        self.temperature = self.target_temperature
    
    def _log_transaction(self, transaction: Transaction) -> None:
        """Log transaction for reporting."""
        print(f"💰 Transaction: {transaction.product.name} - {transaction.total_amount} "
              f"via {transaction.payment_method.value}")
    
    def get_sales_report(self, start_date: datetime = None, 
                        end_date: datetime = None) -> Dict[str, Any]:
        """Generate sales report."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filter transactions by date range
        filtered_transactions = [
            t for t in self.transaction_history
            if (start_date <= t.timestamp <= end_date and 
                t.status == TransactionStatus.COMPLETED)
        ]
        
        # Calculate metrics
        total_revenue = Money(decimal.Decimal('0'), self.currency)
        product_sales = defaultdict(int)
        payment_method_usage = defaultdict(int)
        
        for transaction in filtered_transactions:
            total_revenue = total_revenue + transaction.total_amount
            product_sales[transaction.product.name] += transaction.quantity
            payment_method_usage[transaction.payment_method.value] += 1
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_transactions': len(filtered_transactions),
                'total_revenue': str(total_revenue),
                'average_transaction': str(total_revenue * (1.0 / max(1, len(filtered_transactions))))
            },
            'product_sales': dict(product_sales),
            'payment_methods': dict(payment_method_usage),
            'top_selling_products': sorted(product_sales.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_vending_machine():
    """Demonstrate the vending machine system."""
    print("=== VENDING MACHINE SYSTEM DEMONSTRATION ===\n")
    
    # Create vending machine
    print("1. VENDING MACHINE CREATION:")
    
    machine = VendingMachine("VM001", "Office Building Lobby", Currency.USD)
    print(f"   ✓ Created machine {machine.machine_id} at {machine.location}")
    print()
    
    # Add products to catalog
    print("2. PRODUCT CATALOG SETUP:")
    
    products = [
        Product("P001", "Coca Cola", ProductCategory.BEVERAGES, 
               Money(decimal.Decimal('1.50'), Currency.USD), "12oz", 355, 140, 
               ["carbonated water", "sugar", "caffeine"], [], 365, (2, 8), 
               "123456789012", "Coca Cola Co"),
        Product("P002", "Snickers Bar", ProductCategory.SNACKS, 
               Money(decimal.Decimal('1.25'), Currency.USD), "52g", 52, 250, 
               ["peanuts", "chocolate", "caramel"], ["peanuts"], 180, (15, 25), 
               "123456789013", "Mars Inc"),
        Product("P003", "Hot Coffee", ProductCategory.HOT_DRINKS, 
               Money(decimal.Decimal('2.00'), Currency.USD), "8oz", 240, 5, 
               ["coffee beans", "water"], [], 1, (60, 85), 
               "123456789014", "Coffee Co"),
        Product("P004", "Apple", ProductCategory.HEALTHY, 
               Money(decimal.Decimal('0.75'), Currency.USD), "medium", 180, 80, 
               ["apple"], [], 14, (2, 8), 
               "123456789015", "Fresh Farms"),
        Product("P005", "Ice Cream Sandwich", ProductCategory.ICE_CREAM, 
               Money(decimal.Decimal('2.50'), Currency.USD), "85g", 85, 200, 
               ["milk", "sugar", "cookies"], ["dairy"], 90, (-18, -10), 
               "123456789016", "Ice Cream Co")
    ]
    
    for product in products:
        machine.inventory_manager.add_product(product)
        print(f"   ✓ Added {product.name} ({product.category.value}) - {product.base_price}")
    
    print()
    
    # Stock products in slots
    print("3. INVENTORY STOCKING:")
    
    stocking_plan = [
        ("A1", "P001", 15),  # Coca Cola
        ("A2", "P001", 10),  # More Coca Cola
        ("B1", "P002", 20),  # Snickers
        ("B2", "P002", 15),  # More Snickers
        ("C1", "P003", 8),   # Hot Coffee
        ("D1", "P004", 12),  # Apple
        ("E1", "P005", 6),   # Ice Cream
    ]
    
    for slot_id, product_id, quantity in stocking_plan:
        success = machine.inventory_manager.stock_product(slot_id, product_id, quantity)
        product_name = machine.inventory_manager.products[product_id].name
        print(f"   ✓ Stocked {quantity}x {product_name} in slot {slot_id}")
    
    print()
    
    # Add promotions
    print("4. PROMOTIONS SETUP:")
    
    # 10% off Snickers
    machine.inventory_manager.pricing.add_promotion(
        "P002", "percentage_discount", discount_percent=10,
        end_date=datetime.now() + timedelta(days=7)
    )
    print("   ✓ Added 10% discount on Snickers Bar (7 days)")
    
    # $0.25 off Apple
    machine.inventory_manager.pricing.add_promotion(
        "P004", "fixed_discount", 
        fixed_discount=Money(decimal.Decimal('0.25'), Currency.USD),
        end_date=datetime.now() + timedelta(days=14)
    )
    print("   ✓ Added $0.25 discount on Apple (14 days)")
    
    print()
    
    # Show initial machine status
    print("5. INITIAL MACHINE STATUS:")
    
    status = machine.get_machine_status()
    print(f"   Machine State: {status['state']}")
    print(f"   Temperature: {status['temperature']}°C")
    print(f"   Total Sales: {status['total_sales']}")
    print(f"   Occupied Slots: {status['inventory']['occupied_slots']}")
    print(f"   Total Cash: {status['cash']['total_cash']}")
    print()
    
    # Test product selection and cash payment
    print("6. CASH TRANSACTION TEST:")
    
    # Select product
    success, message = machine.select_product("B1")  # Snickers with discount
    print(f"   Product Selection: {message}")
    
    if success:
        # Prepare cash payment (insert $2.00 for $1.13 item)
        cash_payment = {
            'inserted_cash': [
                {'value': 1.00, 'count': 2, 'type': 'bill'},  # $2.00
            ]
        }
        
        success, message = machine.process_payment(PaymentMethod.CASH, cash_payment)
        print(f"   Cash Payment: {message}")
    
    print()
    
    # Test card payment
    print("7. CARD TRANSACTION TEST:")
    
    # Select another product
    success, message = machine.select_product("A1")  # Coca Cola
    print(f"   Product Selection: {message}")
    
    if success:
        # Prepare card payment
        card_payment = {
            'card_number': '4111111111111111',
            'cvv': '123',
            'expiry': '12/25'
        }
        
        success, message = machine.process_payment(PaymentMethod.CREDIT_CARD, card_payment)
        print(f"   Card Payment: {message}")
    
    print()
    
    # Test mobile payment
    print("8. MOBILE PAYMENT TEST:")
    
    # Select product
    success, message = machine.select_product("C1")  # Hot Coffee
    print(f"   Product Selection: {message}")
    
    if success:
        # Prepare mobile payment
        mobile_payment = {
            'app': 'Apple Pay',
            'user_id': 'user123',
            'auth_token': 'abc123def456ghi789'
        }
        
        success, message = machine.process_payment(PaymentMethod.MOBILE_PAYMENT, mobile_payment)
        print(f"   Mobile Payment: {message}")
    
    print()
    
    # Test transaction cancellation
    print("9. TRANSACTION CANCELLATION TEST:")
    
    # Select product but cancel
    success, message = machine.select_product("D1")  # Apple
    print(f"   Product Selection: {message}")
    
    if success:
        success, message = machine.cancel_transaction()
        print(f"   Transaction Cancellation: {message}")
    
    print()
    
    # Show updated machine status
    print("10. UPDATED MACHINE STATUS:")
    
    status = machine.get_machine_status()
    print(f"   Machine State: {status['state']}")
    print(f"   Total Sales: {status['total_sales']}")
    print(f"   Transactions Today: {status['transactions_today']}")
    
    # Show inventory changes
    print("\n   Inventory Changes:")
    for slot_id, slot_info in status['inventory']['slots'].items():
        if slot_info:
            print(f"     {slot_id}: {slot_info['product']['name']} "
                  f"({slot_info['quantity']} left) - {slot_info['price']}")
    
    print()
    
    # Test maintenance
    print("11. MAINTENANCE OPERATIONS:")
    
    # Check if maintenance needed
    if status['maintenance_required']:
        print("   ⚠️  Maintenance required")
    
    # Perform maintenance
    success = machine.perform_maintenance("full", "Scheduled maintenance")
    print(f"   Full Maintenance: {'Completed' if success else 'Failed'}")
    
    # Check status after maintenance
    status = machine.get_machine_status()
    print(f"   Post-maintenance state: {status['state']}")
    print(f"   Maintenance required: {status['maintenance_required']}")
    
    print()
    
    # Generate sales report
    print("12. SALES REPORTING:")
    
    report = machine.get_sales_report()
    print(f"   Report Period: {report['period']['start_date'][:10]} to {report['period']['end_date'][:10]}")
    print(f"   Total Transactions: {report['summary']['total_transactions']}")
    print(f"   Total Revenue: {report['summary']['total_revenue']}")
    print(f"   Average Transaction: {report['summary']['average_transaction']}")
    
    if report['top_selling_products']:
        print("   Top Selling Products:")
        for product, sales in report['top_selling_products']:
            print(f"     {product}: {sales} units")
    
    if report['payment_methods']:
        print("   Payment Method Usage:")
        for method, count in report['payment_methods'].items():
            print(f"     {method}: {count} transactions")
    
    print()
    
    # Test error scenarios
    print("13. ERROR SCENARIO TESTING:")
    
    # Try to select invalid slot
    success, message = machine.select_product("Z9")
    print(f"   Invalid Slot Selection: {message}")
    
    # Try to buy when machine is in maintenance
    machine.state = MachineState.MAINTENANCE
    success, message = machine.select_product("A1")
    print(f"   Selection During Maintenance: {message}")
    machine.state = MachineState.IDLE
    
    # Try insufficient cash payment
    success, message = machine.select_product("A1")  # $1.50 item
    if success:
        insufficient_cash = {
            'inserted_cash': [
                {'value': 1.00, 'count': 1, 'type': 'bill'},  # Only $1.00
            ]
        }
        success, message = machine.process_payment(PaymentMethod.CASH, insufficient_cash)
        print(f"   Insufficient Cash: {message}")
    
    print()
    
    # Show final status
    print("14. FINAL SYSTEM STATUS:")
    
    final_status = machine.get_machine_status()
    
    print(f"   Machine ID: {final_status['machine_id']}")
    print(f"   Location: {final_status['location']}")
    print(f"   Current State: {final_status['state']}")
    print(f"   Total Sales: {final_status['total_sales']}")
    print(f"   Cash Available: {final_status['cash']['total_cash']}")
    print(f"   Can Make Change: {final_status['cash']['can_make_change']}")
    print(f"   Products Available: {final_status['inventory']['occupied_slots']}")
    print(f"   Low Stock Alerts: {final_status['inventory']['low_stock_slots']}")
    print(f"   Last Maintenance: {final_status['last_maintenance'][:10]}")
    
    print()
    print("=== VENDING MACHINE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_vending_machine()
