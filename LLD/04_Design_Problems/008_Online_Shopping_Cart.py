"""
ONLINE SHOPPING CART SYSTEM - Complete System Design
====================================================

Problem Statement:
Design a comprehensive online shopping cart system that handles:
- Product catalog and inventory management
- Shopping cart operations (add, remove, update)
- User session management
- Pricing and discount calculations
- Tax and shipping calculations
- Order processing and checkout
- Payment integration
- Inventory tracking and updates
- Wishlist and saved items
- Product recommendations

Requirements:
- Support multiple product types and variations
- Handle guest and registered user carts
- Implement cart persistence across sessions
- Support multiple currencies and tax rates
- Handle promotional codes and discounts
- Calculate shipping costs based on location
- Process various payment methods
- Track inventory in real-time
- Support cart sharing and saved carts
- Implement cart abandonment recovery

Design Patterns Used:
- Strategy: Pricing and discount strategies
- Observer: Cart change notifications
- Command: Cart operations
- Factory: Product and cart creation
- State: Cart and order states
- Decorator: Product features and discounts
- Singleton: Cart manager
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
from dataclasses import dataclass, field
from decimal import Decimal
import json
from collections import defaultdict


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ProductType(Enum):
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"


class CartStatus(Enum):
    ACTIVE = "active"
    SAVED = "saved"
    ABANDONED = "abandoned"
    CONVERTED = "converted"
    EXPIRED = "expired"


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class DiscountType(Enum):
    PERCENTAGE = "percentage"
    FIXED_AMOUNT = "fixed_amount"
    BUY_X_GET_Y = "buy_x_get_y"
    FREE_SHIPPING = "free_shipping"


class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"


@dataclass
class Address:
    """Shipping/billing address."""
    street: str
    city: str
    state: str
    country: str
    zip_code: str
    is_default: bool = False


@dataclass
class ProductVariation:
    """Product variation (size, color, etc.)."""
    variation_id: str
    name: str
    value: str
    price_modifier: Decimal = Decimal('0')
    stock_quantity: int = 0


@dataclass
class ShippingOption:
    """Shipping method option."""
    option_id: str
    name: str
    description: str
    cost: Decimal
    estimated_days: int
    is_available: bool = True


# ============================================================================
# DISCOUNT STRATEGIES
# ============================================================================

class DiscountStrategy(ABC):
    """Abstract discount strategy."""
    
    @abstractmethod
    def calculate_discount(self, cart_total: Decimal, items: List['CartItem'], 
                          customer_tier: str = "regular") -> Decimal:
        """Calculate discount amount."""
        pass
    
    @abstractmethod
    def is_applicable(self, cart_total: Decimal, items: List['CartItem']) -> bool:
        """Check if discount is applicable."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get discount description."""
        pass


class PercentageDiscountStrategy(DiscountStrategy):
    """Percentage-based discount."""
    
    def __init__(self, percentage: Decimal, min_amount: Decimal = Decimal('0'),
                 max_discount: Decimal = None):
        self.percentage = percentage
        self.min_amount = min_amount
        self.max_discount = max_discount
    
    def calculate_discount(self, cart_total: Decimal, items: List['CartItem'], 
                          customer_tier: str = "regular") -> Decimal:
        """Calculate percentage discount."""
        if not self.is_applicable(cart_total, items):
            return Decimal('0')
        
        discount = cart_total * (self.percentage / Decimal('100'))
        
        if self.max_discount:
            discount = min(discount, self.max_discount)
        
        return discount
    
    def is_applicable(self, cart_total: Decimal, items: List['CartItem']) -> bool:
        """Check if minimum amount is met."""
        return cart_total >= self.min_amount
    
    def get_description(self) -> str:
        """Get discount description."""
        desc = f"{self.percentage}% off"
        if self.min_amount > 0:
            desc += f" on orders over ${self.min_amount}"
        if self.max_discount:
            desc += f" (max ${self.max_discount})"
        return desc


class FixedAmountDiscountStrategy(DiscountStrategy):
    """Fixed amount discount."""
    
    def __init__(self, amount: Decimal, min_amount: Decimal = Decimal('0')):
        self.amount = amount
        self.min_amount = min_amount
    
    def calculate_discount(self, cart_total: Decimal, items: List['CartItem'], 
                          customer_tier: str = "regular") -> Decimal:
        """Calculate fixed amount discount."""
        if not self.is_applicable(cart_total, items):
            return Decimal('0')
        
        return min(self.amount, cart_total)
    
    def is_applicable(self, cart_total: Decimal, items: List['CartItem']) -> bool:
        """Check if minimum amount is met."""
        return cart_total >= self.min_amount
    
    def get_description(self) -> str:
        """Get discount description."""
        desc = f"${self.amount} off"
        if self.min_amount > 0:
            desc += f" on orders over ${self.min_amount}"
        return desc


class BuyXGetYDiscountStrategy(DiscountStrategy):
    """Buy X get Y free discount."""
    
    def __init__(self, buy_quantity: int, get_quantity: int, 
                 applicable_products: Set[str] = None):
        self.buy_quantity = buy_quantity
        self.get_quantity = get_quantity
        self.applicable_products = applicable_products or set()
    
    def calculate_discount(self, cart_total: Decimal, items: List['CartItem'], 
                          customer_tier: str = "regular") -> Decimal:
        """Calculate buy X get Y discount."""
        if not self.is_applicable(cart_total, items):
            return Decimal('0')
        
        total_discount = Decimal('0')
        
        for item in items:
            if (not self.applicable_products or 
                item.product.product_id in self.applicable_products):
                
                eligible_sets = item.quantity // self.buy_quantity
                free_items = eligible_sets * self.get_quantity
                free_items = min(free_items, item.quantity - eligible_sets * self.buy_quantity)
                
                discount_per_item = item.unit_price
                total_discount += discount_per_item * free_items
        
        return total_discount
    
    def is_applicable(self, cart_total: Decimal, items: List['CartItem']) -> bool:
        """Check if any applicable items meet the buy quantity."""
        for item in items:
            if (not self.applicable_products or 
                item.product.product_id in self.applicable_products):
                if item.quantity >= self.buy_quantity:
                    return True
        return False
    
    def get_description(self) -> str:
        """Get discount description."""
        return f"Buy {self.buy_quantity} get {self.get_quantity} free"


# ============================================================================
# PRODUCT CLASSES
# ============================================================================

class Product:
    """Product with variations and inventory."""
    
    def __init__(self, product_id: str, name: str, description: str, 
                 base_price: Decimal, product_type: ProductType):
        self.product_id = product_id
        self.name = name
        self.description = description
        self.base_price = base_price
        self.product_type = product_type
        
        # Inventory
        self.stock_quantity = 0
        self.reserved_quantity = 0  # Items in carts but not ordered
        self.low_stock_threshold = 10
        
        # Variations
        self.variations: Dict[str, ProductVariation] = {}
        
        # Categories and attributes
        self.categories: Set[str] = set()
        self.tags: Set[str] = set()
        self.attributes: Dict[str, str] = {}
        
        # Media and details
        self.images: List[str] = []
        self.weight = Decimal('0')  # For shipping calculations
        self.dimensions = {"length": 0, "width": 0, "height": 0}
        
        # Pricing and promotions
        self.sale_price: Optional[Decimal] = None
        self.sale_start: Optional[datetime] = None
        self.sale_end: Optional[datetime] = None
        
        # Statistics
        self.view_count = 0
        self.purchase_count = 0
        self.rating_sum = 0
        self.rating_count = 0
        
        # Availability
        self.is_active = True
        self.is_featured = False
        
        self.created_at = datetime.now()
        self._lock = threading.Lock()
    
    def add_variation(self, variation: ProductVariation) -> None:
        """Add product variation."""
        self.variations[variation.variation_id] = variation
    
    def remove_variation(self, variation_id: str) -> bool:
        """Remove product variation."""
        if variation_id in self.variations:
            del self.variations[variation_id]
            return True
        return False
    
    def add_category(self, category: str) -> None:
        """Add product category."""
        self.categories.add(category)
    
    def add_tag(self, tag: str) -> None:
        """Add product tag."""
        self.tags.add(tag)
    
    def set_attribute(self, key: str, value: str) -> None:
        """Set product attribute."""
        self.attributes[key] = value
    
    def get_current_price(self, variation_id: str = None) -> Decimal:
        """Get current price including sales and variations."""
        base_price = self.sale_price if self.is_on_sale() else self.base_price
        
        if variation_id and variation_id in self.variations:
            variation = self.variations[variation_id]
            base_price += variation.price_modifier
        
        return base_price
    
    def is_on_sale(self) -> bool:
        """Check if product is currently on sale."""
        if not self.sale_price or not self.sale_start or not self.sale_end:
            return False
        
        now = datetime.now()
        return self.sale_start <= now <= self.sale_end
    
    def get_available_stock(self, variation_id: str = None) -> int:
        """Get available stock quantity."""
        with self._lock:
            if variation_id and variation_id in self.variations:
                return self.variations[variation_id].stock_quantity
            
            return max(0, self.stock_quantity - self.reserved_quantity)
    
    def reserve_stock(self, quantity: int, variation_id: str = None) -> bool:
        """Reserve stock for cart."""
        with self._lock:
            available = self.get_available_stock(variation_id)
            if available >= quantity:
                if variation_id and variation_id in self.variations:
                    self.variations[variation_id].stock_quantity -= quantity
                else:
                    self.reserved_quantity += quantity
                return True
            return False
    
    def release_stock(self, quantity: int, variation_id: str = None) -> None:
        """Release reserved stock."""
        with self._lock:
            if variation_id and variation_id in self.variations:
                self.variations[variation_id].stock_quantity += quantity
            else:
                self.reserved_quantity = max(0, self.reserved_quantity - quantity)
    
    def purchase_stock(self, quantity: int, variation_id: str = None) -> bool:
        """Purchase stock (convert from reserved to sold)."""
        with self._lock:
            if variation_id and variation_id in self.variations:
                # Stock already deducted during reservation
                pass
            else:
                if self.reserved_quantity >= quantity:
                    self.reserved_quantity -= quantity
                    self.stock_quantity -= quantity
                else:
                    return False
            
            self.purchase_count += quantity
            return True
    
    def is_in_stock(self, quantity: int = 1, variation_id: str = None) -> bool:
        """Check if product is in stock."""
        return self.get_available_stock(variation_id) >= quantity
    
    def is_low_stock(self, variation_id: str = None) -> bool:
        """Check if product is low in stock."""
        return self.get_available_stock(variation_id) <= self.low_stock_threshold
    
    def add_rating(self, rating: int) -> None:
        """Add customer rating (1-5)."""
        if 1 <= rating <= 5:
            self.rating_sum += rating
            self.rating_count += 1
    
    def get_average_rating(self) -> float:
        """Get average rating."""
        if self.rating_count == 0:
            return 0.0
        return self.rating_sum / self.rating_count
    
    def increment_view_count(self) -> None:
        """Increment product view count."""
        self.view_count += 1
    
    def get_product_info(self, variation_id: str = None) -> Dict[str, Any]:
        """Get product information."""
        info = {
            'product_id': self.product_id,
            'name': self.name,
            'description': self.description,
            'base_price': float(self.base_price),
            'current_price': float(self.get_current_price(variation_id)),
            'product_type': self.product_type.value,
            'stock_quantity': self.get_available_stock(variation_id),
            'is_in_stock': self.is_in_stock(1, variation_id),
            'is_low_stock': self.is_low_stock(variation_id),
            'categories': list(self.categories),
            'tags': list(self.tags),
            'attributes': self.attributes,
            'images': self.images,
            'weight': float(self.weight),
            'dimensions': self.dimensions,
            'is_active': self.is_active,
            'is_featured': self.is_featured,
            'is_on_sale': self.is_on_sale(),
            'sale_price': float(self.sale_price) if self.sale_price else None,
            'variations': {
                var_id: {
                    'name': var.name,
                    'value': var.value,
                    'price_modifier': float(var.price_modifier),
                    'stock_quantity': var.stock_quantity
                }
                for var_id, var in self.variations.items()
            },
            'statistics': {
                'view_count': self.view_count,
                'purchase_count': self.purchase_count,
                'average_rating': self.get_average_rating(),
                'rating_count': self.rating_count
            },
            'created_at': self.created_at.isoformat()
        }
        
        if variation_id and variation_id in self.variations:
            variation = self.variations[variation_id]
            info['selected_variation'] = {
                'variation_id': variation_id,
                'name': variation.name,
                'value': variation.value,
                'price_modifier': float(variation.price_modifier)
            }
        
        return info
    
    def __str__(self) -> str:
        return f"{self.name} - ${self.get_current_price()} ({self.product_type.value})"


# ============================================================================
# CART CLASSES
# ============================================================================

class CartItem:
    """Item in shopping cart."""
    
    def __init__(self, product: Product, quantity: int, variation_id: str = None):
        self.product = product
        self.quantity = quantity
        self.variation_id = variation_id
        self.unit_price = product.get_current_price(variation_id)
        self.total_price = self.unit_price * quantity
        self.added_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Reserve stock
        self.stock_reserved = product.reserve_stock(quantity, variation_id)
    
    def update_quantity(self, new_quantity: int) -> bool:
        """Update item quantity."""
        if new_quantity <= 0:
            return False
        
        quantity_diff = new_quantity - self.quantity
        
        if quantity_diff > 0:
            # Need to reserve more stock
            if self.product.reserve_stock(quantity_diff, self.variation_id):
                self.quantity = new_quantity
                self.total_price = self.unit_price * self.quantity
                self.updated_at = datetime.now()
                return True
            return False
        elif quantity_diff < 0:
            # Release some stock
            self.product.release_stock(-quantity_diff, self.variation_id)
            self.quantity = new_quantity
            self.total_price = self.unit_price * self.quantity
            self.updated_at = datetime.now()
            return True
        
        return True  # No change needed
    
    def update_price(self) -> None:
        """Update price based on current product price."""
        new_unit_price = self.product.get_current_price(self.variation_id)
        if new_unit_price != self.unit_price:
            self.unit_price = new_unit_price
            self.total_price = self.unit_price * self.quantity
            self.updated_at = datetime.now()
    
    def release_stock(self) -> None:
        """Release reserved stock."""
        if self.stock_reserved:
            self.product.release_stock(self.quantity, self.variation_id)
            self.stock_reserved = False
    
    def get_item_info(self) -> Dict[str, Any]:
        """Get cart item information."""
        return {
            'product': self.product.get_product_info(self.variation_id),
            'quantity': self.quantity,
            'unit_price': float(self.unit_price),
            'total_price': float(self.total_price),
            'variation_id': self.variation_id,
            'added_at': self.added_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'stock_reserved': self.stock_reserved
        }
    
    def __str__(self) -> str:
        return f"{self.product.name} x{self.quantity} - ${self.total_price}"


class ShoppingCart:
    """Shopping cart with items and calculations."""
    
    def __init__(self, cart_id: str, user_id: str = None, session_id: str = None):
        self.cart_id = cart_id
        self.user_id = user_id  # None for guest carts
        self.session_id = session_id
        self.status = CartStatus.ACTIVE
        
        # Cart items
        self.items: Dict[str, CartItem] = {}  # product_id+variation_id -> CartItem
        
        # Pricing
        self.subtotal = Decimal('0')
        self.discount_amount = Decimal('0')
        self.tax_amount = Decimal('0')
        self.shipping_cost = Decimal('0')
        self.total_amount = Decimal('0')
        
        # Applied discounts and coupons
        self.applied_discounts: List[Dict[str, Any]] = []
        self.coupon_codes: Set[str] = set()
        
        # Shipping and billing
        self.shipping_address: Optional[Address] = None
        self.billing_address: Optional[Address] = None
        self.selected_shipping_option: Optional[ShippingOption] = None
        
        # Timestamps
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(days=30)  # Cart expiration
        
        # Settings
        self.currency = "USD"
        self.tax_rate = Decimal('0.08')  # 8% default tax rate
        
        self._lock = threading.Lock()
    
    def _generate_item_key(self, product_id: str, variation_id: str = None) -> str:
        """Generate unique key for cart item."""
        return f"{product_id}_{variation_id or 'default'}"
    
    def add_item(self, product: Product, quantity: int, variation_id: str = None) -> bool:
        """Add item to cart."""
        with self._lock:
            if not product.is_active or not product.is_in_stock(quantity, variation_id):
                return False
            
            item_key = self._generate_item_key(product.product_id, variation_id)
            
            if item_key in self.items:
                # Update existing item
                existing_item = self.items[item_key]
                new_quantity = existing_item.quantity + quantity
                return existing_item.update_quantity(new_quantity)
            else:
                # Add new item
                cart_item = CartItem(product, quantity, variation_id)
                if cart_item.stock_reserved:
                    self.items[item_key] = cart_item
                    self._update_totals()
                    return True
                return False
    
    def update_item_quantity(self, product_id: str, quantity: int, variation_id: str = None) -> bool:
        """Update item quantity."""
        with self._lock:
            item_key = self._generate_item_key(product_id, variation_id)
            
            if item_key not in self.items:
                return False
            
            if quantity <= 0:
                return self.remove_item(product_id, variation_id)
            
            success = self.items[item_key].update_quantity(quantity)
            if success:
                self._update_totals()
            
            return success
    
    def remove_item(self, product_id: str, variation_id: str = None) -> bool:
        """Remove item from cart."""
        with self._lock:
            item_key = self._generate_item_key(product_id, variation_id)
            
            if item_key in self.items:
                item = self.items[item_key]
                item.release_stock()
                del self.items[item_key]
                self._update_totals()
                return True
            
            return False
    
    def clear_cart(self) -> None:
        """Clear all items from cart."""
        with self._lock:
            for item in self.items.values():
                item.release_stock()
            
            self.items.clear()
            self.applied_discounts.clear()
            self.coupon_codes.clear()
            self._update_totals()
    
    def apply_discount(self, discount_strategy: DiscountStrategy, 
                      discount_code: str = "", customer_tier: str = "regular") -> bool:
        """Apply discount to cart."""
        with self._lock:
            if not discount_strategy.is_applicable(self.subtotal, list(self.items.values())):
                return False
            
            discount_amount = discount_strategy.calculate_discount(
                self.subtotal, list(self.items.values()), customer_tier
            )
            
            if discount_amount > 0:
                discount_info = {
                    'code': discount_code,
                    'description': discount_strategy.get_description(),
                    'amount': discount_amount,
                    'applied_at': datetime.now().isoformat()
                }
                
                self.applied_discounts.append(discount_info)
                if discount_code:
                    self.coupon_codes.add(discount_code)
                
                self._update_totals()
                return True
            
            return False
    
    def remove_discount(self, discount_code: str) -> bool:
        """Remove applied discount."""
        with self._lock:
            for i, discount in enumerate(self.applied_discounts):
                if discount['code'] == discount_code:
                    del self.applied_discounts[i]
                    self.coupon_codes.discard(discount_code)
                    self._update_totals()
                    return True
            
            return False
    
    def set_shipping_address(self, address: Address) -> None:
        """Set shipping address."""
        self.shipping_address = address
        self._update_totals()  # Recalculate shipping and tax
    
    def set_billing_address(self, address: Address) -> None:
        """Set billing address."""
        self.billing_address = address
        self._update_totals()  # Recalculate tax
    
    def set_shipping_option(self, shipping_option: ShippingOption) -> None:
        """Set shipping option."""
        self.selected_shipping_option = shipping_option
        self._update_totals()
    
    def _update_totals(self) -> None:
        """Update cart totals."""
        # Calculate subtotal
        self.subtotal = sum(item.total_price for item in self.items.values())
        
        # Update item prices (in case of price changes)
        for item in self.items.values():
            item.update_price()
        
        # Recalculate subtotal after price updates
        self.subtotal = sum(item.total_price for item in self.items.values())
        
        # Calculate total discount
        self.discount_amount = sum(
            Decimal(str(discount['amount'])) for discount in self.applied_discounts
        )
        
        # Calculate shipping cost
        self.shipping_cost = Decimal('0')
        if self.selected_shipping_option and self.has_physical_items():
            # Check if free shipping discount is applied
            has_free_shipping = any(
                'free shipping' in discount['description'].lower()
                for discount in self.applied_discounts
            )
            
            if not has_free_shipping:
                self.shipping_cost = self.selected_shipping_option.cost
        
        # Calculate tax (on subtotal - discount + shipping)
        taxable_amount = self.subtotal - self.discount_amount + self.shipping_cost
        self.tax_amount = taxable_amount * self.tax_rate
        
        # Calculate total
        self.total_amount = self.subtotal - self.discount_amount + self.shipping_cost + self.tax_amount
        
        # Update timestamp
        self.updated_at = datetime.now()
    
    def has_physical_items(self) -> bool:
        """Check if cart has physical items requiring shipping."""
        return any(
            item.product.product_type == ProductType.PHYSICAL
            for item in self.items.values()
        )
    
    def get_item_count(self) -> int:
        """Get total number of items in cart."""
        return sum(item.quantity for item in self.items.values())
    
    def get_unique_item_count(self) -> int:
        """Get number of unique items in cart."""
        return len(self.items)
    
    def is_empty(self) -> bool:
        """Check if cart is empty."""
        return len(self.items) == 0
    
    def is_expired(self) -> bool:
        """Check if cart is expired."""
        return datetime.now() > self.expires_at
    
    def extend_expiration(self, days: int = 30) -> None:
        """Extend cart expiration."""
        self.expires_at = datetime.now() + timedelta(days=days)
    
    def save_cart(self) -> None:
        """Save cart for later."""
        self.status = CartStatus.SAVED
        self.extend_expiration(90)  # Saved carts last longer
    
    def abandon_cart(self) -> None:
        """Mark cart as abandoned."""
        self.status = CartStatus.ABANDONED
    
    def convert_cart(self) -> None:
        """Mark cart as converted to order."""
        self.status = CartStatus.CONVERTED
    
    def get_cart_summary(self) -> Dict[str, Any]:
        """Get cart summary information."""
        return {
            'cart_id': self.cart_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'status': self.status.value,
            'item_count': self.get_item_count(),
            'unique_item_count': self.get_unique_item_count(),
            'subtotal': float(self.subtotal),
            'discount_amount': float(self.discount_amount),
            'tax_amount': float(self.tax_amount),
            'shipping_cost': float(self.shipping_cost),
            'total_amount': float(self.total_amount),
            'currency': self.currency,
            'has_physical_items': self.has_physical_items(),
            'applied_discounts': self.applied_discounts,
            'coupon_codes': list(self.coupon_codes),
            'shipping_address': {
                'street': self.shipping_address.street,
                'city': self.shipping_address.city,
                'state': self.shipping_address.state,
                'country': self.shipping_address.country,
                'zip_code': self.shipping_address.zip_code
            } if self.shipping_address else None,
            'selected_shipping_option': {
                'option_id': self.selected_shipping_option.option_id,
                'name': self.selected_shipping_option.name,
                'cost': float(self.selected_shipping_option.cost),
                'estimated_days': self.selected_shipping_option.estimated_days
            } if self.selected_shipping_option else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }
    
    def get_detailed_cart(self) -> Dict[str, Any]:
        """Get detailed cart information including all items."""
        summary = self.get_cart_summary()
        summary['items'] = [item.get_item_info() for item in self.items.values()]
        return summary
    
    def __str__(self) -> str:
        return f"Cart {self.cart_id} - {self.get_item_count()} items (${self.total_amount})"


# ============================================================================
# CART MANAGER
# ============================================================================

class ShoppingCartManager:
    """Manager for shopping carts and operations."""
    
    def __init__(self):
        self.carts: Dict[str, ShoppingCart] = {}
        self.user_carts: Dict[str, str] = {}  # user_id -> cart_id
        self.session_carts: Dict[str, str] = {}  # session_id -> cart_id
        
        # Product catalog
        self.products: Dict[str, Product] = {}
        
        # Shipping options
        self.shipping_options: Dict[str, ShippingOption] = {}
        
        # Discount strategies
        self.available_discounts: Dict[str, DiscountStrategy] = {}
        
        # Statistics
        self.total_carts_created = 0
        self.total_orders_converted = 0
        self.total_revenue = Decimal('0')
        
        self._lock = threading.Lock()
        
        print("🛒 Shopping Cart Manager initialized")
    
    def add_product(self, product: Product) -> None:
        """Add product to catalog."""
        self.products[product.product_id] = product
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        return self.products.get(product_id)
    
    def add_shipping_option(self, shipping_option: ShippingOption) -> None:
        """Add shipping option."""
        self.shipping_options[shipping_option.option_id] = shipping_option
    
    def add_discount_strategy(self, code: str, strategy: DiscountStrategy) -> None:
        """Add discount strategy."""
        self.available_discounts[code] = strategy
    
    def create_cart(self, user_id: str = None, session_id: str = None) -> ShoppingCart:
        """Create new shopping cart."""
        with self._lock:
            cart_id = str(uuid.uuid4())
            cart = ShoppingCart(cart_id, user_id, session_id)
            
            self.carts[cart_id] = cart
            
            if user_id:
                self.user_carts[user_id] = cart_id
            
            if session_id:
                self.session_carts[session_id] = cart_id
            
            self.total_carts_created += 1
            
            return cart
    
    def get_cart(self, cart_id: str) -> Optional[ShoppingCart]:
        """Get cart by ID."""
        return self.carts.get(cart_id)
    
    def get_user_cart(self, user_id: str) -> Optional[ShoppingCart]:
        """Get cart for user."""
        cart_id = self.user_carts.get(user_id)
        if cart_id:
            return self.carts.get(cart_id)
        return None
    
    def get_session_cart(self, session_id: str) -> Optional[ShoppingCart]:
        """Get cart for session."""
        cart_id = self.session_carts.get(session_id)
        if cart_id:
            return self.carts.get(cart_id)
        return None
    
    def merge_carts(self, source_cart_id: str, target_cart_id: str) -> bool:
        """Merge source cart into target cart."""
        source_cart = self.get_cart(source_cart_id)
        target_cart = self.get_cart(target_cart_id)
        
        if not source_cart or not target_cart:
            return False
        
        # Merge items
        for item in source_cart.items.values():
            target_cart.add_item(
                item.product, 
                item.quantity, 
                item.variation_id
            )
        
        # Clear source cart
        source_cart.clear_cart()
        
        return True
    
    def add_to_cart(self, cart_id: str, product_id: str, quantity: int, 
                   variation_id: str = None) -> bool:
        """Add item to cart."""
        cart = self.get_cart(cart_id)
        product = self.get_product(product_id)
        
        if not cart or not product:
            return False
        
        return cart.add_item(product, quantity, variation_id)
    
    def update_cart_item(self, cart_id: str, product_id: str, quantity: int,
                        variation_id: str = None) -> bool:
        """Update cart item quantity."""
        cart = self.get_cart(cart_id)
        if not cart:
            return False
        
        return cart.update_item_quantity(product_id, quantity, variation_id)
    
    def remove_from_cart(self, cart_id: str, product_id: str, 
                        variation_id: str = None) -> bool:
        """Remove item from cart."""
        cart = self.get_cart(cart_id)
        if not cart:
            return False
        
        return cart.remove_item(product_id, variation_id)
    
    def apply_coupon(self, cart_id: str, coupon_code: str, 
                    customer_tier: str = "regular") -> bool:
        """Apply coupon to cart."""
        cart = self.get_cart(cart_id)
        discount_strategy = self.available_discounts.get(coupon_code)
        
        if not cart or not discount_strategy:
            return False
        
        return cart.apply_discount(discount_strategy, coupon_code, customer_tier)
    
    def set_shipping_info(self, cart_id: str, address: Address, 
                         shipping_option_id: str) -> bool:
        """Set shipping information for cart."""
        cart = self.get_cart(cart_id)
        shipping_option = self.shipping_options.get(shipping_option_id)
        
        if not cart or not shipping_option:
            return False
        
        cart.set_shipping_address(address)
        cart.set_shipping_option(shipping_option)
        
        return True
    
    def checkout_cart(self, cart_id: str, payment_method: PaymentMethod,
                     billing_address: Address = None) -> Optional[str]:
        """Checkout cart and create order."""
        cart = self.get_cart(cart_id)
        if not cart or cart.is_empty():
            return None
        
        # Set billing address
        if billing_address:
            cart.set_billing_address(billing_address)
        
        # Validate cart
        if not self._validate_cart_for_checkout(cart):
            return None
        
        # Convert reserved stock to purchased
        for item in cart.items.values():
            if not item.product.purchase_stock(item.quantity, item.variation_id):
                return None  # Stock not available
        
        # Create order ID
        order_id = str(uuid.uuid4())
        
        # Mark cart as converted
        cart.convert_cart()
        
        # Update statistics
        with self._lock:
            self.total_orders_converted += 1
            self.total_revenue += cart.total_amount
        
        return order_id
    
    def _validate_cart_for_checkout(self, cart: ShoppingCart) -> bool:
        """Validate cart for checkout."""
        # Check if cart has items
        if cart.is_empty():
            return False
        
        # Check stock availability
        for item in cart.items.values():
            if not item.product.is_in_stock(item.quantity, item.variation_id):
                return False
        
        # Check if physical items have shipping address
        if cart.has_physical_items() and not cart.shipping_address:
            return False
        
        # Check if shipping option is selected for physical items
        if cart.has_physical_items() and not cart.selected_shipping_option:
            return False
        
        return True
    
    def abandon_cart(self, cart_id: str) -> bool:
        """Mark cart as abandoned."""
        cart = self.get_cart(cart_id)
        if not cart:
            return False
        
        cart.abandon_cart()
        return True
    
    def cleanup_expired_carts(self) -> int:
        """Clean up expired carts."""
        expired_count = 0
        
        with self._lock:
            expired_cart_ids = []
            
            for cart_id, cart in self.carts.items():
                if cart.is_expired() and cart.status == CartStatus.ACTIVE:
                    cart.abandon_cart()
                    # Release reserved stock
                    for item in cart.items.values():
                        item.release_stock()
                    expired_cart_ids.append(cart_id)
                    expired_count += 1
            
            # Remove expired carts
            for cart_id in expired_cart_ids:
                del self.carts[cart_id]
        
        return expired_count
    
    def get_cart_analytics(self) -> Dict[str, Any]:
        """Get cart analytics."""
        active_carts = len([c for c in self.carts.values() if c.status == CartStatus.ACTIVE])
        abandoned_carts = len([c for c in self.carts.values() if c.status == CartStatus.ABANDONED])
        converted_carts = len([c for c in self.carts.values() if c.status == CartStatus.CONVERTED])
        
        total_cart_value = sum(
            cart.total_amount for cart in self.carts.values() 
            if cart.status == CartStatus.ACTIVE
        )
        
        average_cart_value = total_cart_value / max(1, active_carts)
        
        conversion_rate = (converted_carts / max(1, self.total_carts_created)) * 100
        
        return {
            'total_carts_created': self.total_carts_created,
            'active_carts': active_carts,
            'abandoned_carts': abandoned_carts,
            'converted_carts': converted_carts,
            'conversion_rate': float(conversion_rate),
            'total_cart_value': float(total_cart_value),
            'average_cart_value': float(average_cart_value),
            'total_revenue': float(self.total_revenue),
            'products_in_catalog': len(self.products),
            'available_shipping_options': len(self.shipping_options),
            'available_discounts': len(self.available_discounts)
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_shopping_cart_system():
    """Demonstrate the shopping cart system."""
    print("=== ONLINE SHOPPING CART SYSTEM DEMONSTRATION ===\n")
    
    # Initialize cart manager
    cart_manager = ShoppingCartManager()
    
    print("1. PRODUCT CATALOG SETUP:")
    
    # Create products
    products_data = [
        ("Laptop", "High-performance laptop", Decimal('999.99'), ProductType.PHYSICAL, 50),
        ("E-book", "Digital programming guide", Decimal('29.99'), ProductType.DIGITAL, 1000),
        ("Headphones", "Wireless noise-canceling headphones", Decimal('199.99'), ProductType.PHYSICAL, 25),
        ("Software License", "Annual software subscription", Decimal('99.99'), ProductType.SUBSCRIPTION, 100),
        ("Consulting", "1-hour consulting session", Decimal('150.00'), ProductType.SERVICE, 10)
    ]
    
    products = []
    for name, desc, price, prod_type, stock in products_data:
        product_id = str(uuid.uuid4())
        product = Product(product_id, name, desc, price, prod_type)
        product.stock_quantity = stock
        product.weight = Decimal('2.5') if prod_type == ProductType.PHYSICAL else Decimal('0')
        
        # Add categories and tags
        if "Laptop" in name:
            product.add_category("Electronics")
            product.add_category("Computers")
            product.add_tag("portable")
            product.add_tag("business")
        elif "Headphones" in name:
            product.add_category("Electronics")
            product.add_category("Audio")
            product.add_tag("wireless")
            product.add_tag("music")
        elif "E-book" in name:
            product.add_category("Books")
            product.add_category("Technology")
            product.add_tag("programming")
            product.add_tag("digital")
        
        # Add variations for some products
        if "Laptop" in name:
            # RAM variations
            ram_8gb = ProductVariation("ram_8gb", "RAM", "8GB", Decimal('0'))
            ram_16gb = ProductVariation("ram_16gb", "RAM", "16GB", Decimal('200'))
            ram_32gb = ProductVariation("ram_32gb", "RAM", "32GB", Decimal('500'))
            
            ram_8gb.stock_quantity = 20
            ram_16gb.stock_quantity = 20
            ram_32gb.stock_quantity = 10
            
            product.add_variation(ram_8gb)
            product.add_variation(ram_16gb)
            product.add_variation(ram_32gb)
        
        elif "Headphones" in name:
            # Color variations
            black = ProductVariation("color_black", "Color", "Black", Decimal('0'))
            white = ProductVariation("color_white", "Color", "White", Decimal('0'))
            red = ProductVariation("color_red", "Color", "Red", Decimal('25'))
            
            black.stock_quantity = 15
            white.stock_quantity = 8
            red.stock_quantity = 2
            
            product.add_variation(black)
            product.add_variation(white)
            product.add_variation(red)
        
        # Set some products on sale
        if "E-book" in name:
            product.sale_price = Decimal('19.99')
            product.sale_start = datetime.now() - timedelta(days=1)
            product.sale_end = datetime.now() + timedelta(days=7)
        
        cart_manager.add_product(product)
        products.append(product)
        print(f"   ✓ {name} added - ${price} (Stock: {stock})")
    
    print()
    
    # Setup shipping options
    print("2. SHIPPING OPTIONS SETUP:")
    
    shipping_data = [
        ("standard", "Standard Shipping", "5-7 business days", Decimal('5.99'), 7),
        ("express", "Express Shipping", "2-3 business days", Decimal('12.99'), 3),
        ("overnight", "Overnight Shipping", "Next business day", Decimal('24.99'), 1)
    ]
    
    for option_id, name, desc, cost, days in shipping_data:
        shipping_option = ShippingOption(option_id, name, desc, cost, days)
        cart_manager.add_shipping_option(shipping_option)
        print(f"   ✓ {name}: ${cost} ({days} days)")
    
    print()
    
    # Setup discount strategies
    print("3. DISCOUNT STRATEGIES SETUP:")
    
    # Percentage discount
    percentage_discount = PercentageDiscountStrategy(
        percentage=Decimal('10'),
        min_amount=Decimal('100'),
        max_discount=Decimal('50')
    )
    cart_manager.add_discount_strategy("SAVE10", percentage_discount)
    
    # Fixed amount discount
    fixed_discount = FixedAmountDiscountStrategy(
        amount=Decimal('25'),
        min_amount=Decimal('150')
    )
    cart_manager.add_discount_strategy("SAVE25", fixed_discount)
    
    # Buy X Get Y discount
    bogo_discount = BuyXGetYDiscountStrategy(
        buy_quantity=2,
        get_quantity=1,
        applicable_products={products[2].product_id}  # Headphones
    )
    cart_manager.add_discount_strategy("BOGO", bogo_discount)
    
    print(f"   ✓ SAVE10: {percentage_discount.get_description()}")
    print(f"   ✓ SAVE25: {fixed_discount.get_description()}")
    print(f"   ✓ BOGO: {bogo_discount.get_description()}")
    
    print()
    
    # Create shopping carts
    print("4. SHOPPING CART CREATION:")
    
    # Guest cart
    guest_cart = cart_manager.create_cart(session_id="session_123")
    print(f"   ✓ Guest cart created: {guest_cart.cart_id[:8]}")
    
    # Registered user cart
    user_cart = cart_manager.create_cart(user_id="user_456")
    print(f"   ✓ User cart created: {user_cart.cart_id[:8]}")
    
    print()
    
    # Add items to carts
    print("5. ADDING ITEMS TO CARTS:")
    
    # Add laptop with 16GB RAM to guest cart
    laptop = products[0]
    success = cart_manager.add_to_cart(
        guest_cart.cart_id, 
        laptop.product_id, 
        1, 
        "ram_16gb"
    )
    print(f"   ✓ Added laptop (16GB RAM) to guest cart: {success}")
    
    # Add e-book to guest cart
    ebook = products[1]
    success = cart_manager.add_to_cart(
        guest_cart.cart_id,
        ebook.product_id,
        2
    )
    print(f"   ✓ Added 2 e-books to guest cart: {success}")
    
    # Add headphones to user cart
    headphones = products[2]
    success = cart_manager.add_to_cart(
        user_cart.cart_id,
        headphones.product_id,
        3,
        "color_black"
    )
    print(f"   ✓ Added 3 black headphones to user cart: {success}")
    
    # Add software license to user cart
    software = products[3]
    success = cart_manager.add_to_cart(
        user_cart.cart_id,
        software.product_id,
        1
    )
    print(f"   ✓ Added software license to user cart: {success}")
    
    print()
    
    # Update cart items
    print("6. UPDATING CART ITEMS:")
    
    # Update e-book quantity in guest cart
    success = cart_manager.update_cart_item(
        guest_cart.cart_id,
        ebook.product_id,
        3  # Change from 2 to 3
    )
    print(f"   ✓ Updated e-book quantity to 3: {success}")
    
    # Try to add more headphones than available
    success = cart_manager.update_cart_item(
        user_cart.cart_id,
        headphones.product_id,
        20,  # More than available stock
        "color_black"
    )
    print(f"   ✗ Tried to update headphones to 20 (insufficient stock): {success}")
    
    print()
    
    # Apply discounts
    print("7. APPLYING DISCOUNTS:")
    
    # Apply percentage discount to guest cart
    success = cart_manager.apply_coupon(guest_cart.cart_id, "SAVE10", "regular")
    print(f"   ✓ Applied SAVE10 to guest cart: {success}")
    
    # Apply BOGO discount to user cart (for headphones)
    success = cart_manager.apply_coupon(user_cart.cart_id, "BOGO", "vip")
    print(f"   ✓ Applied BOGO to user cart: {success}")
    
    # Try to apply fixed discount to user cart (should fail - not enough minimum)
    success = cart_manager.apply_coupon(user_cart.cart_id, "SAVE25", "regular")
    print(f"   ✗ Tried to apply SAVE25 to user cart (minimum not met): {success}")
    
    print()
    
    # Set shipping information
    print("8. SETTING SHIPPING INFORMATION:")
    
    # Set shipping for guest cart
    guest_address = Address("123 Main St", "Anytown", "CA", "USA", "12345")
    success = cart_manager.set_shipping_info(
        guest_cart.cart_id,
        guest_address,
        "express"
    )
    print(f"   ✓ Set express shipping for guest cart: {success}")
    
    # Set shipping for user cart
    user_address = Address("456 Oak Ave", "Somewhere", "NY", "USA", "67890")
    success = cart_manager.set_shipping_info(
        user_cart.cart_id,
        user_address,
        "standard"
    )
    print(f"   ✓ Set standard shipping for user cart: {success}")
    
    print()
    
    # Display cart summaries
    print("9. CART SUMMARIES:")
    
    guest_summary = guest_cart.get_cart_summary()
    print(f"   Guest Cart ({guest_summary['cart_id'][:8]}):")
    print(f"     Items: {guest_summary['item_count']} ({guest_summary['unique_item_count']} unique)")
    print(f"     Subtotal: ${guest_summary['subtotal']:.2f}")
    print(f"     Discount: -${guest_summary['discount_amount']:.2f}")
    print(f"     Shipping: ${guest_summary['shipping_cost']:.2f}")
    print(f"     Tax: ${guest_summary['tax_amount']:.2f}")
    print(f"     Total: ${guest_summary['total_amount']:.2f}")
    
    user_summary = user_cart.get_cart_summary()
    print(f"   User Cart ({user_summary['cart_id'][:8]}):")
    print(f"     Items: {user_summary['item_count']} ({user_summary['unique_item_count']} unique)")
    print(f"     Subtotal: ${user_summary['subtotal']:.2f}")
    print(f"     Discount: -${user_summary['discount_amount']:.2f}")
    print(f"     Shipping: ${user_summary['shipping_cost']:.2f}")
    print(f"     Tax: ${user_summary['tax_amount']:.2f}")
    print(f"     Total: ${user_summary['total_amount']:.2f}")
    
    print()
    
    # Display detailed cart contents
    print("10. DETAILED CART CONTENTS:")
    
    guest_details = guest_cart.get_detailed_cart()
    print(f"   Guest Cart Items:")
    for item_info in guest_details['items']:
        product_info = item_info['product']
        quantity = item_info['quantity']
        total = item_info['total_price']
        variation = item_info.get('variation_id', 'default')
        
        print(f"     - {product_info['name']} ({variation}) x{quantity}: ${total:.2f}")
        if product_info['is_on_sale']:
            print(f"       On Sale! Was ${product_info['base_price']:.2f}, now ${product_info['current_price']:.2f}")
    
    print(f"   Applied Discounts:")
    for discount in guest_details['applied_discounts']:
        print(f"     - {discount['description']}: -${discount['amount']:.2f}")
    
    print()
    
    # Test checkout process
    print("11. CHECKOUT PROCESS:")
    
    # Checkout guest cart
    billing_address = Address("123 Main St", "Anytown", "CA", "USA", "12345")
    order_id = cart_manager.checkout_cart(
        guest_cart.cart_id,
        PaymentMethod.CREDIT_CARD,
        billing_address
    )
    
    if order_id:
        print(f"   ✓ Guest cart checkout successful: Order {order_id[:8]}")
    else:
        print(f"   ✗ Guest cart checkout failed")
    
    # Try to checkout user cart (should work)
    order_id = cart_manager.checkout_cart(
        user_cart.cart_id,
        PaymentMethod.PAYPAL,
        user_address
    )
    
    if order_id:
        print(f"   ✓ User cart checkout successful: Order {order_id[:8]}")
    else:
        print(f"   ✗ User cart checkout failed")
    
    print()
    
    # Test cart abandonment and cleanup
    print("12. CART ABANDONMENT AND CLEANUP:")
    
    # Create abandoned cart
    abandoned_cart = cart_manager.create_cart(session_id="session_789")
    cart_manager.add_to_cart(abandoned_cart.cart_id, products[0].product_id, 1)
    
    # Mark as abandoned
    cart_manager.abandon_cart(abandoned_cart.cart_id)
    print(f"   ✓ Cart {abandoned_cart.cart_id[:8]} marked as abandoned")
    
    # Simulate expired carts
    expired_cart = cart_manager.create_cart(session_id="session_expired")
    cart_manager.add_to_cart(expired_cart.cart_id, products[1].product_id, 1)
    expired_cart.expires_at = datetime.now() - timedelta(hours=1)  # Force expiration
    
    # Cleanup expired carts
    cleaned_count = cart_manager.cleanup_expired_carts()
    print(f"   ✓ Cleaned up {cleaned_count} expired carts")
    
    print()
    
    # Show product inventory after transactions
    print("13. INVENTORY STATUS AFTER TRANSACTIONS:")
    
    for product in products[:3]:  # Show first 3 products
        info = product.get_product_info()
        print(f"   {info['name']}:")
        print(f"     Stock: {info['stock_quantity']}")
        print(f"     Purchases: {info['statistics']['purchase_count']}")
        print(f"     In Stock: {info['is_in_stock']}")
        
        if info['variations']:
            print(f"     Variations:")
            for var_id, var_info in info['variations'].items():
                print(f"       {var_info['name']} ({var_info['value']}): {var_info['stock_quantity']} in stock")
    
    print()
    
    # Show analytics
    print("14. CART ANALYTICS:")
    
    analytics = cart_manager.get_cart_analytics()
    
    print(f"   Total Carts Created: {analytics['total_carts_created']}")
    print(f"   Active Carts: {analytics['active_carts']}")
    print(f"   Abandoned Carts: {analytics['abandoned_carts']}")
    print(f"   Converted Carts: {analytics['converted_carts']}")
    print(f"   Conversion Rate: {analytics['conversion_rate']:.1f}%")
    print(f"   Total Cart Value: ${analytics['total_cart_value']:.2f}")
    print(f"   Average Cart Value: ${analytics['average_cart_value']:.2f}")
    print(f"   Total Revenue: ${analytics['total_revenue']:.2f}")
    print(f"   Products in Catalog: {analytics['products_in_catalog']}")
    print(f"   Shipping Options: {analytics['available_shipping_options']}")
    print(f"   Available Discounts: {analytics['available_discounts']}")
    
    print()
    print("=== ONLINE SHOPPING CART SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_shopping_cart_system()
