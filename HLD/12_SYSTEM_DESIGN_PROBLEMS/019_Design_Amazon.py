"""
AMAZON — E-Commerce Platform
==============================

FUNCTIONAL REQUIREMENTS:
- Product catalog: browse, search, filter by category/price/rating
- Shopping cart: add/remove items, save for later
- Checkout: address, payment, order confirmation
- Inventory management: real-time stock levels
- Order management: placed → confirmed → shipped → delivered
- Seller marketplace: third-party listings
- Reviews and ratings

NON-FUNCTIONAL REQUIREMENTS:
- 300 M customers, 1.5 M orders/day
- Product catalog: 350 M SKUs
- Search: < 100 ms p99 response
- Inventory: strongly consistent (prevent overselling)
- 99.99% checkout availability (revenue impact of downtime)

ARCHITECTURE:
  ┌──────────┐    ┌────────────┐    ┌──────────────┐
  │ Client   │───▶│ API GW     │───▶│ Product Svc  │──▶ DynamoDB
  └──────────┘    └────────────┘    └──────────────┘
                        │           ┌──────────────┐
                        ├──────────▶│ Search Svc   │──▶ Elasticsearch
                        │           └──────────────┘
                        │           ┌──────────────┐
                        ├──────────▶│ Cart Svc     │──▶ Redis
                        │           └──────────────┘
                        │           ┌──────────────┐
                        ├──────────▶│ Order Svc    │──▶ RDS (Postgres)
                        │           └──────────────┘
                        │           ┌──────────────┐
                        └──────────▶│ Inventory    │──▶ DynamoDB (strong)
                                    └──────────────┘

KEY DESIGN DECISIONS:
1. INVENTORY CONSISTENCY — use optimistic locking (version number) on each SKU.
   On checkout, atomic decrement with condition: quantity >= requested.
   If condition fails → return out-of-stock.  No distributed transaction needed
   if single-partition per SKU.

2. CART — stored in Redis with TTL; serialised as JSON blob keyed by user_id.
   Anonymous carts keyed by session_id, merged on login.

3. PRODUCT CATALOG — DynamoDB for K-V lookups (product_id → details);
   Elasticsearch for full-text search + faceted filtering.
   CDN-cached product pages (TTL 5 min).

4. ORDERS — relational DB for ACID transactions; order state machine.
   Outbox pattern for reliable event publishing to downstream services
   (fulfillment, payment, notification).

5. SEARCH — Elasticsearch with custom scoring:
   relevance score × popularity_boost × price_discount_factor
   Facets: category, price range, avg rating, Prime eligibility.

6. RECOMMENDATIONS — collaborative filtering offline (Spark batch);
   results written to DynamoDB for low-latency serving.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import threading
import math


# ---------------------------------------------------------------------------
# Enums and Data Models
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    PENDING = "pending"
    PAYMENT_CONFIRMED = "payment_confirmed"
    PREPARING = "preparing"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class FulfillmentType(Enum):
    FBA = "fulfilled_by_amazon"   # Amazon warehouse
    FBM = "fulfilled_by_merchant"  # Seller ships


@dataclass
class Category:
    category_id: str
    name: str
    parent_id: Optional[str] = None


@dataclass
class Product:
    product_id: str
    seller_id: str
    title: str
    description: str
    category_id: str
    price_cents: int            # price in cents
    list_price_cents: int       # original price (for discount display)
    brand: str = ""
    images: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)  # size, colour, etc.
    rating: float = 0.0
    review_count: int = 0
    fulfillment: FulfillmentType = FulfillmentType.FBA
    is_prime: bool = True
    created_at: float = field(default_factory=time.time)

    @property
    def discount_pct(self) -> int:
        if self.list_price_cents <= self.price_cents:
            return 0
        return int(100 * (1 - self.price_cents / self.list_price_cents))

    @property
    def price_display(self) -> str:
        return f"${self.price_cents / 100:.2f}"


# ---------------------------------------------------------------------------
# Inventory Service — strongly consistent
# ---------------------------------------------------------------------------

@dataclass
class InventoryRecord:
    product_id: str
    quantity: int
    reserved: int = 0        # held by pending orders
    version: int = 0         # optimistic lock

    @property
    def available(self) -> int:
        return max(0, self.quantity - self.reserved)


class InventoryService:
    """
    DynamoDB-backed strong consistency via conditional writes.
    Simulated here with a threading.Lock per product.
    """

    def __init__(self):
        self._records: Dict[str, InventoryRecord] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

    def set_stock(self, product_id: str, quantity: int) -> InventoryRecord:
        with self._locks[product_id]:
            rec = self._records.get(product_id)
            if rec:
                rec.quantity = quantity
                rec.version += 1
            else:
                rec = InventoryRecord(product_id, quantity)
                self._records[product_id] = rec
            return rec

    def get(self, product_id: str) -> Optional[InventoryRecord]:
        return self._records.get(product_id)

    def reserve(self, product_id: str, qty: int) -> bool:
        """Atomically reserve qty units. Returns False if insufficient stock."""
        with self._locks[product_id]:
            rec = self._records.get(product_id)
            if not rec or rec.available < qty:
                return False
            rec.reserved += qty
            rec.version += 1
            return True

    def confirm(self, product_id: str, qty: int) -> bool:
        """Deduct from quantity (reserved → sold)."""
        with self._locks[product_id]:
            rec = self._records.get(product_id)
            if not rec or rec.reserved < qty:
                return False
            rec.quantity -= qty
            rec.reserved -= qty
            rec.version += 1
            return True

    def release(self, product_id: str, qty: int) -> None:
        """Release reservation (order cancelled)."""
        with self._locks[product_id]:
            rec = self._records.get(product_id)
            if rec:
                rec.reserved = max(0, rec.reserved - qty)
                rec.version += 1

    def bulk_available(self, items: List[Tuple[str, int]]) -> bool:
        """Check all items available (for cart validation)."""
        return all(
            (rec := self._records.get(pid)) and rec.available >= qty
            for pid, qty in items
        )


# ---------------------------------------------------------------------------
# Product Catalog
# ---------------------------------------------------------------------------

class ProductCatalog:
    def __init__(self):
        self._products: Dict[str, Product] = {}
        self._by_category: Dict[str, List[str]] = defaultdict(list)

    def add_product(self, product: Product) -> Product:
        self._products[product.product_id] = product
        self._by_category[product.category_id].append(product.product_id)
        return product

    def get(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def list_by_category(self, category_id: str,
                          sort_by: str = "relevance",
                          limit: int = 20) -> List[Product]:
        pids = self._by_category.get(category_id, [])
        products = [self._products[pid] for pid in pids if pid in self._products]
        if sort_by == "price_asc":
            products.sort(key=lambda p: p.price_cents)
        elif sort_by == "price_desc":
            products.sort(key=lambda p: p.price_cents, reverse=True)
        elif sort_by == "rating":
            products.sort(key=lambda p: p.rating, reverse=True)
        else:
            # Relevance: rating × log(review_count + 1)
            products.sort(key=lambda p: p.rating * math.log1p(p.review_count), reverse=True)
        return products[:limit]

    def update_rating(self, product_id: str, new_rating: float, review_count: int):
        p = self._products.get(product_id)
        if p:
            p.rating = round(new_rating, 1)
            p.review_count = review_count


# ---------------------------------------------------------------------------
# Search Service (Elasticsearch simulation)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    product: Product
    score: float


class SearchService:
    def __init__(self, catalog: ProductCatalog):
        self._catalog = catalog
        # term → set of product_ids
        self._index: Dict[str, set] = defaultdict(set)

    def index_product(self, product: Product) -> None:
        tokens = self._tokenize(product.title + " " + product.description + " " + product.brand)
        for token in tokens:
            self._index[token].add(product.product_id)

    def search(self, query: str, category_id: Optional[str] = None,
               min_price: int = 0, max_price: int = 10_000_00,
               min_rating: float = 0.0, prime_only: bool = False,
               sort_by: str = "relevance", limit: int = 20) -> List[SearchResult]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        candidates = self._index.get(tokens[0], set()).copy()
        for t in tokens[1:]:
            candidates |= self._index.get(t, set())  # OR semantics

        results = []
        for pid in candidates:
            p = self._catalog.get(pid)
            if not p:
                continue
            if category_id and p.category_id != category_id:
                continue
            if not (min_price <= p.price_cents <= max_price):
                continue
            if p.rating < min_rating:
                continue
            if prime_only and not p.is_prime:
                continue

            # BM25-style simplified scoring
            term_matches = sum(1 for t in tokens if pid in self._index.get(t, set()))
            score = (term_matches / len(tokens)) * (1 + p.rating / 5)
            results.append(SearchResult(p, score))

        if sort_by == "price_asc":
            results.sort(key=lambda r: r.product.price_cents)
        elif sort_by == "price_desc":
            results.sort(key=lambda r: r.product.price_cents, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda r: r.product.rating, reverse=True)
        else:
            results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower().strip(".,!?-") for w in text.split() if len(w) > 2]


# ---------------------------------------------------------------------------
# Cart Service (Redis-backed)
# ---------------------------------------------------------------------------

@dataclass
class CartItem:
    product_id: str
    quantity: int
    price_cents: int     # snapshot price at time of add
    title: str


class CartService:
    def __init__(self, catalog: ProductCatalog, inventory: InventoryService):
        self._carts: Dict[str, Dict[str, CartItem]] = defaultdict(dict)
        self._catalog = catalog
        self._inventory = inventory

    def add_item(self, user_id: str, product_id: str, quantity: int = 1) -> bool:
        product = self._catalog.get(product_id)
        if not product:
            return False
        inv = self._inventory.get(product_id)
        if not inv or inv.available < quantity:
            return False

        cart = self._carts[user_id]
        if product_id in cart:
            cart[product_id].quantity += quantity
        else:
            cart[product_id] = CartItem(product_id, quantity,
                                         product.price_cents, product.title)
        return True

    def remove_item(self, user_id: str, product_id: str) -> bool:
        cart = self._carts.get(user_id, {})
        return cart.pop(product_id, None) is not None

    def update_quantity(self, user_id: str, product_id: str, quantity: int) -> bool:
        cart = self._carts.get(user_id, {})
        if product_id not in cart:
            return False
        if quantity <= 0:
            del cart[product_id]
        else:
            inv = self._inventory.get(product_id)
            if not inv or inv.available < quantity:
                return False
            cart[product_id].quantity = quantity
        return True

    def get_cart(self, user_id: str) -> List[CartItem]:
        return list(self._carts.get(user_id, {}).values())

    def cart_total_cents(self, user_id: str) -> int:
        return sum(i.price_cents * i.quantity for i in self.get_cart(user_id))

    def clear_cart(self, user_id: str) -> None:
        self._carts.pop(user_id, None)


# ---------------------------------------------------------------------------
# Order Service
# ---------------------------------------------------------------------------

@dataclass
class OrderItem:
    product_id: str
    title: str
    quantity: int
    unit_price_cents: int


@dataclass
class Order:
    order_id: str
    user_id: str
    items: List[OrderItem]
    shipping_address: str
    status: OrderStatus = OrderStatus.PENDING
    payment_method: str = "card"
    total_cents: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None

    def total_display(self) -> str:
        return f"${self.total_cents / 100:.2f}"


class OrderService:
    """Relational DB (PostgreSQL) with outbox pattern for event publishing."""

    def __init__(self, inventory: InventoryService, cart: CartService):
        self._orders: Dict[str, Order] = {}
        self._user_orders: Dict[str, List[str]] = defaultdict(list)
        self._inventory = inventory
        self._cart = cart
        self._event_outbox: List[Dict] = []

    def place_order(self, user_id: str, shipping_address: str,
                    payment_method: str = "card") -> Optional[Order]:
        cart_items = self._cart.get_cart(user_id)
        if not cart_items:
            return None

        # Check and reserve inventory for all items atomically
        reservations = [(item.product_id, item.quantity) for item in cart_items]
        for product_id, qty in reservations:
            if not self._inventory.reserve(product_id, qty):
                # Release already-reserved
                for pid, q in reservations[:reservations.index((product_id, qty))]:
                    self._inventory.release(pid, q)
                return None  # Out of stock

        total = sum(i.price_cents * i.quantity for i in cart_items)
        order = Order(
            order_id=str(uuid.uuid4())[:12],
            user_id=user_id,
            items=[OrderItem(i.product_id, i.title, i.quantity, i.price_cents)
                   for i in cart_items],
            shipping_address=shipping_address,
            payment_method=payment_method,
            total_cents=total,
        )
        self._orders[order.order_id] = order
        self._user_orders[user_id].append(order.order_id)
        self._cart.clear_cart(user_id)

        # Outbox event
        self._emit("order.placed", order)
        return order

    def confirm_payment(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return False
        order.status = OrderStatus.PAYMENT_CONFIRMED
        order.updated_at = time.time()
        # Convert reservation to confirmed deduction
        for item in order.items:
            self._inventory.confirm(item.product_id, item.quantity)
        self._emit("order.payment_confirmed", order)
        return True

    def ship_order(self, order_id: str, tracking: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.PAYMENT_CONFIRMED:
            return False
        order.status = OrderStatus.SHIPPED
        order.tracking_number = tracking
        order.updated_at = time.time()
        self._emit("order.shipped", order)
        return True

    def deliver_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.SHIPPED:
            return False
        order.status = OrderStatus.DELIVERED
        order.updated_at = time.time()
        self._emit("order.delivered", order)
        return True

    def cancel_order(self, order_id: str, user_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.user_id != user_id:
            return False
        if order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            return False  # Too late to cancel
        if order.status == OrderStatus.PAYMENT_CONFIRMED:
            for item in order.items:
                self._inventory.release(item.product_id, item.quantity)
        order.status = OrderStatus.CANCELLED
        order.updated_at = time.time()
        self._emit("order.cancelled", order)
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def user_orders(self, user_id: str) -> List[Order]:
        return [self._orders[oid] for oid in self._user_orders.get(user_id, [])
                if oid in self._orders]

    def _emit(self, event_type: str, order: Order):
        self._event_outbox.append({
            "event": event_type,
            "order_id": order.order_id,
            "user_id": order.user_id,
            "ts": time.time(),
        })


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

@dataclass
class Review:
    review_id: str
    product_id: str
    user_id: str
    rating: int          # 1-5
    title: str
    body: str
    verified_purchase: bool = False
    helpful_votes: int = 0
    created_at: float = field(default_factory=time.time)


class ReviewService:
    def __init__(self, catalog: ProductCatalog):
        self._reviews: Dict[str, List[Review]] = defaultdict(list)
        self._catalog = catalog

    def submit_review(self, product_id: str, user_id: str, rating: int,
                      title: str, body: str, verified: bool = False) -> Review:
        review = Review(
            review_id=str(uuid.uuid4())[:8],
            product_id=product_id,
            user_id=user_id,
            rating=max(1, min(5, rating)),
            title=title,
            body=body,
            verified_purchase=verified,
        )
        self._reviews[product_id].append(review)
        self._update_product_rating(product_id)
        return review

    def _update_product_rating(self, product_id: str):
        reviews = self._reviews[product_id]
        if reviews:
            avg = sum(r.rating for r in reviews) / len(reviews)
            self._catalog.update_rating(product_id, avg, len(reviews))

    def get_reviews(self, product_id: str, sort_by: str = "helpful",
                    limit: int = 10) -> List[Review]:
        reviews = list(self._reviews.get(product_id, []))
        if sort_by == "helpful":
            reviews.sort(key=lambda r: r.helpful_votes, reverse=True)
        elif sort_by == "recent":
            reviews.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "rating_high":
            reviews.sort(key=lambda r: r.rating, reverse=True)
        return reviews[:limit]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_catalog_and_search():
    print("\n=== 1. Product Catalog & Search ===")
    catalog = ProductCatalog()
    search = SearchService(catalog)

    products = [
        Product("p001", "seller_a", "Wireless Bluetooth Headphones", "Premium sound quality",
                "cat_electronics", 4999, 7999, "SoundMax", rating=4.5, review_count=1200),
        Product("p002", "seller_b", "USB-C Charging Cable 6ft", "Fast charging cable",
                "cat_electronics", 999, 1499, "ChargePlus", rating=4.2, review_count=300),
        Product("p003", "seller_c", "Running Shoes Men Size 10", "Lightweight running shoes",
                "cat_shoes", 8999, 12000, "SpeedRun", rating=4.7, review_count=850),
        Product("p004", "seller_a", "Bluetooth Speaker Portable", "Waterproof speaker",
                "cat_electronics", 3499, 4999, "SoundMax", rating=4.3, review_count=450),
    ]
    for p in products:
        catalog.add_product(p)
        search.index_product(p)

    print(f"Catalog size: {len(products)} products")

    # Search
    results = search.search("bluetooth", category_id="cat_electronics")
    print(f"\nSearch 'bluetooth' in electronics: {len(results)} results")
    for r in results:
        print(f"  [{r.score:.2f}] {r.product.title} — {r.product.price_display} "
              f"({r.product.discount_pct}% off)")

    # Browse category sorted by rating
    elec = catalog.list_by_category("cat_electronics", sort_by="rating")
    print(f"\nElectronics by rating:")
    for p in elec:
        print(f"  ★{p.rating} {p.title} — {p.price_display}")

    return catalog, search


def demonstrate_2_cart_and_inventory():
    print("\n=== 2. Cart & Inventory ===")
    catalog, search = demonstrate_1_catalog_and_search()
    inventory = InventoryService()

    inventory.set_stock("p001", 50)
    inventory.set_stock("p002", 200)
    inventory.set_stock("p003", 3)  # Low stock

    cart = CartService(catalog, inventory)

    cart.add_item("user_alice", "p001", 1)
    cart.add_item("user_alice", "p002", 2)
    added_low = cart.add_item("user_alice", "p003", 5)  # Only 3 in stock

    print(f"\nAlice's cart:")
    for item in cart.get_cart("user_alice"):
        print(f"  {item.title} × {item.quantity} @ ${item.price_cents/100:.2f}")
    print(f"Cart total: ${cart.cart_total_cents('user_alice')/100:.2f}")
    print(f"Added 5 running shoes (only 3 in stock): {added_low}")

    return catalog, inventory, cart


def demonstrate_3_order_lifecycle():
    print("\n=== 3. Order Lifecycle ===")
    catalog, inventory, cart = demonstrate_2_cart_and_inventory()
    order_svc = OrderService(inventory, cart)

    # Re-populate cart after demonstration 2 cleared it
    cart.add_item("user_alice", "p001", 1)
    cart.add_item("user_alice", "p002", 2)

    order = order_svc.place_order("user_alice", "123 Main St, Seattle WA 98101")
    if order:
        print(f"\nOrder placed: {order.order_id}")
        print(f"  Status: {order.status.value}")
        print(f"  Total: {order.total_display()}")
        print(f"  Items: {len(order.items)}")

        # Check inventory reserved
        inv = inventory.get("p001")
        print(f"\nInventory p001: qty={inv.quantity}, reserved={inv.reserved}, "
              f"available={inv.available}")

        # Confirm payment
        order_svc.confirm_payment(order.order_id)
        print(f"After payment: status={order.status.value}")

        inv = inventory.get("p001")
        print(f"Inventory p001 after confirm: qty={inv.quantity}, reserved={inv.reserved}")

        # Ship
        order_svc.ship_order(order.order_id, "TRACK123456")
        print(f"After ship: status={order.status.value}, tracking={order.tracking_number}")

        # Deliver
        order_svc.deliver_order(order.order_id)
        print(f"Final status: {order.status.value}")
    else:
        print("Order placement failed!")

    return catalog, inventory


def demonstrate_4_reviews(catalog, inventory):
    print("\n=== 4. Reviews & Ratings ===")
    reviews = ReviewService(catalog)

    reviews.submit_review("p001", "user_alice", 5, "Amazing headphones!",
                           "Best purchase ever", verified=True)
    reviews.submit_review("p001", "user_bob", 4, "Great sound",
                           "Worth the price, minor comfort issues", verified=True)
    reviews.submit_review("p001", "user_carol", 3, "Decent but not excellent",
                           "Battery life shorter than advertised", verified=False)

    p = catalog.get("p001")
    print(f"Product: {p.title}")
    print(f"Updated rating: ★{p.rating} ({p.review_count} reviews)")

    top_reviews = reviews.get_reviews("p001", sort_by="rating_high")
    print(f"\nTop reviews:")
    for r in top_reviews:
        print(f"  ★{r.rating} [{r.user_id}] '{r.title}' (verified={r.verified_purchase})")


def demonstrate_5_inventory_concurrency():
    print("\n=== 5. Inventory Concurrent Reservation (Prevent Overselling) ===")
    inventory = InventoryService()
    inventory.set_stock("p_limited", 2)  # Only 2 units

    results = []
    lock = threading.Lock()

    def try_reserve(user_id: str):
        success = inventory.reserve("p_limited", 1)
        with lock:
            results.append((user_id, success))

    threads = [threading.Thread(target=try_reserve, args=(f"user_{i}",))
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successful = [(u, s) for u, s in results if s]
    failed = [(u, s) for u, s in results if not s]
    print(f"5 concurrent reservation attempts for 2 units:")
    print(f"  Successful: {len(successful)} → {[u for u,_ in successful]}")
    print(f"  Failed (out of stock): {len(failed)}")

    inv = inventory.get("p_limited")
    print(f"Inventory: qty={inv.quantity}, reserved={inv.reserved}, available={inv.available}")


if __name__ == "__main__":
    demonstrate_1_catalog_and_search()
    demonstrate_2_cart_and_inventory()
    catalog, inventory = demonstrate_3_order_lifecycle()
    demonstrate_4_reviews(catalog, inventory)
    demonstrate_5_inventory_concurrency()
