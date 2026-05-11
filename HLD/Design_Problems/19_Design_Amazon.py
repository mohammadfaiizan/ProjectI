"""
Design Amazon — Python Simulation
====================================
Simulates core Amazon e-commerce mechanics:
  - Product catalog with Elasticsearch-like search and facets
  - Inventory management with optimistic locking (version-based)
  - Shopping cart with Redis-like TTL store
  - Order state machine (placed -> delivered)
  - Payment processing with idempotency
  - Recommendation engine (collaborative filtering)
  - Pricing engine (base price + dynamic adjustments + coupons)
  - Fulfillment routing (nearest warehouse selection)
"""

import uuid
import time
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import Enum
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    PLACED = "placed"
    PAYMENT_PENDING = "payment_pending"
    PAID = "paid"
    FULFILLMENT_ASSIGNED = "fulfillment_assigned"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    base_price: float
    rating: float = 4.0
    review_count: int = 0
    seller_id: str = ""
    is_active: bool = True
    tags: list = field(default_factory=list)


@dataclass
class InventoryItem:
    product_id: str
    warehouse_id: str
    quantity: int
    reserved: int = 0
    version: int = 0    # For optimistic locking

    def available(self) -> int:
        return max(0, self.quantity - self.reserved)


@dataclass
class CartItem:
    product_id: str
    quantity: int
    unit_price: float
    added_at: float = field(default_factory=time.time)


@dataclass
class Order:
    order_id: str
    user_id: str
    items: list[CartItem]
    status: OrderStatus
    subtotal: float
    tax: float
    shipping_cost: float
    total: float
    warehouse_id: Optional[str] = None
    payment_id: Optional[str] = None
    placed_at: float = field(default_factory=time.time)
    status_history: list = field(default_factory=list)


@dataclass
class Warehouse:
    warehouse_id: str
    name: str
    latitude: float
    longitude: float


# ---------------------------------------------------------------------------
# Inventory Manager (Optimistic Locking)
# ---------------------------------------------------------------------------

class InventoryManager:
    """
    Manages stock with version-based optimistic locking.
    In production: backed by PostgreSQL + Redis atomic DECR for fast path.
    """

    def __init__(self):
        # (product_id, warehouse_id) -> InventoryItem
        self._inventory: dict[tuple, InventoryItem] = {}
        self._contention_stats = {"attempts": 0, "conflicts": 0, "success": 0}

    def add_stock(self, product_id: str, warehouse_id: str, quantity: int):
        key = (product_id, warehouse_id)
        if key in self._inventory:
            self._inventory[key].quantity += quantity
        else:
            self._inventory[key] = InventoryItem(
                product_id=product_id,
                warehouse_id=warehouse_id,
                quantity=quantity
            )

    def check_availability(self, product_id: str) -> dict:
        """Check total available stock across all warehouses."""
        total = 0
        warehouses = []
        for (pid, wid), item in self._inventory.items():
            if pid == product_id:
                avail = item.available()
                total += avail
                if avail > 0:
                    warehouses.append({"warehouse_id": wid, "available": avail})
        return {"product_id": product_id, "total_available": total, "warehouses": warehouses}

    def reserve_stock(self, product_id: str, warehouse_id: str,
                      quantity: int, max_retries: int = 3) -> bool:
        """
        Optimistic locking: read version, reserve, update WHERE version = read_version.
        Retry on conflict (simulated).
        """
        key = (product_id, warehouse_id)
        for attempt in range(max_retries):
            self._contention_stats["attempts"] += 1
            if key not in self._inventory:
                return False

            item = self._inventory[key]
            read_version = item.version
            available = item.available()

            if available < quantity:
                return False   # Out of stock

            # Simulate optimistic locking: check version hasn't changed
            # In a real DB: UPDATE inventory SET reserved=reserved+qty, version=version+1
            #               WHERE product_id=? AND version=read_version
            if item.version == read_version:
                item.reserved += quantity
                item.version += 1
                self._contention_stats["success"] += 1
                return True
            else:
                # Version conflict (concurrent modification) — retry
                self._contention_stats["conflicts"] += 1
                time.sleep(0.001 * (2 ** attempt))   # Exponential backoff

        return False

    def confirm_sale(self, product_id: str, warehouse_id: str, quantity: int):
        """Deduct from stock after successful payment."""
        key = (product_id, warehouse_id)
        if key in self._inventory:
            item = self._inventory[key]
            item.quantity -= quantity
            item.reserved = max(0, item.reserved - quantity)

    def release_reservation(self, product_id: str, warehouse_id: str, quantity: int):
        """Release reservation on payment failure or cancellation."""
        key = (product_id, warehouse_id)
        if key in self._inventory:
            self._inventory[key].reserved = max(
                0, self._inventory[key].reserved - quantity
            )

    def get_stats(self) -> dict:
        return self._contention_stats


# ---------------------------------------------------------------------------
# Shopping Cart (Redis-like TTL Store)
# ---------------------------------------------------------------------------

class ShoppingCart:
    """Redis-backed cart with TTL. Per user."""

    CART_TTL = 7 * 24 * 3600   # 7 days

    def __init__(self):
        # user_id -> (items_dict, expiry_timestamp)
        self._carts: dict[str, tuple[dict[str, CartItem], float]] = {}

    def add_item(self, user_id: str, product_id: str,
                 quantity: int, unit_price: float):
        self._ensure_cart(user_id)
        items, _ = self._carts[user_id]
        if product_id in items:
            items[product_id].quantity += quantity
        else:
            items[product_id] = CartItem(
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price
            )
        self._refresh_ttl(user_id)

    def remove_item(self, user_id: str, product_id: str):
        self._ensure_cart(user_id)
        items, _ = self._carts[user_id]
        items.pop(product_id, None)

    def get_cart(self, user_id: str) -> list[CartItem]:
        if user_id not in self._carts:
            return []
        items, expiry = self._carts[user_id]
        if time.time() > expiry:
            del self._carts[user_id]
            return []
        return list(items.values())

    def clear_cart(self, user_id: str):
        self._carts.pop(user_id, None)

    def get_subtotal(self, user_id: str) -> float:
        return sum(i.quantity * i.unit_price for i in self.get_cart(user_id))

    def _ensure_cart(self, user_id: str):
        if user_id not in self._carts:
            self._carts[user_id] = ({}, time.time() + self.CART_TTL)

    def _refresh_ttl(self, user_id: str):
        if user_id in self._carts:
            items, _ = self._carts[user_id]
            self._carts[user_id] = (items, time.time() + self.CART_TTL)


# ---------------------------------------------------------------------------
# Order State Machine
# ---------------------------------------------------------------------------

class OrderStateMachine:
    """Manages order lifecycle transitions."""

    TRANSITIONS = {
        OrderStatus.PLACED: [OrderStatus.PAYMENT_PENDING, OrderStatus.CANCELLED],
        OrderStatus.PAYMENT_PENDING: [OrderStatus.PAID, OrderStatus.CANCELLED],
        OrderStatus.PAID: [OrderStatus.FULFILLMENT_ASSIGNED, OrderStatus.CANCELLED],
        OrderStatus.FULFILLMENT_ASSIGNED: [OrderStatus.SHIPPED],
        OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
        OrderStatus.DELIVERED: [OrderStatus.RETURNED],
        OrderStatus.CANCELLED: [],
        OrderStatus.RETURNED: []
    }

    def __init__(self):
        self._orders: dict[str, Order] = {}

    def create_order(self, user_id: str, items: list[CartItem],
                     subtotal: float, tax_rate: float = 0.08) -> Order:
        tax = subtotal * tax_rate
        shipping = 5.99 if subtotal < 35 else 0.0
        order = Order(
            order_id=str(uuid.uuid4()),
            user_id=user_id,
            items=items,
            status=OrderStatus.PLACED,
            subtotal=subtotal,
            tax=tax,
            shipping_cost=shipping,
            total=subtotal + tax + shipping
        )
        order.status_history.append({
            "status": order.status.value,
            "timestamp": time.time()
        })
        self._orders[order.order_id] = order
        return order

    def transition(self, order_id: str, new_status: OrderStatus,
                   metadata: dict = None) -> bool:
        order = self._orders.get(order_id)
        if not order:
            return False
        allowed = self.TRANSITIONS.get(order.status, [])
        if new_status not in allowed:
            return False
        order.status = new_status
        order.status_history.append({
            "status": new_status.value,
            "timestamp": time.time(),
            **(metadata or {})
        })
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_user_orders(self, user_id: str) -> list[Order]:
        return [o for o in self._orders.values() if o.user_id == user_id]


# ---------------------------------------------------------------------------
# Recommendation Engine (Collaborative Filtering)
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    User-based collaborative filtering.
    Pre-computed in production (nightly Spark job).
    """

    def __init__(self):
        # user_id -> set of product_ids purchased/viewed
        self._user_interactions: dict[str, set[str]] = defaultdict(set)
        # product_id -> set of user_ids who bought together
        self._co_purchase: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_interaction(self, user_id: str, product_id: str):
        self._user_interactions[user_id].add(product_id)

    def record_purchase(self, user_id: str, product_ids: list[str]):
        for pid in product_ids:
            self.record_interaction(user_id, pid)
        # Co-purchase tracking
        for i, pid_a in enumerate(product_ids):
            for pid_b in product_ids[i+1:]:
                self._co_purchase[pid_a][pid_b] += 1
                self._co_purchase[pid_b][pid_a] += 1

    def get_recommendations(self, user_id: str, n: int = 5) -> list[str]:
        """Find users most similar to target; recommend their items."""
        user_items = self._user_interactions.get(user_id, set())
        if not user_items:
            return []

        # Compute cosine similarity with other users
        similarities = {}
        for other_id, other_items in self._user_interactions.items():
            if other_id == user_id:
                continue
            intersection = len(user_items & other_items)
            if intersection == 0:
                continue
            similarity = intersection / math.sqrt(len(user_items) * len(other_items))
            similarities[other_id] = similarity

        # Get items from top-K similar users not yet seen by target user
        top_users = sorted(similarities, key=similarities.get, reverse=True)[:10]
        candidate_scores: dict[str, float] = defaultdict(float)
        for similar_user in top_users:
            for item in self._user_interactions[similar_user]:
                if item not in user_items:
                    candidate_scores[item] += similarities[similar_user]

        return sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:n]

    def get_frequently_bought_together(self, product_id: str, n: int = 4) -> list[str]:
        """Items frequently co-purchased with the given product."""
        co = self._co_purchase.get(product_id, {})
        return sorted(co, key=co.get, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Search Engine (Inverted Index + Facets)
# ---------------------------------------------------------------------------

class SearchEngine:
    """Simulated Elasticsearch with inverted index and facet filtering."""

    def __init__(self):
        self._products: dict[str, Product] = {}
        self._inverted_index: dict[str, set[str]] = defaultdict(set)

    def index_product(self, product: Product):
        self._products[product.product_id] = product
        tokens = self._tokenize(f"{product.title} {product.brand} {' '.join(product.tags)}")
        for token in tokens:
            self._inverted_index[token].add(product.product_id)

    def search(self, query: str, category: str = None, min_price: float = None,
               max_price: float = None, min_rating: float = None,
               sort_by: str = "relevance") -> dict:
        # Candidate set from inverted index
        tokens = self._tokenize(query)
        if not tokens:
            candidates = set(self._products.keys())
        else:
            candidates = set()
            for token in tokens:
                candidates |= self._inverted_index.get(token, set())

        # Apply filters
        results = []
        facets = {"categories": defaultdict(int), "brands": defaultdict(int)}

        for pid in candidates:
            p = self._products[pid]
            if not p.is_active:
                continue
            if category and p.category.lower() != category.lower():
                continue
            if min_price and p.base_price < min_price:
                continue
            if max_price and p.base_price > max_price:
                continue
            if min_rating and p.rating < min_rating:
                continue
            results.append(p)
            facets["categories"][p.category] += 1
            facets["brands"][p.brand] += 1

        # Sort
        if sort_by == "price_asc":
            results.sort(key=lambda p: p.base_price)
        elif sort_by == "price_desc":
            results.sort(key=lambda p: p.base_price, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda p: p.rating, reverse=True)

        return {
            "results": results,
            "total": len(results),
            "facets": {k: dict(v) for k, v in facets.items()}
        }

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower() for w in text.split() if len(w) > 2]


# ---------------------------------------------------------------------------
# Pricing Engine
# ---------------------------------------------------------------------------

class PricingEngine:
    """Base price + dynamic adjustments + coupon application."""

    def __init__(self):
        self._demand_multipliers: dict[str, float] = {}
        self._coupons: dict[str, dict] = {}

    def set_demand_multiplier(self, product_id: str, multiplier: float):
        self._demand_multipliers[product_id] = multiplier

    def add_coupon(self, code: str, discount_pct: float, min_order: float = 0):
        self._coupons[code.upper()] = {
            "discount_pct": discount_pct,
            "min_order": min_order,
            "used": 0
        }

    def calculate_price(self, product: Product, quantity: int = 1,
                        coupon_code: str = None, is_prime: bool = False) -> dict:
        base = product.base_price
        demand_mult = self._demand_multipliers.get(product.product_id, 1.0)
        adjusted_price = base * demand_mult

        # Prime discount (5% for eligible items)
        prime_discount = adjusted_price * 0.05 if is_prime else 0

        unit_price = adjusted_price - prime_discount
        subtotal = unit_price * quantity

        # Coupon application
        coupon_discount = 0
        coupon_applied = False
        if coupon_code:
            code = coupon_code.upper()
            coupon = self._coupons.get(code)
            if coupon and subtotal >= coupon["min_order"]:
                coupon_discount = subtotal * coupon["discount_pct"]
                coupon_applied = True
                coupon["used"] += 1

        final_price = subtotal - coupon_discount

        return {
            "base_price": base,
            "demand_multiplier": demand_mult,
            "unit_price": round(unit_price, 2),
            "subtotal": round(subtotal, 2),
            "prime_discount": round(prime_discount * quantity, 2),
            "coupon_discount": round(coupon_discount, 2),
            "final_price": round(final_price, 2),
            "coupon_applied": coupon_applied
        }


# ---------------------------------------------------------------------------
# Fulfillment Router
# ---------------------------------------------------------------------------

class FulfillmentRouter:
    """Select optimal warehouse based on distance and stock availability."""

    def __init__(self, warehouses: list[Warehouse], inventory: InventoryManager):
        self._warehouses = {w.warehouse_id: w for w in warehouses}
        self._inventory = inventory

    def find_warehouse(self, product_id: str, quantity: int,
                       delivery_lat: float, delivery_lon: float) -> Optional[str]:
        """Return best warehouse_id for the order."""
        availability = self._inventory.check_availability(product_id)
        candidates = [
            w for w in availability["warehouses"] if w["available"] >= quantity
        ]

        if not candidates:
            return None

        scored = []
        for c in candidates:
            w = self._warehouses.get(c["warehouse_id"])
            if w:
                dist = self._haversine(delivery_lat, delivery_lon, w.latitude, w.longitude)
                scored.append((dist, c["warehouse_id"]))

        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else None

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Distance in km between two lat/lon coordinates."""
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ---------------------------------------------------------------------------
# Main Amazon System
# ---------------------------------------------------------------------------

class AmazonSystem:
    def __init__(self, warehouses: list[Warehouse]):
        self.inventory = InventoryManager()
        self.cart = ShoppingCart()
        self.order_machine = OrderStateMachine()
        self.recommender = RecommendationEngine()
        self.search_engine = SearchEngine()
        self.pricing = PricingEngine()
        self.fulfillment = FulfillmentRouter(warehouses, self.inventory)
        self._products: dict[str, Product] = {}
        self._payments_processed: dict[str, str] = {}   # idempotency_key -> payment_id

    def add_product(self, product: Product):
        self._products[product.product_id] = product
        self.search_engine.index_product(product)

    def search_products(self, **kwargs) -> dict:
        return self.search_engine.search(**kwargs)

    def add_to_cart(self, user_id: str, product_id: str,
                    quantity: int, coupon_code: str = None) -> dict:
        product = self._products.get(product_id)
        if not product:
            return {"error": "product_not_found"}
        pricing = self.pricing.calculate_price(product, quantity, coupon_code)
        self.cart.add_item(user_id, product_id, quantity, pricing["unit_price"])
        return {
            "product_id": product_id,
            "quantity": quantity,
            "unit_price": pricing["unit_price"],
            "cart_subtotal": self.cart.get_subtotal(user_id)
        }

    def place_order(self, user_id: str, delivery_lat: float,
                    delivery_lon: float) -> dict:
        cart_items = self.cart.get_cart(user_id)
        if not cart_items:
            return {"error": "empty_cart"}

        # Reserve inventory for all items
        reserved = []
        for item in cart_items:
            warehouse_id = self.fulfillment.find_warehouse(
                item.product_id, item.quantity, delivery_lat, delivery_lon
            )
            if not warehouse_id:
                # Release previous reservations (saga compensation)
                for r in reserved:
                    self.inventory.release_reservation(
                        r["product_id"], r["warehouse_id"], r["quantity"]
                    )
                return {"error": f"out_of_stock: {item.product_id}"}

            success = self.inventory.reserve_stock(
                item.product_id, warehouse_id, item.quantity
            )
            if not success:
                for r in reserved:
                    self.inventory.release_reservation(
                        r["product_id"], r["warehouse_id"], r["quantity"]
                    )
                return {"error": f"reservation_failed: {item.product_id}"}
            reserved.append({"product_id": item.product_id,
                              "warehouse_id": warehouse_id,
                              "quantity": item.quantity})

        subtotal = self.cart.get_subtotal(user_id)
        order = self.order_machine.create_order(user_id, cart_items, subtotal)
        self.order_machine.transition(order.order_id, OrderStatus.PAYMENT_PENDING)

        # Assign warehouse
        if reserved:
            order.warehouse_id = reserved[0]["warehouse_id"]

        self.cart.clear_cart(user_id)
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "total": order.total,
            "warehouse_id": order.warehouse_id
        }

    def process_payment(self, order_id: str, idempotency_key: str) -> dict:
        if idempotency_key in self._payments_processed:
            return {"status": "already_processed",
                    "payment_id": self._payments_processed[idempotency_key]}

        order = self.order_machine.get_order(order_id)
        if not order:
            return {"error": "order_not_found"}

        payment_id = str(uuid.uuid4())
        self._payments_processed[idempotency_key] = payment_id

        # Confirm inventory deduction
        for item in order.items:
            if order.warehouse_id:
                self.inventory.confirm_sale(item.product_id, order.warehouse_id,
                                            item.quantity)
                self.recommender.record_purchase(
                    order.user_id, [item.product_id for item in order.items]
                )

        self.order_machine.transition(order.order_id, OrderStatus.PAID,
                                       {"payment_id": payment_id})
        return {"status": "success", "payment_id": payment_id}

    def track_order(self, order_id: str) -> dict:
        order = self.order_machine.get_order(order_id)
        if not order:
            return {"error": "not_found"}
        return {
            "order_id": order_id,
            "status": order.status.value,
            "total": order.total,
            "history": order.status_history
        }


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 65)
    print("  Amazon E-Commerce System Simulation")
    print("=" * 65)

    # Setup warehouses
    warehouses = [
        Warehouse("wh_east", "East US", 40.7128, -74.0060),
        Warehouse("wh_west", "West US", 37.7749, -122.4194),
        Warehouse("wh_central", "Central US", 41.8781, -87.6298)
    ]

    amazon = AmazonSystem(warehouses)
    pricing = amazon.pricing

    # Add products
    products = [
        Product("p1", "Sony WH-1000XM5 Wireless Headphones", "Sony",
                "Electronics", 299.99, 4.7, 8420, tags=["wireless", "noise cancelling"]),
        Product("p2", "Apple AirPods Pro 2nd Gen", "Apple",
                "Electronics", 249.99, 4.8, 15300, tags=["wireless", "earbuds", "apple"]),
        Product("p3", "Kindle Paperwhite", "Amazon",
                "Electronics", 139.99, 4.6, 22000, tags=["ebook", "reader", "kindle"]),
        Product("p4", "The Art of System Design", "O'Reilly",
                "Books", 49.99, 4.5, 890, tags=["system design", "engineering"]),
        Product("p5", "Nike Air Max 270", "Nike",
                "Shoes", 150.00, 4.3, 5200, tags=["running", "nike", "sneakers"])
    ]
    for p in products:
        amazon.add_product(p)

    # Add inventory to warehouses
    amazon.inventory.add_stock("p1", "wh_east", 50)
    amazon.inventory.add_stock("p1", "wh_west", 30)
    amazon.inventory.add_stock("p2", "wh_east", 100)
    amazon.inventory.add_stock("p3", "wh_central", 200)
    amazon.inventory.add_stock("p4", "wh_east", 75)
    amazon.inventory.add_stock("p5", "wh_west", 40)

    # Add coupons
    pricing.add_coupon("SAVE10", 0.10, min_order=50)
    pricing.add_coupon("PRIME20", 0.20, min_order=100)

    # --- Scenario 1: Search ---
    print("\n[1] Product search")
    results = amazon.search_products(query="wireless headphones", min_rating=4.5)
    print(f"    Query: 'wireless headphones' (min_rating=4.5)")
    print(f"    Results: {results['total']}")
    for p in results["results"]:
        print(f"      {p.title[:40]:<40} ${p.base_price:.2f}  *{p.rating}")
    print(f"    Facets: {results['facets']}")

    # --- Scenario 2: Cart + Pricing ---
    print("\n[2] Shopping cart with coupon")
    r1 = amazon.add_to_cart("alice", "p1", 1, coupon_code="SAVE10")
    r2 = amazon.add_to_cart("alice", "p4", 2)
    print(f"    Added Sony headphones: ${r1['unit_price']}")
    print(f"    Added 2x System Design book")
    print(f"    Cart subtotal: ${amazon.cart.get_subtotal('alice'):.2f}")

    # Price breakdown
    breakdown = pricing.calculate_price(products[0], 1, "SAVE10", is_prime=True)
    print(f"    Price breakdown for headphones:")
    for k, v in breakdown.items():
        print(f"      {k:<22}: {v}")

    # --- Scenario 3: Inventory availability ---
    print("\n[3] Inventory availability for Sony headphones")
    avail = amazon.inventory.check_availability("p1")
    print(f"    Total available: {avail['total_available']}")
    for w in avail["warehouses"]:
        print(f"      Warehouse {w['warehouse_id']}: {w['available']} units")

    # --- Scenario 4: Place order ---
    print("\n[4] Place order (delivery to NYC)")
    order_result = amazon.place_order("alice", delivery_lat=40.7128, delivery_lon=-74.0060)
    print(f"    Order ID   : {order_result.get('order_id', '')[:8]}...")
    print(f"    Status     : {order_result.get('status')}")
    print(f"    Total      : ${order_result.get('total', 0):.2f}")
    print(f"    Warehouse  : {order_result.get('warehouse_id')}")

    # --- Scenario 5: Payment with idempotency ---
    print("\n[5] Payment processing with idempotency")
    order_id = order_result.get("order_id")
    idem_key = "payment-idem-xyz-123"
    pay1 = amazon.process_payment(order_id, idem_key)
    pay2 = amazon.process_payment(order_id, idem_key)   # retry
    print(f"    First payment  : {pay1['status']} | id={pay1.get('payment_id', '')[:8]}...")
    print(f"    Retry payment  : {pay2['status']} | same_id={pay1.get('payment_id') == pay2.get('payment_id')}")

    # --- Scenario 6: Order tracking ---
    print("\n[6] Order state transitions")
    amazon.order_machine.transition(order_id, OrderStatus.FULFILLMENT_ASSIGNED)
    amazon.order_machine.transition(order_id, OrderStatus.SHIPPED)
    tracking = amazon.track_order(order_id)
    print(f"    Current status: {tracking['status']}")
    print(f"    History:")
    for event in tracking["history"]:
        ts = datetime.fromtimestamp(event["timestamp"]).strftime('%H:%M:%S')
        print(f"      [{ts}] {event['status']}")

    # --- Scenario 7: Recommendations ---
    print("\n[7] Collaborative filtering recommendations")
    # Simulate other users' purchases
    amazon.recommender.record_purchase("bob", ["p1", "p2", "p3"])
    amazon.recommender.record_purchase("carol", ["p1", "p4", "p5"])
    amazon.recommender.record_purchase("dave", ["p2", "p3"])
    # Alice bought p1 and p4 (from her order above)
    amazon.recommender.record_purchase("alice", ["p1", "p4"])
    recs = amazon.recommender.get_recommendations("alice")
    print(f"    Alice's recommendations (based on similar users):")
    for pid in recs:
        p = amazon._products.get(pid)
        if p:
            print(f"      {p.title[:45]:<45} ${p.base_price:.2f}")

    fbt = amazon.recommender.get_frequently_bought_together("p1")
    print(f"    Frequently bought with Sony Headphones: {fbt}")

    # --- Scenario 8: Inventory optimistic locking stats ---
    print("\n[8] Inventory manager statistics")
    stats = amazon.inventory.get_stats()
    for k, v in stats.items():
        print(f"    {k:<15}: {v}")

    print("\n" + "=" * 65)
    print("  Simulation Complete")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()
