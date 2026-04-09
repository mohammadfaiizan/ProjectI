"""
DATABASE PER SERVICE PATTERN
================================

Problem Statement:
In a monolith, all modules share one database. Easy to JOIN across tables.
In microservices, if services share a database:
  - Schema changes in one service can break another service.
  - Services are coupled at the data layer (not just the API layer).
  - Can't scale or replace the DB for one service independently.
  - One service can read another's data directly, bypassing business rules.

Solution: Each service owns its own database (schema/cluster/engine).
  No other service can read/write that database directly.
  Cross-service data access happens via API calls only.

Polyglot Persistence:
  Each service can choose the database best suited to its data shape:
    OrderService     → PostgreSQL (relational, ACID transactions)
    ProductService   → MongoDB (flexible schema, JSON documents)
    SessionService   → Redis (in-memory, key-value, fast expiry)
    SearchService    → Elasticsearch (full-text search, facets)
    TimeSeriesData   → InfluxDB / TimescaleDB (metrics, events)

Cross-Service Queries (API Composition):
  Problem: you can't do SQL JOIN across two services' databases.
  Solution: fetch data from each service separately, join in-memory.
    Client or BFF calls OrderService + UserService, merges results.
  Trade-off: N+1 problem; fan-out latency; eventual consistency.

Eventual Consistency:
  Without a shared DB, you can't have cross-service ACID transactions.
  Use SAGA pattern: sequence of local transactions with compensating actions.
  Services communicate via events: OrderPlaced → InventoryService reserves stock.
  Data is eventually consistent across service boundaries.

Shared Database Anti-patterns:
  - Two services reading the same table → coupling at data layer.
  - Service A calling Service B's stored procedures → coupling via DB.
  - Shared ORM models across services → tight version coupling.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


# ─────────────────────────────────────────────
# DATABASE SIMULATORS (polyglot persistence)
# ─────────────────────────────────────────────

class PostgreSQLSimulator:
    """Relational DB — supports transactions, foreign keys, rich queries."""

    def __init__(self, name: str):
        self.name   = name
        self._store : Dict[str, Dict] = {}    # table:id → row
        self._seq   : Dict[str, int]  = {}    # table → next_id

    def insert(self, table: str, data: Dict) -> str:
        row_id = data.get("id") or f"{table[:3]}-{self._seq.get(table, 1):04d}"
        self._seq[table] = self._seq.get(table, 1) + 1
        row = {"id": row_id, **data}
        self._store[f"{table}:{row_id}"] = row
        return row_id

    def select(self, table: str, id: str) -> Optional[Dict]:
        return self._store.get(f"{table}:{id}")

    def select_where(self, table: str, **conditions) -> List[Dict]:
        results = []
        prefix = f"{table}:"
        for key, row in self._store.items():
            if not key.startswith(prefix):
                continue
            if all(row.get(k) == v for k, v in conditions.items()):
                results.append(row)
        return results

    def update(self, table: str, id: str, updates: Dict) -> bool:
        key = f"{table}:{id}"
        if key in self._store:
            self._store[key].update(updates)
            return True
        return False

    def __repr__(self):
        return f"PostgreSQL({self.name})"


class MongoDBSimulator:
    """Document DB — flexible schema, JSON documents, no rigid schema."""

    def __init__(self, name: str):
        self.name        = name
        self._collections: Dict[str, Dict[str, Dict]] = {}

    def insert_one(self, collection: str, document: Dict) -> str:
        doc_id = document.get("_id") or str(uuid.uuid4())[:8]
        doc    = {"_id": doc_id, **document}
        self._collections.setdefault(collection, {})[doc_id] = doc
        return doc_id

    def find_one(self, collection: str, filter: Dict) -> Optional[Dict]:
        for doc in self._collections.get(collection, {}).values():
            if all(doc.get(k) == v for k, v in filter.items()):
                return doc
        return None

    def find(self, collection: str, filter: Dict) -> List[Dict]:
        results = []
        for doc in self._collections.get(collection, {}).values():
            if all(doc.get(k) == v for k, v in filter.items()):
                results.append(doc)
        return results

    def __repr__(self):
        return f"MongoDB({self.name})"


class RedisSimulator:
    """In-memory key-value store — fast, with TTL support."""

    def __init__(self, name: str):
        self.name   = name
        self._store : Dict[str, Any]   = {}
        self._expiry: Dict[str, float] = {}

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        self._store[key] = value
        if ttl_seconds:
            self._expiry[key] = time.time() + ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        expiry = self._expiry.get(key)
        if expiry and time.time() > expiry:
            del self._store[key]
            del self._expiry[key]
            return None
        return self._store.get(key)

    def delete(self, key: str):
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self):
        return f"Redis({self.name})"


# ─────────────────────────────────────────────
# SERVICES (each owns its database)
# ─────────────────────────────────────────────

class OrderService:
    """Owns PostgreSQL. Nobody else reads this DB."""

    def __init__(self):
        self._db = PostgreSQLSimulator("orders_db")

    @property
    def db_type(self): return repr(self._db)

    def create_order(self, customer_id: str, items: List[Dict]) -> Dict:
        order_id = f"ord-{str(uuid.uuid4())[:6]}"
        order = {
            "id"          : order_id,
            "customer_id" : customer_id,
            "items"       : items,
            "total"       : sum(i["price"] * i["qty"] for i in items),
            "status"      : "pending",
            "created_at"  : time.time(),
        }
        self._db.insert("orders", order)
        return order

    def get_order(self, order_id: str) -> Optional[Dict]:
        return self._db.select("orders", order_id)

    def get_orders_for_customer(self, customer_id: str) -> List[Dict]:
        return self._db.select_where("orders", customer_id=customer_id)

    def update_status(self, order_id: str, status: str) -> bool:
        return self._db.update("orders", order_id, {"status": status})


class ProductService:
    """Owns MongoDB. Flexible product schema (varying attributes per category)."""

    def __init__(self):
        self._db = MongoDBSimulator("products_db")

    @property
    def db_type(self): return repr(self._db)

    def create_product(self, sku: str, name: str, price: float,
                        attributes: Dict) -> str:
        """MongoDB allows different attributes per product (polyglot schema)."""
        return self._db.insert_one("products", {
            "_id"       : sku,
            "name"      : name,
            "price"     : price,
            "attributes": attributes,   # electronics: {wattage, voltage}
                                        # clothing: {size, color, material}
        })

    def get_product(self, sku: str) -> Optional[Dict]:
        return self._db.find_one("products", {"_id": sku})

    def get_products_by_name(self, name: str) -> List[Dict]:
        return self._db.find("products", {"name": name})


class SessionService:
    """Owns Redis. Sessions expire automatically; fast read/write."""

    def __init__(self):
        self._db = RedisSimulator("session_store")

    @property
    def db_type(self): return repr(self._db)

    def create_session(self, user_id: str, metadata: Dict,
                        ttl_seconds: int = 3600) -> str:
        session_id = str(uuid.uuid4())
        self._db.set(f"session:{session_id}", {
            "user_id"   : user_id,
            "metadata"  : metadata,
            "created_at": time.time(),
        }, ttl_seconds=ttl_seconds)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self._db.get(f"session:{session_id}")

    def invalidate(self, session_id: str):
        self._db.delete(f"session:{session_id}")


# ─────────────────────────────────────────────
# API COMPOSITION (cross-service query)
# ─────────────────────────────────────────────

class OrderCompositionService:
    """
    Aggregates data from multiple services via API calls.
    No direct DB access to other services' databases.
    This is the correct pattern for cross-service queries.
    """

    def __init__(self, order_svc: OrderService, product_svc: ProductService):
        self._orders   = order_svc
        self._products = product_svc

    def get_order_with_product_details(self, order_id: str) -> Optional[Dict]:
        """
        Fetch order + enrich each line item with product details.
        API call to OrderService + N calls to ProductService.
        Trade-off: N+1 problem; mitigated by batching or caching.
        """
        order = self._orders.get_order(order_id)
        if not order:
            return None

        enriched_items = []
        for item in order.get("items", []):
            product = self._products.get_product(item["sku"])
            enriched_items.append({
                **item,
                "product_name"      : product.get("name") if product else "unknown",
                "product_attributes": product.get("attributes") if product else {},
            })

        return {**order, "items": enriched_items}

    def get_customer_order_history(self, customer_id: str) -> List[Dict]:
        """All orders for a customer with enriched product info."""
        orders   = self._orders.get_orders_for_customer(customer_id)
        enriched = []
        for order in orders:
            enriched_items = []
            for item in order.get("items", []):
                product = self._products.get_product(item["sku"])
                enriched_items.append({
                    "sku"  : item["sku"],
                    "qty"  : item["qty"],
                    "price": item["price"],
                    "name" : product.get("name") if product else "unknown",
                })
            enriched.append({**order, "items": enriched_items})
        return enriched


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_database_per_service():
    print("=" * 65)
    print("DATABASE PER SERVICE PATTERN")
    print("=" * 65)

    order_svc   = OrderService()
    product_svc = ProductService()
    session_svc = SessionService()
    composer    = OrderCompositionService(order_svc, product_svc)

    # ── 1. Polyglot persistence ───────────────────
    print("\n[1] POLYGLOT PERSISTENCE — EACH SERVICE PICKS ITS DB")
    print("─" * 55)
    svcs = [
        ("OrderService",   order_svc.db_type,   "ACID transactions, relational queries"),
        ("ProductService", product_svc.db_type, "Flexible schema, varying attributes per category"),
        ("SessionService", session_svc.db_type, "In-memory, auto-expiry, sub-ms latency"),
    ]
    print(f"  {'Service':<18} {'Database':<25} {'Why'}")
    print(f"  {'─'*70}")
    for svc, db, reason in svcs:
        print(f"  {svc:<18} {db:<25} {reason}")

    # ── 2. Each service populates its own data ────
    print("\n\n[2] SERVICE DATA — OWNED INDEPENDENTLY")
    print("─" * 55)

    # Product service: flexible MongoDB schema (different attributes per product)
    product_svc.create_product("SKU-LAPTOP-01", "Pro Laptop 15",  1299.99,
                                {"cpu": "M3", "ram_gb": 16, "storage_gb": 512,
                                 "weight_kg": 1.6, "os": "macOS"})
    product_svc.create_product("SKU-TSHIRT-01", "Classic Tee",    29.99,
                                {"size": "M", "color": "Navy", "material": "100% cotton"})
    product_svc.create_product("SKU-MOUSE-01",  "Ergonomic Mouse", 79.99,
                                {"dpi": 1600, "wireless": True, "buttons": 6})

    p = product_svc.get_product("SKU-LAPTOP-01")
    print(f"  Product (MongoDB): {p['name']}  price={p['price']}")
    print(f"    Attributes: {p['attributes']}")

    # Order service: relational PostgreSQL
    order1 = order_svc.create_order("cust-42", [
        {"sku": "SKU-LAPTOP-01", "qty": 1, "price": 1299.99},
        {"sku": "SKU-MOUSE-01",  "qty": 1, "price": 79.99},
    ])
    order2 = order_svc.create_order("cust-42", [
        {"sku": "SKU-TSHIRT-01", "qty": 3, "price": 29.99},
    ])
    print(f"\n  Orders (PostgreSQL):")
    print(f"    {order1['id']}: total={order1['total']} status={order1['status']}")
    print(f"    {order2['id']}: total={order2['total']:.2f} status={order2['status']}")

    # Session service: Redis with TTL
    sess_id = session_svc.create_session("cust-42",
                                         {"device": "iPhone", "ip": "1.2.3.4"},
                                         ttl_seconds=3600)
    session = session_svc.get_session(sess_id)
    print(f"\n  Session (Redis):  {sess_id[:16]}... user={session['user_id']}")

    # ── 3. Cross-service join via API composition ─
    print("\n\n[3] CROSS-SERVICE JOIN — API COMPOSITION (not SQL JOIN)")
    print("─" * 55)
    print("  Enriching order with product details:")
    print("  Step 1: Call OrderService → get order + SKUs")
    print("  Step 2: Call ProductService per SKU → get product names")
    print("  Step 3: Merge in-memory (no DB JOIN)")
    print()

    enriched = composer.get_order_with_product_details(order1["id"])
    if enriched:
        print(f"  Order {enriched['id']} (total={enriched['total']}):")
        for item in enriched["items"]:
            print(f"    SKU={item['sku']}  qty={item['qty']}  "
                  f"price={item['price']}  name={item['product_name']}")

    # ── 4. Customer order history ─────────────────
    print("\n\n[4] CUSTOMER ORDER HISTORY (API COMPOSITION)")
    print("─" * 55)
    history = composer.get_customer_order_history("cust-42")
    print(f"  Customer cust-42 has {len(history)} orders:")
    for order in history:
        print(f"  Order {order['id']}: total={order['total']:.2f}")
        for item in order["items"]:
            print(f"    {item['name']:<25} qty={item['qty']}  sku={item['sku']}")

    # ── 5. Bounded context violation attempt ──────
    print("\n\n[5] BOUNDED CONTEXT — SERVICES CANNOT ACCESS EACH OTHER'S DB")
    print("─" * 55)
    print("  ✗ WRONG: inventory_service._db.select('orders', 'ord-001')")
    print("    This would bypass OrderService business rules.")
    print("    Creates coupling at the data layer.")
    print()
    print("  ✓ CORRECT: response = order_service.get_order('ord-001')")
    print("    Goes through the API; respects business rules.")
    print("    OrderService can add auth, validation, events, etc.")

    # ── 6. Eventual consistency ───────────────────
    print("\n\n[6] EVENTUAL CONSISTENCY ACROSS SERVICES")
    print("─" * 55)
    print("  Scenario: order placed → inventory must be reserved")
    print()

    order = order_svc.create_order("cust-99",
                                   [{"sku": "SKU-LAPTOP-01", "qty": 1, "price": 1299.99}])
    print(f"  t=0ms:  OrderService creates order {order['id']} (status=pending)")
    print(f"  t=0ms:  OrderService publishes OrderPlaced event (async)")

    time.sleep(0.005)
    print(f"  t=5ms:  InventoryService consumes event, reserves 1 unit")

    time.sleep(0.005)
    order_svc.update_status(order["id"], "confirmed")
    print(f"  t=10ms: OrderService receives StockReserved event, status=confirmed")

    print()
    print("  During t=0→10ms: order exists in OrderService DB (pending)")
    print("  but InventoryService has not yet updated its DB.")
    print("  → Eventual consistency; not immediately consistent.")
    print("  → Use SAGA pattern for cross-service transactions.")

    # ── 7. Comparison ─────────────────────────────
    print("\n\n[7] SHARED DB vs DATABASE-PER-SERVICE")
    print("─" * 55)
    rows = [
        ("Schema changes",    "Can break other services",          "Service controls its own schema"),
        ("Scaling",           "Scale the whole DB",                "Scale each DB independently"),
        ("Technology choice", "One DB for all services",           "Best DB per service needs"),
        ("Cross-service query","SQL JOIN (easy but coupling)",      "API composition (harder but decoupled)"),
        ("Transactions",      "ACID across tables (easy)",         "SAGA pattern required (complex)"),
        ("Data isolation",    "Services share data access paths",  "Each service owns its data fully"),
    ]
    print(f"  {'Concern':<22} {'Shared DB':<35} {'DB per Service'}")
    print(f"  {'─'*80}")
    for concern, shared, per_svc in rows:
        print(f"  {concern:<22} {shared:<35} {per_svc}")


if __name__ == "__main__":
    demonstrate_database_per_service()
