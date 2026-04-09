"""
BACKEND FOR FRONTEND (BFF) PATTERN
=====================================

Problem Statement:
A general-purpose API serves the lowest common denominator.
Mobile apps need small payloads and fewer fields (battery/bandwidth).
Web dashboards need richer data with aggregated stats.
Desktop apps may need different pagination, export formats, and bulk operations.
One API cannot serve all clients well without burdening each client with
filtering, aggregating, and transforming data.

BFF Pattern:
  Create a dedicated backend layer per client type.
  Each BFF aggregates the right data from downstream services,
  shapes the response for its specific client, and owns nothing else.

  Mobile BFF:  Minimal payload. Battery-aware. Offline-first fields.
               Compresses images to thumbnails. Single round trip per screen.
  Web BFF:     Rich payload. Server-side pagination. Aggregated analytics.
               Can afford multiple joins because server-side.
  Desktop BFF: Bulk operations. Large datasets. Export support.
               Power-user features (batch, CSV, advanced filters).

Why Not Just Use the API Gateway?
  API Gateway: infrastructure concerns (auth, routing, rate limit).
  BFF: client-specific business logic (what data, how formatted).
  Mixing them leads to a bloated, fragile gateway.

BFF Responsibilities:
  - Aggregate data from multiple downstream services.
  - Shape/filter response for the client (include only needed fields).
  - Handle client-specific auth flows (OAuth PKCE for mobile, session for web).
  - Versioning: mobile BFF v2 can exist independently of web BFF.
  - Orchestrate multi-step calls on behalf of client.

Trade-offs:
  Pro:  Each client gets a perfectly shaped API. Fewer round trips.
  Pro:  Backend services remain generic; BFF absorbs client differences.
  Con:  Code duplication if BFFs share too much logic (extract shared libs).
  Con:  Another layer to deploy and maintain per client type.
  Rule: BFF per team that owns a client, not per every individual client variant.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import threading
import uuid


# ─────────────────────────────────────────────
# DOWNSTREAM SERVICES (stubs)
# ─────────────────────────────────────────────

class OrderService:
    def get_orders(self, user_id: str, limit: int = 20) -> List[Dict]:
        return [
            {"order_id": f"ord-{i:03d}", "user_id": user_id,
             "total": round(10.5 * i, 2), "status": "delivered",
             "items": [{"sku": f"SKU{i}", "name": f"Product {i}",
                        "qty": i, "price": round(10.5 * i, 2),
                        "image_url": f"https://cdn.example.com/img/{i}_full.jpg",
                        "thumbnail_url": f"https://cdn.example.com/img/{i}_thumb.jpg"}],
             "created_at": "2025-03-01T10:00:00Z",
             "shipping_address": {"street": f"{i} Main St", "city": "Springfield",
                                  "zip": "12345", "country": "US"},
             "internal_metadata": {"warehouse_id": f"WH{i%3}", "picker_id": f"P{i}"}}
            for i in range(1, limit + 1)
        ]


class UserService:
    def get_profile(self, user_id: str) -> Dict:
        return {
            "user_id"      : user_id,
            "name"         : "Alice Smith",
            "email"        : "alice@example.com",
            "phone"        : "+1-555-0100",
            "avatar_url"   : "https://cdn.example.com/avatars/alice_full.jpg",
            "avatar_thumb" : "https://cdn.example.com/avatars/alice_32x32.jpg",
            "membership"   : "gold",
            "points"       : 4200,
            "join_date"    : "2022-01-15",
            "preferences"  : {"theme": "dark", "language": "en",
                               "notifications": {"email": True, "push": True}},
            "internal"     : {"sso_provider": "google", "last_login_ip": "1.2.3.4"},
        }


class RecommendationService:
    def get_recs(self, user_id: str, limit: int = 5) -> List[Dict]:
        return [
            {"sku": f"REC{i}", "name": f"Recommended Item {i}",
             "price": round(29.99 + i * 5, 2),
             "image_url": f"https://cdn.example.com/img/rec{i}_full.jpg",
             "thumbnail_url": f"https://cdn.example.com/img/rec{i}_thumb.jpg",
             "rating": round(3.5 + i * 0.1, 1),
             "review_count": 100 + i * 23}
            for i in range(1, limit + 1)
        ]


class AnalyticsService:
    def get_user_stats(self, user_id: str) -> Dict:
        return {
            "total_orders"      : 47,
            "total_spend"       : 1842.50,
            "avg_order_value"   : 39.20,
            "favorite_category" : "Electronics",
            "orders_this_month" : 3,
            "spend_this_month"  : 112.40,
            "cohort"            : "high_value",
        }


# ─────────────────────────────────────────────
# BASE BFF
# ─────────────────────────────────────────────

class BaseBFF:
    """Common aggregation utilities shared across all BFFs."""

    def __init__(self):
        self.order_svc  = OrderService()
        self.user_svc   = UserService()
        self.rec_svc    = RecommendationService()
        self.analytics  = AnalyticsService()

    def _parallel_fetch(self, tasks: List[tuple]) -> Dict[str, Any]:
        """Run multiple service calls in parallel. tasks = [(key, fn), ...]"""
        results  = {}
        errors   = {}
        lock     = threading.Lock()

        def run(key, fn):
            try:
                result = fn()
                with lock:
                    results[key] = result
            except Exception as e:
                with lock:
                    errors[key] = str(e)

        threads = [threading.Thread(target=run, args=(k, f)) for k, f in tasks]
        for t in threads: t.start()
        for t in threads: t.join()

        return results, errors


# ─────────────────────────────────────────────
# MOBILE BFF
# ─────────────────────────────────────────────

class MobileBFF(BaseBFF):
    """
    Mobile clients: bandwidth constrained, battery sensitive.
    - Return only fields needed by mobile screens.
    - Use thumbnails not full images.
    - Limit list sizes aggressively.
    - Single endpoint per screen (reduce round trips).
    """

    def home_screen(self, user_id: str, correlation_id: str) -> Dict:
        t0 = time.time()
        data, _ = self._parallel_fetch([
            ("profile", lambda: self.user_svc.get_profile(user_id)),
            ("orders",  lambda: self.order_svc.get_orders(user_id, limit=3)),
            ("recs",    lambda: self.rec_svc.get_recs(user_id, limit=4)),
        ])

        profile = data.get("profile", {})
        orders  = data.get("orders",  [])
        recs    = data.get("recs",    [])

        # Shape for mobile: minimal fields, thumbnails only
        result = {
            "user": {
                "name"         : profile.get("name"),
                "avatar"       : profile.get("avatar_thumb"),   # thumb not full
                "points"       : profile.get("points"),
                "membership"   : profile.get("membership"),
            },
            "recent_orders": [
                {
                    "order_id" : o["order_id"],
                    "total"    : o["total"],
                    "status"   : o["status"],
                    "item_count": len(o["items"]),
                }
                for o in orders[:3]        # max 3
            ],
            "recommendations": [
                {
                    "sku"       : r["sku"],
                    "name"      : r["name"],
                    "price"     : r["price"],
                    "thumbnail" : r["thumbnail_url"],   # thumb only
                }
                for r in recs[:4]
            ],
            "_meta": {
                "bff"            : "mobile",
                "correlation_id" : correlation_id,
                "latency_ms"     : round((time.time() - t0) * 1000, 1),
            }
        }
        return result


# ─────────────────────────────────────────────
# WEB BFF
# ─────────────────────────────────────────────

class WebBFF(BaseBFF):
    """
    Web dashboard: richer data, analytics, full images.
    - More fields per item; full-resolution images.
    - Aggregated analytics summary.
    - Larger page sizes.
    """

    def dashboard(self, user_id: str, correlation_id: str) -> Dict:
        t0 = time.time()
        data, _ = self._parallel_fetch([
            ("profile",   lambda: self.user_svc.get_profile(user_id)),
            ("orders",    lambda: self.order_svc.get_orders(user_id, limit=10)),
            ("recs",      lambda: self.rec_svc.get_recs(user_id, limit=6)),
            ("analytics", lambda: self.analytics.get_user_stats(user_id)),
        ])

        profile   = data.get("profile",   {})
        orders    = data.get("orders",    [])
        recs      = data.get("recs",      [])
        analytics = data.get("analytics", {})

        result = {
            "user": {
                "user_id"     : profile.get("user_id"),
                "name"        : profile.get("name"),
                "email"       : profile.get("email"),
                "avatar"      : profile.get("avatar_url"),    # full resolution
                "membership"  : profile.get("membership"),
                "points"      : profile.get("points"),
                "join_date"   : profile.get("join_date"),
                "preferences" : profile.get("preferences"),   # full prefs
            },
            "order_history": [
                {
                    "order_id"        : o["order_id"],
                    "total"           : o["total"],
                    "status"          : o["status"],
                    "created_at"      : o["created_at"],
                    "items"           : [
                        {
                            "sku"   : item["sku"],
                            "name"  : item["name"],
                            "qty"   : item["qty"],
                            "price" : item["price"],
                            "image" : item["image_url"],   # full image
                        }
                        for item in o["items"]
                    ],
                    "shipping_address": o["shipping_address"],
                }
                for o in orders[:10]
            ],
            "analytics_summary": analytics,     # full analytics block
            "recommendations": recs,            # full rec objects
            "_meta": {
                "bff"           : "web",
                "correlation_id": correlation_id,
                "latency_ms"    : round((time.time() - t0) * 1000, 1),
            }
        }
        return result


# ─────────────────────────────────────────────
# DESKTOP BFF
# ─────────────────────────────────────────────

class DesktopBFF(BaseBFF):
    """
    Desktop: power-user features, bulk data, exports.
    - Full order history with internal metadata.
    - Analytics at maximum detail.
    - Export-ready format (flat rows).
    """

    def full_export(self, user_id: str, correlation_id: str) -> Dict:
        t0 = time.time()
        data, _ = self._parallel_fetch([
            ("profile",   lambda: self.user_svc.get_profile(user_id)),
            ("orders",    lambda: self.order_svc.get_orders(user_id, limit=20)),
            ("analytics", lambda: self.analytics.get_user_stats(user_id)),
        ])

        profile   = data.get("profile",   {})
        orders    = data.get("orders",    [])
        analytics = data.get("analytics", {})

        # Flatten for export (CSV-ready structure)
        flat_rows = []
        for o in orders:
            for item in o["items"]:
                flat_rows.append({
                    "order_id"       : o["order_id"],
                    "order_status"   : o["status"],
                    "order_total"    : o["total"],
                    "order_date"     : o["created_at"],
                    "sku"            : item["sku"],
                    "item_name"      : item["name"],
                    "qty"            : item["qty"],
                    "item_price"     : item["price"],
                    "warehouse_id"   : o["internal_metadata"]["warehouse_id"],
                })

        result = {
            "user_profile"   : profile,    # everything including internal
            "analytics"      : analytics,
            "export_rows"    : flat_rows,
            "export_row_count": len(flat_rows),
            "_meta": {
                "bff"           : "desktop",
                "correlation_id": correlation_id,
                "latency_ms"    : round((time.time() - t0) * 1000, 1),
            }
        }
        return result


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_bff_pattern():
    print("=" * 65)
    print("BACKEND FOR FRONTEND (BFF) PATTERN")
    print("=" * 65)

    user_id = "user-42"
    corr_id = str(uuid.uuid4())[:8]

    mobile  = MobileBFF()
    web     = WebBFF()
    desktop = DesktopBFF()

    # ── 1. Mobile BFF ─────────────────────────────
    print("\n[1] MOBILE BFF — MINIMAL PAYLOAD")
    print("─" * 55)
    m_resp = mobile.home_screen(user_id, corr_id)

    print(f"  Fetched in: {m_resp['_meta']['latency_ms']}ms  (parallel calls)")
    print(f"  User fields returned: {list(m_resp['user'].keys())}")
    print(f"  Recent orders:        {len(m_resp['recent_orders'])} (capped at 3)")
    print(f"  Recommendations:      {len(m_resp['recommendations'])} (capped at 4)")
    print(f"  Image field:          thumbnail_url only (not full image)")
    order_fields = list(m_resp['recent_orders'][0].keys()) if m_resp['recent_orders'] else []
    print(f"  Order fields:         {order_fields}  (no address, no items detail)")
    print(f"\n  Total payload keys (approx): ~{_count_keys(m_resp)}")

    # ── 2. Web BFF ────────────────────────────────
    print("\n\n[2] WEB BFF — RICH PAYLOAD WITH ANALYTICS")
    print("─" * 55)
    w_resp = web.dashboard(user_id, corr_id)

    print(f"  Fetched in: {w_resp['_meta']['latency_ms']}ms  (parallel calls)")
    print(f"  User fields returned: {list(w_resp['user'].keys())}")
    print(f"  Order history:        {len(w_resp['order_history'])} orders (full items)")
    print(f"  Analytics:            {list(w_resp['analytics_summary'].keys())}")
    print(f"  Image field:          full-resolution image_url")
    if w_resp['order_history']:
        item_fields = list(w_resp['order_history'][0]['items'][0].keys())
        print(f"  Item fields:          {item_fields}")
    print(f"\n  Total payload keys (approx): ~{_count_keys(w_resp)}")

    # ── 3. Desktop BFF ────────────────────────────
    print("\n\n[3] DESKTOP BFF — FULL EXPORT WITH INTERNAL METADATA")
    print("─" * 55)
    d_resp = desktop.full_export(user_id, corr_id)

    print(f"  Fetched in: {d_resp['_meta']['latency_ms']}ms")
    print(f"  Export rows: {d_resp['export_row_count']} (flattened order+item rows)")
    print(f"  Includes:    warehouse_id (internal metadata visible to power users)")
    if d_resp['export_rows']:
        print(f"  Row fields:  {list(d_resp['export_rows'][0].keys())}")
    print(f"  Full user profile fields: {len(d_resp['user_profile'])} fields")
    print(f"\n  Total payload keys (approx): ~{_count_keys(d_resp)}")

    # ── 4. Payload size comparison ────────────────
    print("\n\n[4] PAYLOAD SIZE COMPARISON")
    print("─" * 55)
    import json
    sizes = {
        "Mobile BFF":  len(json.dumps(m_resp)),
        "Web BFF":     len(json.dumps(w_resp)),
        "Desktop BFF": len(json.dumps(d_resp)),
    }
    max_size = max(sizes.values())
    print(f"  {'BFF':<16} {'Payload (chars)':<18} {'Relative size'}")
    print(f"  {'─'*50}")
    for bff, size in sizes.items():
        bar = "█" * int(size / max_size * 20)
        print(f"  {bff:<16} {size:<18} {bar}")

    # ── 5. BFF principles ─────────────────────────
    print("\n\n[5] BFF DESIGN PRINCIPLES")
    print("─" * 55)
    principles = [
        ("One BFF per client type",    "Mobile, Web, Desktop — not per every screen"),
        ("BFF owns aggregation",       "Calls multiple services, shapes response"),
        ("BFF owned by frontend team", "Team building the UI owns the BFF"),
        ("No business logic",          "BFF transforms; business rules in services"),
        ("Parallel downstream calls",  "Fan-out to reduce latency"),
        ("Independent versioning",     "Mobile BFF v2 while Web BFF stays v1"),
        ("Different auth flows",       "PKCE for mobile, session for web"),
        ("Not a generic API gateway",  "Gateway handles infra; BFF handles client needs"),
    ]
    for name, desc in principles:
        print(f"  {name:<30} {desc}")


def _count_keys(d: Any, depth: int = 3) -> int:
    """Rough count of total keys in a nested dict."""
    if depth == 0 or not isinstance(d, dict):
        return 0
    count = len(d)
    for v in d.values():
        if isinstance(v, dict):
            count += _count_keys(v, depth - 1)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            count += _count_keys(v[0], depth - 1)
    return count


if __name__ == "__main__":
    demonstrate_bff_pattern()
