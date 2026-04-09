"""
REST VS GRAPHQL VS gRPC
========================

Problem Statement:
Modern distributed systems expose APIs using different protocols. Each has
unique strengths. REST is ubiquitous; GraphQL solves the over/under-fetching
problem; gRPC excels for low-latency internal microservice communication.

Comparison:
  REST    : Stateless, resource-based, HTTP verbs, JSON, widely adopted
  GraphQL : Single endpoint, client specifies exact fields, solves N+1 problem
  gRPC    : Binary (Protobuf), strongly typed, streaming, low latency, HTTP/2

N+1 Problem (REST pain-point):
  Fetch user → 1 request
  Fetch each user's posts → N requests
  Total: 1 + N requests for N users

When to Use:
  REST   → Public APIs, simple CRUD, external clients
  GraphQL→ Complex data graphs, mobile clients (bandwidth), multiple consumer types
  gRPC   → Internal microservices, streaming, high-throughput RPC
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
import time


class APIProtocol(Enum):
    REST    = "REST"
    GRAPHQL = "GraphQL"
    GRPC    = "gRPC"


# ─────────────────────────────────────────────
# SHARED DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class User:
    user_id : str
    name    : str
    email   : str
    bio     : str
    followers: int
    posts   : List[dict] = field(default_factory=list)


@dataclass
class Post:
    post_id  : str
    user_id  : str
    content  : str
    likes    : int
    created_at: str


# Simulated DB
USERS_DB: Dict[str, User] = {
    "u1": User("u1", "Alice", "alice@example.com", "Software engineer at BigCo", 5000),
    "u2": User("u2", "Bob",   "bob@example.com",   "Designer and artist",         2000),
}
POSTS_DB: List[Post] = [
    Post("p1", "u1", "Hello World!", 42, "2024-01-01"),
    Post("p2", "u1", "REST vs gRPC explained", 120, "2024-01-02"),
    Post("p3", "u2", "Design patterns 101", 88, "2024-01-03"),
]


# ─────────────────────────────────────────────
# REST CLIENT
# ─────────────────────────────────────────────

@dataclass
class RESTRequest:
    method  : str
    endpoint: str
    params  : Dict = field(default_factory=dict)
    body    : Dict = field(default_factory=dict)


@dataclass
class RESTResponse:
    status_code: int
    data       : Any
    latency_ms : float
    bytes_sent : int = 0


class RESTClient:
    """Simulates REST API calls."""

    def __init__(self, base_url: str = "https://api.example.com"):
        self.base_url    = base_url
        self.request_log : List[RESTRequest] = []

    def get(self, path: str, params: Dict = None) -> RESTResponse:
        req = RESTRequest("GET", f"{self.base_url}{path}", params or {})
        self.request_log.append(req)
        start = time.perf_counter()

        # Simulate server processing
        if path.startswith("/users/") and "/posts" not in path:
            uid = path.split("/")[2]
            user = USERS_DB.get(uid)
            data = {"user_id": user.user_id, "name": user.name,
                    "email": user.email, "followers": user.followers} if user else None
        elif path.startswith("/users/") and "/posts" in path:
            uid  = path.split("/")[2]
            data = [{"post_id": p.post_id, "content": p.content, "likes": p.likes,
                     "created_at": p.created_at, "user_id": p.user_id,  # ← over-fetch
                     "extra_field": "not_needed"}   # ← over-fetch
                    for p in POSTS_DB if p.user_id == uid]
        else:
            data = None

        latency = (time.perf_counter() - start) * 1000 + 10
        body = json.dumps(data)
        return RESTResponse(200 if data else 404, data, round(latency, 2), len(body.encode()))

    def fetch_user_with_posts(self, user_id: str) -> Dict:
        """Classic N+1: 1 call for user, 1 call for posts."""
        r1 = self.get(f"/users/{user_id}")
        r2 = self.get(f"/users/{user_id}/posts")
        return {"user": r1.data, "posts": r2.data,
                "requests_made": 2, "total_bytes": r1.bytes_sent + r2.bytes_sent}


# ─────────────────────────────────────────────
# GRAPHQL CLIENT
# ─────────────────────────────────────────────

@dataclass
class GraphQLQuery:
    query    : str
    variables: Dict = field(default_factory=dict)


class GraphQLClient:
    """Simulates GraphQL query execution on a single endpoint."""

    def __init__(self, endpoint: str = "https://api.example.com/graphql"):
        self.endpoint  = endpoint
        self.query_log: List[GraphQLQuery] = []

    def execute(self, query: GraphQLQuery) -> Dict:
        self.query_log.append(query)
        start = time.perf_counter()

        # Parse requested fields from query (simplified simulation)
        requested_user_fields  = self._parse_fields(query.query, "user")
        requested_post_fields  = self._parse_fields(query.query, "posts")

        uid  = query.variables.get("userId", "u1")
        user = USERS_DB.get(uid)
        if not user:
            return {"errors": [{"message": f"User {uid} not found"}]}

        # Build response with ONLY requested fields
        user_data = {}
        if "name"      in requested_user_fields: user_data["name"]      = user.name
        if "email"     in requested_user_fields: user_data["email"]     = user.email
        if "followers" in requested_user_fields: user_data["followers"] = user.followers

        posts_data = []
        if requested_post_fields:
            for p in POSTS_DB:
                if p.user_id == uid:
                    post = {}
                    if "content" in requested_post_fields: post["content"] = p.content
                    if "likes"   in requested_post_fields: post["likes"]   = p.likes
                    posts_data.append(post)
            user_data["posts"] = posts_data

        latency = (time.perf_counter() - start) * 1000 + 10
        body    = json.dumps({"data": {"user": user_data}})
        return {"data": {"user": user_data}, "requests_made": 1,
                "total_bytes": len(body.encode()), "latency_ms": round(latency, 2)}

    def _parse_fields(self, query: str, section: str) -> List[str]:
        """Very simplified field extractor."""
        import re
        if section not in query:
            return []
        pattern = rf"{section}\s*\{{([^}}]+)\}}"
        m = re.search(pattern, query, re.DOTALL)
        if not m:
            return []
        return [f.strip() for f in m.group(1).split("\n") if f.strip()]


# ─────────────────────────────────────────────
# gRPC CLIENT
# ─────────────────────────────────────────────

@dataclass
class GRPCMethod:
    service      : str
    method       : str
    request_type : str
    response_type: str
    streaming    : bool = False


class GRPCClient:
    """Simulates gRPC calls with Protobuf serialization (simplified)."""

    METHODS = {
        "GetUser"      : GRPCMethod("UserService", "GetUser",      "GetUserRequest",  "User"),
        "GetUserPosts" : GRPCMethod("UserService", "GetUserPosts", "GetPostsRequest", "PostList"),
        "WatchFeed"    : GRPCMethod("FeedService", "WatchFeed",    "FeedRequest",     "FeedEvent", streaming=True),
    }

    def __init__(self, address: str = "user-service.internal:9090"):
        self.address  = address
        self.call_log : List[str] = []

    def call(self, method_name: str, request: Dict) -> Dict:
        method = self.METHODS.get(method_name)
        if not method:
            return {"error": f"Unknown method {method_name}"}

        self.call_log.append(method_name)
        start = time.perf_counter()

        # Simulate Protobuf: binary serialization is ~30% smaller than JSON
        if method_name == "GetUser":
            uid = request.get("user_id", "u1")
            u   = USERS_DB.get(uid)
            data = {"user_id": u.user_id, "name": u.name, "email": u.email} if u else {}
        else:
            uid  = request.get("user_id", "u1")
            data = {"posts": [{"post_id": p.post_id, "content": p.content, "likes": p.likes}
                               for p in POSTS_DB if p.user_id == uid]}

        latency = (time.perf_counter() - start) * 1000 + 3   # gRPC is faster (binary + HTTP/2)
        json_bytes  = len(json.dumps(data).encode())
        proto_bytes = int(json_bytes * 0.65)  # Protobuf ~35% smaller
        return {"data": data, "latency_ms": round(latency, 2),
                "bytes_sent": proto_bytes, "format": "protobuf"}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_rest_vs_graphql_vs_grpc():
    print("=" * 65)
    print("REST vs GraphQL vs gRPC")
    print("Use case: Fetch user profile with recent posts")
    print("=" * 65)

    # ── REST ──────────────────────────────────
    print("\n[1] REST API")
    print("─" * 50)
    rest = RESTClient()
    result = rest.fetch_user_with_posts("u1")
    print(f"  GET /users/u1      → {result['user']}")
    print(f"  GET /users/u1/posts → {len(result['posts'])} posts (includes extra fields)")
    print(f"  Requests made : {result['requests_made']}")
    print(f"  Total bytes   : {result['total_bytes']} B (includes un-needed fields)")

    # ── GraphQL ───────────────────────────────
    print("\n\n[2] GraphQL API")
    print("─" * 50)
    gql = GraphQLClient()
    query = GraphQLQuery(
        query="""
          query GetUserFeed($userId: ID!) {
            user(id: $userId) {
              name
              followers
              posts {
                content
                likes
              }
            }
          }
        """,
        variables={"userId": "u1"}
    )
    result = gql.execute(query)
    print(f"  Single request to /graphql")
    print(f"  Response: {json.dumps(result['data'], indent=4)[:300]}")
    print(f"  Requests made : {result['requests_made']}  (no N+1!)")
    print(f"  Total bytes   : {result['total_bytes']} B  (only requested fields)")

    # ── gRPC ──────────────────────────────────
    print("\n\n[3] gRPC API")
    print("─" * 50)
    grpc = GRPCClient()
    r1 = grpc.call("GetUser", {"user_id": "u1"})
    r2 = grpc.call("GetUserPosts", {"user_id": "u1"})
    print(f"  UserService.GetUser(u1)      → latency: {r1['latency_ms']:.1f}ms  size: {r1['bytes_sent']} B (protobuf)")
    print(f"  UserService.GetUserPosts(u1) → latency: {r2['latency_ms']:.1f}ms  size: {r2['bytes_sent']} B (protobuf)")
    print(f"  Protocol : binary (Protobuf) — ~35% smaller than JSON")
    print(f"  Transport: HTTP/2 multiplexed — both calls over one connection")

    # ── Comparison Table ──────────────────────
    print("\n\n[4] COMPARISON SUMMARY")
    print("─" * 50)
    print(f"  {'Aspect':<22} {'REST':<22} {'GraphQL':<22} {'gRPC'}")
    print(f"  {'─'*85}")
    rows = [
        ("Protocol",         "HTTP/1.1+",           "HTTP/1.1+",           "HTTP/2"),
        ("Payload format",   "JSON",                "JSON",                "Protobuf (binary)"),
        ("Payload size",     "Baseline",            "Smaller (exact)",     "~35% smaller"),
        ("Over-fetching",    "❌ Common",            "✅ Never",            "✅ Defined schema"),
        ("N+1 problem",      "❌ Yes",               "✅ Solved",           "✅ N/A (RPC style)"),
        ("Streaming",        "❌ Limited (SSE)",     "✅ Subscriptions",    "✅ Full streaming"),
        ("Type safety",      "❌ No schema (OpenAPI)","⚠  Schema optional", "✅ Proto IDL"),
        ("Learning curve",   "✅ Low",               "⚠  Medium",          "⚠  Medium"),
        ("Browser support",  "✅ Native",            "✅ Native",           "⚠  Needs grpc-web"),
        ("Use case",         "Public APIs, CRUD",   "Mobile, complex UI",  "Internal microsvcs"),
    ]
    for aspect, rest_v, gql_v, grpc_v in rows:
        print(f"  {aspect:<22} {rest_v:<22} {gql_v:<22} {grpc_v}")

    # ── When to use ───────────────────────────
    print("\n\n[5] DECISION GUIDE")
    print("─" * 50)
    print("  Use REST when:    Public API, external partners, simple CRUD")
    print("  Use GraphQL when: Mobile clients (bandwidth), BFF layer, complex object graphs")
    print("  Use gRPC when:    Internal microservice RPC, streaming, performance-critical paths")


if __name__ == "__main__":
    demonstrate_rest_vs_graphql_vs_grpc()
