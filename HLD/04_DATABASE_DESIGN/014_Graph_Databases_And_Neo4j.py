"""
GRAPH DATABASES AND NEO4J
===========================

Problem Statement:
Highly connected data — social networks, recommendation systems, fraud
detection, knowledge graphs — requires traversing many relationships.
In a relational DB, each hop requires a JOIN; with 5+ hops the query
becomes exponentially slow. Graph databases store relationships as
first-class citizens, making traversal O(relationships) not O(table_size).

Graph Model:
  Nodes: entities (Person, Movie, Product)
  Edges (Relationships): connections (KNOWS, ACTED_IN, BOUGHT)
  Properties: key-value pairs on nodes and edges

Cypher Query Language (Neo4j):
  MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name
  MATCH (p:Person)-[:BOUGHT]->(product)<-[:BOUGHT]-(other)
      WHERE p.id = 123 AND other.id <> 123
      RETURN DISTINCT other.id, COUNT(product) AS common

Graph Algorithms:
  BFS/DFS          : shortest path, friend-of-friend
  PageRank         : importance scoring (used in Google)
  Community Detection: Louvain, Label Propagation
  Centrality       : who is most connected? (betweenness, closeness)
  Similarity       : Jaccard, cosine (recommendations)

Use Cases:
  Social Networks  : Facebook friends, Twitter follows
  Recommendations  : "People who bought X also bought Y"
  Fraud Detection  : Rings of fraudulent accounts sharing properties
  Knowledge Graphs : Google Knowledge Panel, Wikidata
  Network Topology : routing, data center connectivity
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
import time


class GraphDirection(Enum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH     = "both"


@dataclass
class Node:
    node_id    : str
    labels     : List[str]   # e.g., ["Person"], ["Movie"]
    properties : Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.node_id == other.node_id

    def __str__(self):
        label = self.labels[0] if self.labels else "Node"
        name  = self.properties.get("name", self.node_id)
        return f"({label}: {name})"


@dataclass
class Edge:
    edge_id    : str
    src_id     : str
    dst_id     : str
    label      : str           # relationship type, e.g., KNOWS, ACTED_IN
    properties : Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.edge_id)


# ─────────────────────────────────────────────
# GRAPH DATABASE
# ─────────────────────────────────────────────

class GraphDB:
    """
    Adjacency list graph database.
    O(1) node lookup, O(k) relationship traversal where k = degree.
    """

    def __init__(self):
        self._nodes        : Dict[str, Node] = {}
        self._edges        : Dict[str, Edge] = {}
        self._adj_out      : Dict[str, List[Edge]] = defaultdict(list)  # src→[edges]
        self._adj_in       : Dict[str, List[Edge]] = defaultdict(list)  # dst→[edges]
        self._edge_counter = 0
        self.query_count   = 0

    def create_node(self, node_id: str, labels: List[str], **props) -> Node:
        node = Node(node_id, labels, props)
        self._nodes[node_id] = node
        return node

    def create_edge(self, src_id: str, dst_id: str, label: str, **props) -> Edge:
        self._edge_counter += 1
        edge_id = f"e{self._edge_counter}"
        edge    = Edge(edge_id, src_id, dst_id, label, props)
        self._edges[edge_id] = edge
        self._adj_out[src_id].append(edge)
        self._adj_in[dst_id].append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: str, label: str = None,
                       direction: GraphDirection = GraphDirection.OUTGOING) -> List[Node]:
        self.query_count += 1
        neighbors = []
        if direction in (GraphDirection.OUTGOING, GraphDirection.BOTH):
            for edge in self._adj_out.get(node_id, []):
                if label is None or edge.label == label:
                    n = self._nodes.get(edge.dst_id)
                    if n:
                        neighbors.append(n)
        if direction in (GraphDirection.INCOMING, GraphDirection.BOTH):
            for edge in self._adj_in.get(node_id, []):
                if label is None or edge.label == label:
                    n = self._nodes.get(edge.src_id)
                    if n:
                        neighbors.append(n)
        return neighbors

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def nodes_by_label(self, label: str) -> List[Node]:
        return [n for n in self._nodes.values() if label in n.labels]


# ─────────────────────────────────────────────
# GRAPH ALGORITHMS
# ─────────────────────────────────────────────

class GraphAlgorithms:
    def __init__(self, db: GraphDB):
        self.db = db

    def bfs_shortest_path(self, start_id: str, end_id: str,
                           rel_label: str = None) -> Optional[List[str]]:
        """BFS shortest path between two nodes. O(V+E)."""
        if start_id == end_id:
            return [start_id]
        visited   = {start_id}
        parent    : Dict[str, str] = {}
        queue     = deque([start_id])

        while queue:
            current = queue.popleft()
            for neighbor in self.db.get_neighbors(current, rel_label):
                nid = neighbor.node_id
                if nid not in visited:
                    visited.add(nid)
                    parent[nid] = current
                    if nid == end_id:
                        # Reconstruct path
                        path = []
                        cur  = end_id
                        while cur in parent:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(start_id)
                        return list(reversed(path))
                    queue.append(nid)
        return None   # not reachable

    def friends_of_friends(self, person_id: str, depth: int = 2) -> Set[str]:
        """BFS up to given depth, return all discovered nodes."""
        visited  = {person_id}
        frontier = {person_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                for neighbor in self.db.get_neighbors(nid, "KNOWS"):
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        next_frontier.add(neighbor.node_id)
            frontier = next_frontier
        visited.discard(person_id)
        return visited

    def page_rank(self, iterations: int = 10, damping: float = 0.85) -> Dict[str, float]:
        """Simplified PageRank. O(iterations × E)."""
        nodes = list(self.db._nodes.keys())
        N     = len(nodes)
        if N == 0:
            return {}
        pr    = {nid: 1.0 / N for nid in nodes}

        for _ in range(iterations):
            new_pr: Dict[str, float] = {}
            for nid in nodes:
                incoming = self.db.get_neighbors(nid, direction=GraphDirection.INCOMING)
                rank_sum  = 0.0
                for src_node in incoming:
                    out_degree = len(self.db.get_neighbors(src_node.node_id))
                    if out_degree > 0:
                        rank_sum += pr[src_node.node_id] / out_degree
                new_pr[nid] = (1 - damping) / N + damping * rank_sum
            pr = new_pr

        # Normalize
        total = sum(pr.values())
        return {nid: v / total for nid, v in pr.items()}

    def collaborative_filter(self, user_id: str, item_label: str = "Product",
                              rel_label: str = "BOUGHT") -> List[Tuple[str, int]]:
        """
        Collaborative filtering: find items bought by people who bought
        the same items as user. (User-based collaborative filtering)
        """
        # Items this user bought
        my_items = {n.node_id for n in self.db.get_neighbors(user_id, rel_label)}
        if not my_items:
            return []

        # Find similar users (bought at least one same item)
        similar_users: Set[str] = set()
        for item_id in my_items:
            for buyer in self.db.get_neighbors(item_id, rel_label, GraphDirection.INCOMING):
                if buyer.node_id != user_id:
                    similar_users.add(buyer.node_id)

        # Items bought by similar users that I haven't bought
        candidate_counts: Dict[str, int] = defaultdict(int)
        for sim_user_id in similar_users:
            for item in self.db.get_neighbors(sim_user_id, rel_label):
                if item.node_id not in my_items and item_label in item.labels:
                    candidate_counts[item.node_id] += 1

        return sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)

    def detect_fraud_ring(self, node_id: str, shared_attr: str,
                           max_depth: int = 3) -> Set[str]:
        """
        Fraud detection: find accounts sharing properties (phone, IP, device)
        within N hops.
        """
        fraud_ring = {node_id}
        queue      = deque([(node_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            node = self.db.get_node(current)
            if not node:
                continue
            # Find nodes sharing the attribute
            attr_val = node.properties.get(shared_attr)
            if attr_val:
                for other in self.db._nodes.values():
                    if (other.node_id not in fraud_ring and
                            other.properties.get(shared_attr) == attr_val):
                        fraud_ring.add(other.node_id)
                        queue.append((other.node_id, depth + 1))
        return fraud_ring


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_graph_db():
    print("=" * 65)
    print("GRAPH DATABASES AND NEO4J")
    print("=" * 65)

    db = GraphDB()

    # ── Social Network ────────────────────────
    print("\n[1] SOCIAL NETWORK GRAPH")
    print("─" * 55)
    people = {
        "alice": {"name": "Alice", "age": 30},
        "bob":   {"name": "Bob",   "age": 28},
        "carol": {"name": "Carol", "age": 35},
        "dave":  {"name": "Dave",  "age": 25},
        "eve":   {"name": "Eve",   "age": 32},
        "frank": {"name": "Frank", "age": 29},
    }
    for pid, props in people.items():
        db.create_node(pid, ["Person"], **props)

    # Create relationships
    connections = [
        ("alice", "bob",   "KNOWS"),
        ("alice", "carol", "KNOWS"),
        ("bob",   "dave",  "KNOWS"),
        ("carol", "eve",   "KNOWS"),
        ("dave",  "frank", "KNOWS"),
        ("eve",   "frank", "KNOWS"),
    ]
    for src, dst, label in connections:
        db.create_edge(src, dst, label, since="2020")

    print(f"  Graph: {db.node_count()} nodes, {db.edge_count()} edges")

    algo = GraphAlgorithms(db)

    # Shortest path
    path = algo.bfs_shortest_path("alice", "frank", "KNOWS")
    if path:
        path_names = [db.get_node(nid).properties["name"] for nid in path]
        print(f"\n  Shortest path Alice → Frank: {' → '.join(path_names)} ({len(path)-1} hops)")

    # Friends of friends
    fof = algo.friends_of_friends("alice", depth=2)
    fof_names = [db.get_node(nid).properties["name"] for nid in fof]
    print(f"\n  Alice's friends-of-friends (depth=2): {fof_names}")

    # ── PageRank ──────────────────────────────
    print("\n\n[2] PAGERANK (influence score)")
    print("─" * 55)
    ranks = algo.page_rank(iterations=20)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    for nid, rank in sorted_ranks:
        node = db.get_node(nid)
        name = node.properties.get("name", nid)
        bar  = "█" * int(rank * 100)
        print(f"  {name:<10} {rank:.4f}  {bar}")

    # ── Recommendation Engine ─────────────────
    print("\n\n[3] COLLABORATIVE FILTERING — PRODUCT RECOMMENDATIONS")
    print("─" * 55)
    # Add products
    for pid in ["laptop", "mouse", "keyboard", "monitor", "headset", "webcam"]:
        db.create_node(pid, ["Product"], name=pid.title())

    # Add users
    for uid in ["u1", "u2", "u3", "u4"]:
        db.create_node(uid, ["User"], name=f"User{uid[1:]}")

    # Purchase history
    purchases = [
        ("u1", ["laptop", "mouse", "keyboard"]),
        ("u2", ["laptop", "mouse", "headset"]),
        ("u3", ["laptop", "keyboard", "monitor"]),
        ("u4", ["mouse", "webcam"]),
    ]
    for uid, prods in purchases:
        for prod in prods:
            db.create_edge(uid, prod, "BOUGHT")

    recs = algo.collaborative_filter("u1", item_label="Product")
    print("  u1 bought: laptop, mouse, keyboard")
    print("  Recommendations (items bought by similar users):")
    for item_id, count in recs:
        node = db.get_node(item_id)
        print(f"    {node.properties['name']}: endorsed by {count} similar user(s)")

    # ── Fraud Detection ───────────────────────
    print("\n\n[4] FRAUD RING DETECTION")
    print("─" * 55)
    # Create accounts sharing phone number (fraud ring)
    accounts = [
        ("acct_1", {"name": "John Doe",   "phone": "555-1234", "email": "j1@mail.com"}),
        ("acct_2", {"name": "Jane Doe",   "phone": "555-1234", "email": "j2@mail.com"}),  # same phone!
        ("acct_3", {"name": "Jim Doe",    "phone": "555-5678", "email": "j3@mail.com"}),
        ("acct_4", {"name": "Jack Doe",   "phone": "555-1234", "email": "j4@mail.com"}),  # same phone!
        ("acct_5", {"name": "Jenny Doe",  "phone": "555-9999", "email": "j5@mail.com"}),
    ]
    for aid, props in accounts:
        db.create_node(aid, ["Account"], **props)

    fraud_ring = algo.detect_fraud_ring("acct_1", shared_attr="phone")
    print(f"  Starting from acct_1 (phone=555-1234):")
    print(f"  Fraud ring found: {len(fraud_ring)} accounts share the same phone")
    for aid in fraud_ring:
        node = db.get_node(aid)
        if node:
            print(f"    {node.properties['name']} ({node.properties['phone']})")

    # ── Comparison ────────────────────────────
    print("\n\n[5] GRAPH DB vs RELATIONAL FOR CONNECTED DATA")
    print("─" * 55)
    print("  Friends-of-friends query (depth=5, 1M users):")
    rows = [
        ("SQL (5 JOINs)", "SELECT ...", "Minutes (exponential joins)"),
        ("Neo4j/Graph",   "MATCH p=(a)-[:KNOWS*1..5]->(b)", "Milliseconds (pointer chasing)"),
    ]
    for db_type, query, perf in rows:
        print(f"  {db_type:<20} {query:<40} {perf}")

    print("\n  Why graph is faster:")
    print("  SQL: JOIN scans entire friends table for each hop")
    print("  Graph: follows pointers directly (index-free adjacency)")
    print("  Constant O(k) per hop regardless of graph size")

    print(f"\n  DB stats: {db.node_count()} nodes, {db.edge_count()} edges, "
          f"{db.query_count} queries executed")


if __name__ == "__main__":
    demonstrate_graph_db()
