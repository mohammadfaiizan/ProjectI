"""
GOOGLE DOCS — Collaborative Real-time Document Editing
=======================================================

FUNCTIONAL REQUIREMENTS:
- Multiple users edit the same document simultaneously
- Changes appear in near real-time for all collaborators
- Offline editing with automatic sync on reconnect
- Version history and named checkpoints
- Comments, suggestions, and resolved threads

NON-FUNCTIONAL REQUIREMENTS:
- < 100 ms latency for operation propagation (same region)
- 1 B documents, 50 M concurrent editors
- Eventual consistency — all clients converge to same state
- Conflict-free merging of concurrent edits

CONSISTENCY PROBLEM:
  User A deletes char at position 5.
  User B (concurrently) inserts 'X' at position 7.
  After A's delete, B's position 7 is now position 6.
  B's operation must be *transformed* before applying.

OPERATIONAL TRANSFORMATION (OT):
  - Document = sequence of characters
  - Operation = Insert(pos, char) | Delete(pos)
  - transform(op1, op2) → op1' such that applying op1 then op2
    gives the same result as applying op2 then op1'
  - Server acts as total-order arbiter (one central OT server)

CRDT ALTERNATIVE (Conflict-free Replicated Data Type):
  - Each character gets a globally unique position ID (fractional indexing)
  - No transformation needed — operations commute by design
  - Used by: Figma, Notion, Loom
  - Tradeoff: tombstones grow forever; harder to implement rich text

ARCHITECTURE:
  Client ──WebSocket──▶ Collab Server ──▶ OT Engine
                              │                │
                        Redis Pub/Sub    Document Store
                              │                │
                         Other Clients   Version History

COLLABORATION FLOW:
  1. Client opens doc → gets current snapshot + version_id
  2. Client makes local edit → apply immediately (optimistic)
  3. Client sends Operation(version=N, op=...) to server
  4. Server queues ops for document
  5. Server transforms op against any ops committed since version N
  6. Server applies transformed op → new version N+1
  7. Server broadcasts (version=N+1, op_transformed) to all clients
  8. Clients apply received op (already transformed by server)

PRESENCE:
  - Cursor positions broadcast via WebSocket
  - User colour assigned on join; shown as coloured caret
  - "X is editing section 2" awareness
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import threading


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class OpType(Enum):
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"


@dataclass
class Op:
    op_type: OpType
    position: int
    char: str = ""        # only for INSERT
    op_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_id: str = ""
    base_version: int = 0

    def __repr__(self):
        if self.op_type == OpType.INSERT:
            return f"Insert(pos={self.position}, char='{self.char}')"
        elif self.op_type == OpType.DELETE:
            return f"Delete(pos={self.position})"
        return f"Retain(pos={self.position})"


# ---------------------------------------------------------------------------
# Operational Transformation
# ---------------------------------------------------------------------------

class OTEngine:
    """
    Transform operations against each other so concurrent edits converge.

    Core property: apply(apply(doc, op1), transform(op2, op1)) ==
                   apply(apply(doc, op2), transform(op1, op2))
    """

    @staticmethod
    def transform(op: Op, against: Op) -> Op:
        """Transform `op` as if `against` was already applied."""
        if op.op_type == OpType.INSERT:
            return OTEngine._transform_insert(op, against)
        elif op.op_type == OpType.DELETE:
            return OTEngine._transform_delete(op, against)
        return op

    @staticmethod
    def _transform_insert(ins: Op, against: Op) -> Op:
        pos = ins.position
        if against.op_type == OpType.INSERT:
            # If against inserted at or before us, shift right
            if against.position <= pos:
                pos += 1
        elif against.op_type == OpType.DELETE:
            # If against deleted before us, shift left
            if against.position < pos:
                pos -= 1
        return Op(OpType.INSERT, pos, ins.char, ins.op_id, ins.client_id, ins.base_version)

    @staticmethod
    def _transform_delete(dlt: Op, against: Op) -> Op:
        pos = dlt.position
        if against.op_type == OpType.INSERT:
            if against.position <= pos:
                pos += 1
        elif against.op_type == OpType.DELETE:
            if against.position < pos:
                pos -= 1
            elif against.position == pos:
                # Both deleted same char — become no-op (retain)
                return Op(OpType.RETAIN, pos, op_id=dlt.op_id, client_id=dlt.client_id)
        return Op(OpType.DELETE, pos, op_id=dlt.op_id, client_id=dlt.client_id,
                  base_version=dlt.base_version)


# ---------------------------------------------------------------------------
# Document State
# ---------------------------------------------------------------------------

@dataclass
class DocumentVersion:
    version: int
    op: Op
    timestamp: float = field(default_factory=time.time)


class Document:
    """A text document as a mutable list of characters."""

    def __init__(self, doc_id: str, title: str, initial_content: str = ""):
        self.doc_id = doc_id
        self.title = title
        self._chars: List[str] = list(initial_content)
        self._version = 0
        self._history: List[DocumentVersion] = []
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        return self._version

    @property
    def content(self) -> str:
        return "".join(self._chars)

    def apply(self, op: Op) -> bool:
        """Apply a pre-transformed operation. Returns True if successful."""
        with self._lock:
            if op.op_type == OpType.INSERT:
                pos = max(0, min(op.position, len(self._chars)))
                self._chars.insert(pos, op.char)
                self._version += 1
                self._history.append(DocumentVersion(self._version, op))
                return True
            elif op.op_type == OpType.DELETE:
                if 0 <= op.position < len(self._chars):
                    self._chars.pop(op.position)
                    self._version += 1
                    self._history.append(DocumentVersion(self._version, op))
                    return True
            elif op.op_type == OpType.RETAIN:
                return True  # no-op (conflict neutralised)
        return False

    def ops_since(self, version: int) -> List[Op]:
        return [dv.op for dv in self._history if dv.version > version]

    def snapshot(self) -> Tuple[str, int]:
        with self._lock:
            return self.content, self._version


# ---------------------------------------------------------------------------
# Collaboration Server
# ---------------------------------------------------------------------------

@dataclass
class ClientSession:
    client_id: str
    user_id: str
    doc_id: str
    cursor_pos: int = 0
    colour: str = "#FF5733"
    joined_at: float = field(default_factory=time.time)
    pending_ops: List[Op] = field(default_factory=list)


class CollabServer:
    """
    Central OT server — applies total ordering to operations.
    In production: one server per document shard; ZooKeeper for leader election.
    """

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._sessions: Dict[str, ClientSession] = {}        # client_id → session
        self._doc_clients: Dict[str, List[str]] = defaultdict(list)  # doc_id → client_ids
        self._broadcast_log: List[Dict] = []   # simulated broadcast bus

    def create_document(self, title: str, initial: str = "") -> Document:
        doc = Document(str(uuid.uuid4()), title, initial)
        self._documents[doc.doc_id] = doc
        return doc

    def join(self, doc_id: str, user_id: str,
             colour: str = "#3498DB") -> Tuple[Optional[ClientSession], str, int]:
        """Returns (session, snapshot_content, snapshot_version)."""
        if doc_id not in self._documents:
            return None, "", 0
        client_id = str(uuid.uuid4())[:8]
        session = ClientSession(client_id, user_id, doc_id, colour=colour)
        self._sessions[client_id] = session
        self._doc_clients[doc_id].append(client_id)
        content, version = self._documents[doc_id].snapshot()
        return session, content, version

    def submit_op(self, client_id: str, op: Op) -> Optional[Op]:
        """
        Receive client op, transform against concurrent ops, apply, broadcast.
        Returns the final (transformed) op or None on failure.
        """
        session = self._sessions.get(client_id)
        if not session:
            return None
        doc = self._documents.get(session.doc_id)
        if not doc:
            return None

        # Get all ops committed since client's base version
        concurrent_ops = doc.ops_since(op.base_version)

        # Transform op against each concurrent op
        transformed = op
        for concurrent in concurrent_ops:
            transformed = OTEngine.transform(transformed, concurrent)

        # Apply transformed op to document
        doc.apply(transformed)

        # Broadcast to all other clients in the document
        self._broadcast(session.doc_id, client_id, transformed, doc.version)

        return transformed

    def _broadcast(self, doc_id: str, sender_id: str, op: Op, version: int):
        entry = {
            "doc_id": doc_id,
            "sender": sender_id,
            "op": op,
            "version": version,
            "timestamp": time.time(),
        }
        self._broadcast_log.append(entry)
        # In real system: push to WebSocket for each client except sender
        recipients = [c for c in self._doc_clients[doc_id] if c != sender_id]
        entry["recipients"] = recipients

    def update_cursor(self, client_id: str, pos: int) -> None:
        if client_id in self._sessions:
            self._sessions[client_id].cursor_pos = pos

    def leave(self, client_id: str) -> None:
        session = self._sessions.pop(client_id, None)
        if session:
            clients = self._doc_clients.get(session.doc_id, [])
            if client_id in clients:
                clients.remove(client_id)

    def presence(self, doc_id: str) -> List[Dict]:
        result = []
        for cid in self._doc_clients.get(doc_id, []):
            s = self._sessions.get(cid)
            if s:
                result.append({
                    "client_id": cid,
                    "user_id": s.user_id,
                    "cursor": s.cursor_pos,
                    "colour": s.colour,
                })
        return result


# ---------------------------------------------------------------------------
# Comment System
# ---------------------------------------------------------------------------

@dataclass
class Comment:
    comment_id: str
    doc_id: str
    author_id: str
    anchor_start: int     # character position when comment was created
    anchor_end: int
    text: str
    replies: List[Dict] = field(default_factory=list)
    resolved: bool = False
    created_at: float = field(default_factory=time.time)


class CommentService:
    def __init__(self):
        self._comments: Dict[str, List[Comment]] = defaultdict(list)

    def add_comment(self, doc_id: str, author_id: str,
                    start: int, end: int, text: str) -> Comment:
        c = Comment(
            comment_id=str(uuid.uuid4())[:8],
            doc_id=doc_id,
            author_id=author_id,
            anchor_start=start,
            anchor_end=end,
            text=text,
        )
        self._comments[doc_id].append(c)
        return c

    def reply(self, comment_id: str, author_id: str, text: str) -> bool:
        for comments in self._comments.values():
            for c in comments:
                if c.comment_id == comment_id:
                    c.replies.append({"author": author_id, "text": text,
                                      "ts": time.time()})
                    return True
        return False

    def resolve(self, comment_id: str) -> bool:
        for comments in self._comments.values():
            for c in comments:
                if c.comment_id == comment_id:
                    c.resolved = True
                    return True
        return False

    def open_comments(self, doc_id: str) -> List[Comment]:
        return [c for c in self._comments[doc_id] if not c.resolved]


# ---------------------------------------------------------------------------
# Version History / Named Checkpoints
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    name: str
    version: int
    snapshot: str
    created_by: str
    created_at: float = field(default_factory=time.time)


class VersionHistory:
    def __init__(self):
        self._checkpoints: Dict[str, List[Checkpoint]] = defaultdict(list)

    def create_checkpoint(self, doc: Document, name: str, user_id: str) -> Checkpoint:
        content, version = doc.snapshot()
        cp = Checkpoint(name=name, version=version, snapshot=content,
                        created_by=user_id)
        self._checkpoints[doc.doc_id].append(cp)
        return cp

    def list_checkpoints(self, doc_id: str) -> List[Checkpoint]:
        return sorted(self._checkpoints[doc_id], key=lambda c: c.version, reverse=True)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_basic_ot():
    print("\n=== 1. Operational Transformation — Concurrent Edits ===")
    doc = Document("doc1", "Test Doc", "Hello World")
    print(f"Initial: '{doc.content}' (version {doc.version})")

    # User A: insert '!' at position 11 (end)
    op_a = Op(OpType.INSERT, 11, "!", client_id="alice", base_version=0)

    # User B: insert ' Beautiful' at position 5 (concurrent, base_version=0)
    op_b = Op(OpType.INSERT, 5, " Beautiful", client_id="bob", base_version=0)

    # Server applies A first
    doc.apply(op_a)
    print(f"After A's insert: '{doc.content}'")

    # Transform B's op against A's op before applying
    op_b_transformed = OTEngine.transform(op_b, op_a)
    print(f"B's op transformed: {op_b_transformed}")
    doc.apply(op_b_transformed)
    print(f"After B's insert (transformed): '{doc.content}'")


def demonstrate_2_delete_conflict():
    print("\n=== 2. OT — Concurrent Deletes (Same Position) ===")
    doc = Document("doc2", "Delete Test", "abcdef")
    print(f"Initial: '{doc.content}'")

    # Both users delete 'c' (position 2) simultaneously
    op_a = Op(OpType.DELETE, 2, client_id="alice", base_version=0)
    op_b = Op(OpType.DELETE, 2, client_id="bob", base_version=0)

    doc.apply(op_a)
    print(f"After A deletes pos 2: '{doc.content}'")

    # Transform B's delete against A's delete
    op_b_prime = OTEngine.transform(op_b, op_a)
    print(f"B's op after transform: {op_b_prime}")
    doc.apply(op_b_prime)   # becomes RETAIN (no-op)
    print(f"After B's transformed op: '{doc.content}' (no double-delete)")


def demonstrate_3_collab_server():
    print("\n=== 3. Collaboration Server — Multi-user Session ===")
    server = CollabServer()
    doc = server.create_document("Meeting Notes", "Agenda:\n")
    print(f"Document created: '{doc.title}', id={doc.doc_id[:8]}...")

    # Alice joins
    session_a, content_a, ver_a = server.join(doc.doc_id, "alice", "#E74C3C")
    print(f"Alice joined at version {ver_a}, sees: {repr(content_a)}")

    # Bob joins
    session_b, content_b, ver_b = server.join(doc.doc_id, "bob", "#2ECC71")
    print(f"Bob joined at version {ver_b}, sees: {repr(content_b)}")

    # Alice types "1. Review PR\n" at position 8
    op1 = Op(OpType.INSERT, 8, "1", client_id=session_a.client_id, base_version=ver_a)
    transformed1 = server.submit_op(session_a.client_id, op1)
    print(f"\nAlice inserted '1' at pos 8 → version {doc.version}")
    print(f"Document: {repr(doc.content)}")

    # Bob inserts 'X' at position 8 concurrently (base_version still 0)
    op2 = Op(OpType.INSERT, 8, "X", client_id=session_b.client_id, base_version=ver_b)
    transformed2 = server.submit_op(session_b.client_id, op2)
    print(f"\nBob inserted 'X' at pos 8 (base={ver_b}) → version {doc.version}")
    print(f"Document: {repr(doc.content)}")

    presence = server.presence(doc.doc_id)
    print(f"\nPresence: {len(presence)} users")
    for p in presence:
        print(f"  {p['user_id']} (colour={p['colour']}, cursor={p['cursor']})")


def demonstrate_4_version_history():
    print("\n=== 4. Version History & Checkpoints ===")
    server = CollabServer()
    history = VersionHistory()
    doc = server.create_document("Proposal", "Introduction\n")

    session_a, _, ver = server.join(doc.doc_id, "alice")

    # Make some edits
    for i, char in enumerate("Draft v1"):
        op = Op(OpType.INSERT, 12 + i, char,
                client_id=session_a.client_id, base_version=doc.version)
        server.submit_op(session_a.client_id, op)

    cp1 = history.create_checkpoint(doc, "First Draft", "alice")
    print(f"Checkpoint '{cp1.name}' at version {cp1.version}: {repr(cp1.snapshot)}")

    # More edits
    op = Op(OpType.DELETE, 12, client_id=session_a.client_id, base_version=doc.version)
    server.submit_op(session_a.client_id, op)
    cp2 = history.create_checkpoint(doc, "After Edit", "alice")
    print(f"Checkpoint '{cp2.name}' at version {cp2.version}: {repr(cp2.snapshot)}")

    checkpoints = history.list_checkpoints(doc.doc_id)
    print(f"\nAll checkpoints (newest first):")
    for cp in checkpoints:
        print(f"  v{cp.version}: '{cp.name}' by {cp.created_by}")


def demonstrate_5_comments():
    print("\n=== 5. Comments & Threads ===")
    server = CollabServer()
    comments = CommentService()
    doc = server.create_document("Article", "The quick brown fox jumps")

    # Alice comments on "quick brown"
    c1 = comments.add_comment(doc.doc_id, "alice", 4, 15, "Can we use a more vivid adjective?")
    print(f"Comment added: '{c1.text}'")
    print(f"Anchored to chars {c1.anchor_start}-{c1.anchor_end}")

    # Bob replies
    comments.reply(c1.comment_id, "bob", "How about 'swift golden'?")
    print(f"Bob replied. Total replies: {len(c1.replies)}")

    # Alice resolves
    comments.resolve(c1.comment_id)
    open_count = len(comments.open_comments(doc.doc_id))
    print(f"Comment resolved. Open comments remaining: {open_count}")


if __name__ == "__main__":
    demonstrate_1_basic_ot()
    demonstrate_2_delete_conflict()
    demonstrate_3_collab_server()
    demonstrate_4_version_history()
    demonstrate_5_comments()
